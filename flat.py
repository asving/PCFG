"""Is a component's belief readable from one layer?

Two different questions, both answered here on ONE component process alone (a flat sequence, no brackets):
  --layers 1   can a one-layer network hold the belief at all?
  --layers 4   in a deeper network, is the belief readable from a single block's output, or only from
               all blocks together (the RRXOR case in Shai et al.)?
The belief is read linearly from the residual stream. Mess3 is the control that should pass (Piotrowski et al.,
arXiv 2502.01954) and RRXOR the control that should fail (Shai et al., arXiv 2405.15943).

  python flat.py NAME [--steps 20000] [--device mps] [--layers 1]

Prints excess loss over Bayes (mnat/token), then R^2 of a linear readout of the belief (fit on 75% of
held-out sequences, scored on the rest) from the embedding, from the residual after attention, and
from the residual after the MLP, each beside the same readout from an untrained copy.
"""
import argparse, math, time
import numpy as np
import torch
import torch.nn.functional as F
from process import BracketProcess, mess3, tom_quantum, rrxor, fanizza
from model import GPT
import algorithm

COMPONENTS = {
    "mess3": lambda: mess3(0.05, 0.85),
    "mess3_noisy": lambda: mess3(0.15, 0.6),
    "mess3_anti": lambda: mess3(0.45, 0.85),
    "rrxor": lambda: rrxor(),
    "tomq_a": lambda: tom_quantum(1, 0.364),
    "tomq_b": lambda: tom_quantum(1, 0.839),
    "tomq_c": lambda: tom_quantum(1, 2.747),
    "tomq_paper": lambda: tom_quantum(1, 51 ** 0.5),      # the Bloch Walk setting reported for arXiv 2507.07432
    "clock": lambda: mess3(0.05, 0.75, shift=1),
    "fern": lambda: fanizza(2000, 0.49),
    "fern_b": lambda: fanizza(1.0, 0.49),
    "fern_c": lambda: fanizza(3.0, 0.45),
}

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--T", type=int, default=32)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--device", default="mps")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
proc = BracketProcess([COMPONENTS[args.name]()], p_open=0.0, p_close=0.0, name=args.name)
dev = torch.device(args.device)
model = GPT(proc.VOCAB, args.T, args.d, args.layers, args.heads).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
rng = np.random.default_rng(args.seed + 1)

N = 4096
toks, _ = proc.sample(N, args.T, np.random.default_rng(999))
ref = algorithm.run(proc, toks)
xin = torch.from_numpy(ref["read"])
belief = ref["state"][:, :, 0, :]                          # [N, T, S]


def excess():
    model.eval()
    with torch.no_grad():
        lp = torch.cat([F.log_softmax(model(xin[i:i + 1024].to(dev)), -1).cpu() for i in range(0, N, 1024)])
    model.train()
    return 1e3 * algorithm.excess(ref["pred"], lp.double().numpy()).mean()


bayes = -np.log(np.take_along_axis(ref["pred"], toks[..., None], -1)).mean()
print(f"{args.name}: {proc.comps[0].name}, {proc.C} tokens, {proc.S} states; entropy rate {bayes:.3f} nats; "
      f"{args.layers} layer(s), d={args.d}, {args.heads} heads, context {args.T}, on {args.device}", flush=True)
t0 = time.time()
for s in range(args.steps + 1):
    if s % (args.steps // 4) == 0:
        print(f"  step {s:6d}  excess loss {excess():8.2f} mnat/token   {time.time() - t0:.0f}s", flush=True)
    if s < args.steps:
        y = torch.from_numpy(proc.sample(args.batch, args.T, rng)[0])
        x = torch.cat([torch.full((args.batch, 1), proc.BOS), y[:, :-1]], 1).to(dev)
        loss = F.cross_entropy(model(x).reshape(-1, proc.VOCAB), y.to(dev).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()


@torch.no_grad()
def stages(m):
    """Residual after the embedding, after each block's attention, and after each block's MLP."""
    m = m.to("cpu").eval()
    T = xin.shape[1]
    h = m.tok(xin) + m.pos(torch.arange(T))
    out = [("embedding", h)]
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for l, blk in enumerate(m.blocks, 1):
        B, _, D = h.shape
        q, k, v = blk.qkv(blk.ln1(h)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, blk.heads, D // blk.heads).transpose(1, 2)
        a = (sh(q) @ sh(k).transpose(-1, -2) / (D // blk.heads) ** 0.5).masked_fill(future, -1e9).softmax(-1)
        h = h + blk.o((a @ sh(v)).transpose(1, 2).reshape(B, T, D))
        out.append((f"after attention {l}", h))
        h = h + blk.mlp(blk.ln2(h))
        out.append((f"after MLP {l}", h))
    return [(n, x.double().numpy()) for n, x in out]


def r2(X, Y):
    n = int(0.75 * N)
    Xf, Yf, Xt, Yt = X[:n].reshape(-1, X.shape[-1]), Y[:n].reshape(-1, Y.shape[-1]), \
        X[n:].reshape(-1, X.shape[-1]), Y[n:].reshape(-1, Y.shape[-1])
    mx, my = Xf.mean(0), Yf.mean(0)
    W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
    return 1 - ((Yt - my - (Xt - mx) @ W) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()


torch.manual_seed(123)
blank = GPT(proc.VOCAB, args.T, args.d, args.layers, args.heads)
print(f"  linear readout of the belief, R^2 (untrained copy in brackets):")
st, st0 = stages(model), stages(blank)
for (name, X), (_, X0) in zip(st, st0):
    print(f"    {name:18s} {r2(X, belief):6.3f}  ({r2(X0, belief):6.3f})", flush=True)
if args.layers > 1:                                        # as in Shai et al.: every block's output side by side
    cat = lambda ss: np.concatenate([X for n, X in ss if n.startswith("after MLP")], -1)
    print(f"    {'all blocks together':18s} {r2(cat(st), belief):6.3f}  ({r2(cat(st0), belief):6.3f})", flush=True)
