"""Train a small causal transformer on the bracketed-Mess3 process and compare it with the exact filter.

Eval reports KL(Bayes || model) in nats/token (the expected excess loss), overall and split by
depth and by the positions where remembering a suspended frame matters. The same numbers for the
ablated observers in process.py are printed once for reference.

  python train.py --bench                 # timing only
  python train.py --steps 20000 --device mps
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from process import BracketProcess, VOCAB, CLOSE0

BOS = VOCAB


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o, self.heads = nn.Linear(d, 3 * d), nn.Linear(d, d), heads
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, T, d, layers, heads):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB + 1, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf, self.unembed = nn.LayerNorm(d), nn.Linear(d, VOCAB)

    def forward(self, x, return_hidden=False):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        hidden = [h]
        for b in self.blocks:
            h = b(h)
            hidden.append(h)
        logits = self.unembed(self.lnf(h))
        return (logits, hidden) if return_hidden else logits


def batch(proc, B, T, rng, device):
    toks, deps = proc.sample(B, T, rng)
    y = torch.from_numpy(toks)
    x = torch.cat([torch.full((B, 1), BOS), y[:, :-1]], 1)
    return x.to(device), y.to(device), toks, deps


def kl_to(pred, other_logp):
    """KL(pred || other) per position; pred [B,T,V] probabilities, other_logp log-probabilities."""
    logp = np.log(np.where(pred > 0, pred, 1.0))
    return (pred * (logp - other_logp)).sum(-1)


def split(kl, toks, deps, resumed, D):
    first = np.zeros_like(toks, dtype=bool)
    first[:, 1:] = toks[:, :-1] >= CLOSE0
    out = dict(all=kl.mean(), fresh=kl[resumed == 0].mean(), resumed=kl[resumed > 0].mean(),
               first_after_return=kl[first].mean())
    out["by_depth"] = [float(kl[deps == d].mean()) if (deps == d).any() else None for d in range(D + 1)]
    return {k: (float(v) if not isinstance(v, list) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--p_open", type=float, default=0.10)
    ap.add_argument("--p_close", type=float, default=0.12)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_seqs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=1e9)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    proc = BracketProcess(p_open=args.p_open, p_close=args.p_close, max_depth=args.max_depth)
    rng = np.random.default_rng(args.seed + 1)
    model = GPT(args.T, args.d, args.layers, args.heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    name = f"{args.tag}_d{args.d}_L{args.layers}_T{args.T}_s{args.seed}"

    def step():
        x, y, _, _ = batch(proc, args.batch, args.T, rng, dev)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        return loss

    if args.bench:
        t0 = time.time()
        for _ in range(20):
            proc.sample(args.batch, args.T, rng)
        t_data = (time.time() - t0) / 20
        for _ in range(5):
            step()
        n, t0 = 0, time.time()
        while time.time() - t0 < 8:
            step().item()
            n += 1
        dt = (time.time() - t0) / n
        print(f"{args.device:4s} d{args.d} L{args.layers} T{args.T} B{args.batch} "
              f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M: {1 / dt:6.1f} steps/s "
              f"({args.batch * args.T / dt / 1e6:.2f}M tok/s; data {t_data * 1e3:.0f} ms of {dt * 1e3:.0f} ms/step) "
              f"-> 50k steps in {50000 * dt / 3600:.1f} h", flush=True)
        return

    # fixed eval set with the exact filter and the ablated observers
    erng = np.random.default_rng(12345)
    ex, _, etoks, edeps = batch(proc, args.eval_seqs, args.T, erng, dev)
    bayes, _, resumed = proc.filter(etoks, "bayes")
    refs = {}
    for m in ["type_only", "amnesic"]:
        p = proc.filter(etoks, m)[0]
        refs[m] = split(kl_to(bayes, np.log(np.where(p > 0, p, 1e-300))), etoks, edeps, resumed, proc.D)
    fmt = lambda r: (f"all {r['all']:.4f} fresh {r['fresh']:.4f} resumed {r['resumed']:.4f} "
                     f"first-after-return {r['first_after_return']:.4f}")
    print("excess nats/token over Bayes for the ablated observers on the eval set:")
    for m, r in refs.items():
        print(f"  {m:10s} {fmt(r)}")
    log = dict(args=vars(args), refs=refs, history=[])
    os.makedirs("runs", exist_ok=True)
    t0 = time.time()
    for s in range(args.steps + 1):
        if s % args.eval_every == 0 or s == args.steps:
            model.eval()
            with torch.no_grad():
                lp = torch.cat([F.log_softmax(model(ex[i:i + 512]), -1).cpu() for i in range(0, len(ex), 512)])
            model.train()
            r = split(kl_to(bayes, lp.double().numpy()), etoks, edeps, resumed, proc.D)
            r.update(step=s, time=time.time() - t0)
            log["history"].append(r)
            json.dump(log, open(f"runs/{name}.json", "w"))
            print(f"[{name}] step {s:6d} {fmt(r)} | depth " +
                  " ".join(f"{v:.3f}" for v in r["by_depth"][:7] if v is not None) + f" | {r['time']:.0f}s", flush=True)
            if time.time() - t0 > args.time_limit:
                break
        if s < args.steps:
            step()
    torch.save(model.state_dict(), f"runs/{name}.pt")


if __name__ == "__main__":
    main()
