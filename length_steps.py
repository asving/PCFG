"""What breaks past the training length?

length_ood.py shows the loss rising with position beyond the 128 tokens a network was trained on. This
splits that loss and follows the mechanism out, by position band:

  A  the loss split into the open/close/content split and the rest (content token and close type)
  B  where the heads that read the own bracket put their attention, and how much every head gives to BOS
  C  probes fitted on the trained band and applied unchanged to later bands

  python length_steps.py runs/NAME [length]
"""
import json, sys
import numpy as np
import torch
from process import make
from model import trace, load
import algorithm

run = sys.argv[1]
LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 512
args = json.load(open(run + ".json"))["args"]
name, T, L, H = args.get("process", "mess3"), args["T"], args["layers"], args["heads"]
proc = make(name, p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
C, O, Z = proc.C, proc.OPEN0, proc.CLOSE0
model = load(run, args, proc)
N = 400
toks = proc.sample(N, LEN, np.random.default_rng(42))[0]
ref = algorithm.run(proc, toks, chunk=8)
depth, own, index = ref["depth"], ref["own"], ref["index"]
J = np.arange(LEN)
bands = [(a, min(a + T, LEN)) for a in range(0, LEN, T)]
inband = [np.broadcast_to((J >= a) & (J < b), depth.shape) for a, b in bands]
head = f"{'positions':34s}" + "".join(f"{f'{a}-{b - 1}':>10s}" for a, b in bands)

mass = np.zeros((L, H, len(bands), 4))                       # own bracket, BOS, elsewhere near / far; query positions at depth >= 1
bos_all = np.zeros((L, H, len(bands)))                       # attention to BOS from every position
cnt, cnt_all = np.zeros(len(bands)), np.zeros(len(bands))
resid, lps = [[] for _ in range(L + 1)], []
for a in range(0, N, 20):
    sl = slice(a, a + 20)
    lg, rs, attn = trace(model, torch.from_numpy(ref["read"][sl]))
    lps.append(torch.log_softmax(lg, -1).double().numpy())
    for l in range(L + 1):
        resid[l].append(rs[l].numpy())
    same = own[sl][:, :, None] == own[sl][:, None, :]
    isbos = np.broadcast_to(J == 0, same.shape)
    far = np.broadcast_to((J[:, None] - J[None, :]) >= T, same.shape)      # further back than any distance seen in training
    keys = [same & ~isbos, isbos, ~same & ~isbos & ~far, ~same & ~isbos & far]
    for bi, m in enumerate(inband):
        q = (depth[sl] >= 1) & m[sl]
        cnt[bi] += q.sum()
        cnt_all[bi] += m[sl].sum()
        for l in range(L):
            A = attn[l].numpy()
            for ci, k in enumerate(keys):
                mass[l, :, bi, ci] += np.einsum("bhij,bij->h", A, (k & q[:, :, None]).astype(np.float32))
            bos_all[l, :, bi] += (A[:, :, :, 0] * m[sl][:, None, :]).sum((0, 2))
mass /= cnt[None, None, :, None]
bos_all /= cnt_all[None, None, :]
lp = np.concatenate(lps)
resid = [np.concatenate(r) for r in resid]
print(f"{run}: {name}, trained on {T} tokens; {N} sequences of {LEN}")

# A. the loss, split
exact = ref["pred"]
tot = algorithm.excess(exact, lp)
inside = np.zeros(tot.shape)
for a, b in ((0, C), (O, Z), (Z, exact.shape[-1])):
    p = exact[..., a:b]
    m_ = p.sum(-1)
    cond = p / np.where(m_ > 0, m_, 1.0)[..., None]
    lq = lp[..., a:b] - np.logaddexp.reduce(lp[..., a:b], -1)[..., None]
    inside += m_ * (cond * (np.log(np.where(cond > 0, cond, 1.0)) - lq)).sum(-1)
print("\n=== A. EXCESS LOSS, mnat/token")
print(head)
print(f"{'total':34s}" + "".join(f"{1e3 * tot[m].mean():10.2f}" for m in inband))
print(f"{'  open / close / content split':34s}" + "".join(f"{1e3 * (tot - inside)[m].mean():10.2f}" for m in inband))
print(f"{'  content token and close type':34s}" + "".join(f"{1e3 * inside[m].mean():10.2f}" for m in inband))
po, pc = np.exp(lp[..., O:Z]).sum(-1), np.exp(lp[..., Z:]).sum(-1)
print(f"{'network P(close) at the root':34s}" + "".join(f"{pc[m & (depth == 0)].mean():10.3f}" for m in inband) + "   (exact: 0)")
print(f"{'network P(open) at depths 1-7':34s}" + "".join(f"{po[m & (depth >= 1) & (depth < proc.D)].mean():10.3f}" for m in inband) + f"   (exact: {proc.p_open})")

# B. attention
readers = [(l, h) for l in range(L) for h in range(H) if mass[l, h, 0, 0] > 0.5]
print(f"\n=== B. ATTENTION.  Reader heads (more than half on the own bracket in the trained band): "
      + ", ".join(f"block {l + 1} head {h}" for l, h in readers))
print(head)
for ci, cname in enumerate(["readers: own bracket", "readers: BOS", f"readers: elsewhere, < {T} back", f"readers: elsewhere, >= {T} back"]):
    print(f"{cname:34s}" + "".join(f"{np.mean([mass[l, h, bi, ci] for l, h in readers]):10.2f}" for bi in range(len(bands))))
far_share = [np.mean([(J[:, None] - J[None, :] >= T)[a:b].sum(-1) / np.maximum(J[a:b], 1)]) for a, b in bands]
print(f"{'(share of earlier positions >= ' + str(T) + ' back)':34s}" + "".join(f"{v:10.2f}" for v in far_share))
order = sorted(((bos_all[l, h, 0], l, h) for l in range(L) for h in range(H)), reverse=True)[:4]
for v, l, h in order:
    print(f"{f'block {l + 1} head {h}: share on BOS':34s}" + "".join(f"{bos_all[l, h, bi]:10.2f}" for bi in range(len(bands))))

# C. probes fitted on the trained band
lw = lambda w: np.clip(np.log(w + 1e-12) - np.log(w + 1e-12).mean(-1, keepdims=True), -6, 6)
targets = [("depth", depth.astype(float), np.ones_like(depth, dtype=bool)),
           ("distance back to own #", (J[None] - own).astype(float), depth >= 1),
           ("state belief per type", ref["state"], index >= 1), ("type log-posterior", lw(ref["w"]), index >= 1)]
nf = int(0.75 * N)
seq_fit = np.arange(N)[:, None] < nf
print(f"\n=== C. PROBES fitted on positions 0-{T - 1} of {nf} sequences, applied unchanged to the other sequences (R^2)")
print(f"{'':28s}{'block':>6s}" + "".join(f"{f'{a}-{b - 1}':>10s}" for a, b in bands))
for label, Y, sel in targets:
    Y = Y.reshape(N, LEN, -1)
    best = None
    for l in range(L + 1):
        X = resid[l].astype(float)
        f, t0 = sel & inband[0] & seq_fit, sel & inband[0] & ~seq_fit
        mx, my = X[f].mean(0), Y[f].mean(0)
        W = np.linalg.solve((X[f] - mx).T @ (X[f] - mx) + 1e-4 * f.sum() * np.eye(X.shape[-1]), (X[f] - mx).T @ (Y[f] - my))
        r2 = 1 - ((Y[t0] - my - (X[t0] - mx) @ W) ** 2).sum() / ((Y[t0] - Y[t0].mean(0)) ** 2).sum()
        if best is None or r2 > best[0]:
            best = (r2, l, mx, my, W)
    _, l, mx, my, W = best
    X = resid[l].astype(float)
    row = []
    for m in inband:
        t_ = sel & m & ~seq_fit
        row.append(1 - ((Y[t_] - my - (X[t_] - mx) @ W) ** 2).sum() / ((Y[t_] - Y[t_].mean(0)) ** 2).sum())
    print(f"{label:28s}{l:>6d}" + "".join(f"{v:10.2f}" for v in row))

# D. is it the distance to BOS? With rotary positions only differences of position enter, so shifting every
# position leaves the network's output unchanged. Shifting every position except BOS moves BOS further away
# and changes nothing else: same tokens, same distances between them, sequences of the trained length.
if getattr(model, "pos_rot", False):
    from model import ROT_PERIODS, POS_BASE
    M = len(ROT_PERIODS)
    short = proc.sample(1024, T, np.random.default_rng(43))[0]
    r = algorithm.run(proc, short)
    x = torch.from_numpy(r["read"])

    def forward(delta_bos, delta_all=0):
        B = len(x)
        period = torch.tensor(ROT_PERIODS, dtype=torch.float32)
        parts = [model.depth(x, model.depth_rot)[0][:, None, :, None] * (2 * torch.pi / period)] if model.depth_rot is not None \
            else [torch.zeros(B, 1, T, M)]
        pos = torch.arange(T, dtype=torch.float32) + delta_all
        pos[1:] += delta_bos
        freq = POS_BASE ** (-torch.arange(M, dtype=torch.float32) / M)
        parts.append((pos[None, None, :, None] * freq).expand(B, 1, T, M))
        ang = torch.cat(parts, -1)
        rot = (ang.cos(), ang.sin())
        with torch.no_grad():
            outs = []
            for a in range(0, B, 256):
                h = model.embed(x[a:a + 256])
                for blk in model.blocks:
                    h = blk(h, (rot[0][a:a + 256], rot[1][a:a + 256]))
                outs.append(torch.log_softmax(model.unembed(model.lnf(h)), -1))
        return 1e3 * algorithm.excess(r["pred"], torch.cat(outs).double().numpy()).mean()

    print(f"\n=== D. MOVE ONLY BOS  (1024 sequences of the trained length {T}; excess loss, mnat/token)")
    print(f"every position shifted by 300 (BOS too): {forward(0, 300):.2f}   (unshifted: {forward(0):.2f})")
    print("BOS moved further away by:   " + "".join(f"{d:>9d}" for d in (0, 16, 32, 64, 128, 256, 384)))
    print("excess loss                  " + "".join(f"{forward(d):9.2f}" for d in (0, 16, 32, 64, 128, 256, 384)))

# E. the causal check: hide from every position the keys further back than the training length, and see
# whether the loss comes back. The exact law still uses the whole sequence, so some loss is owed to the
# information hidden; "exact with the same window" gives that floor for the root's thread only roughly,
# so the comparison to read is against the unhidden network in the same band.
def windowed(keep_bos):
    Tn = LEN
    dist = torch.arange(Tn)[:, None] - torch.arange(Tn)[None, :]
    hide = (dist < 0) | (dist >= T)
    if keep_bos:
        hide[:, 0] = False
    outs = []
    with torch.no_grad():
        for a in range(0, N, 20):
            x = torch.from_numpy(ref["read"][a:a + 20])
            h, rot = model.embed(x), model.rotation(x)
            for blk in model.blocks:
                B, _, D = h.shape
                q, k, v = blk.qkv_heads(h, rot)
                att = (q @ k.transpose(-1, -2) / (D // blk.heads) ** 0.5).masked_fill(hide, -1e9).softmax(-1)
                h = h + blk.o((att @ v).transpose(1, 2).reshape(B, Tn, D))
                h = h + blk.mlp(blk.ln2(h))
            outs.append(torch.log_softmax(model.unembed(model.lnf(h)), -1))
    return algorithm.excess(exact, torch.cat(outs).double().numpy())


print(f"\n=== E. HIDE THE KEYS {T} OR MORE POSITIONS BACK  (same {N} sequences of {LEN}; excess loss over the exact law, mnat/token)")
print(head)
print(f"{'the network as it is':34s}" + "".join(f"{1e3 * tot[m].mean():10.2f}" for m in inband))
for label, kb in (("far keys hidden, BOS kept", True), ("far keys hidden, BOS too", False)):
    v = windowed(kb)
    print(f"{label:34s}" + "".join(f"{1e3 * v[m].mean():10.2f}" for m in inband))
    print(f"{'   at depth >= 1 only':34s}" + "".join(f"{1e3 * v[m & (depth >= 1)].mean():10.2f}" for m in inband))
print(f"{'the network as it is, depth >= 1':34s}" + "".join(f"{1e3 * tot[m & (depth >= 1)].mean():10.2f}" for m in inband))
