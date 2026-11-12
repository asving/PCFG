"""How much of the path's relabelling does a network hold?

With kinds of open token (vpa.py), a bracket's likely type is the root's type carried through the relabellings
of the kinds on its path, in order. This asks, block by block, which parts of that composition a linear readout
can name at positions inside a bracket:

  the whole relabelling from the root        (one of up to K! classes)
  its sign                                   (even or odd permutation: the abelian part of a non-commuting pair)
  the relabelling over the last m levels     (the kinds of the m nearest enclosing opens, composed)

  python vpa_relabel.py runs/NAME

Accuracy on held-out sequences; chance is the share of the commonest class.
"""
import itertools, json, sys
import numpy as np
import torch
import vpa
from model import trace, load
import algorithm

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
proc = vpa.make(args["config"])
K, D = proc.K, proc.D
N, T, L = 768, args["T"], args["layers"]
toks, _ = proc.sample(N, T, np.random.default_rng(2030))
xin = algorithm.read_tokens(proc, toks)
is_open, is_close = proc.is_open(xin), proc.is_close(xin)
depth = np.cumsum(is_open.astype(int) - is_close.astype(int), 1)
model = load(run, args, proc)
resid = [[] for _ in range(L + 1)]
for a in range(0, N, 128):
    rs = trace(model, torch.from_numpy(xin[a:a + 128]), want_attn=False)[1]
    for l in range(L + 1):
        resid[l].append(rs[l].numpy())
resid = [np.concatenate(r) for r in resid]

hard = np.eye(K, dtype=np.int64)[proc.PIs.argmax(-1)].argmax(-1)         # [kind, type] -> favoured child type
perms = list(itertools.permutations(range(K)))
sign = {p: (-1) ** sum(p[i] > p[j] for i in range(K) for j in range(i + 1, K)) for p in perms}
rows = np.arange(N)
kinds = np.zeros((N, D + 1), dtype=np.int64)                               # kind of the open at each depth of the current path
M = 4
whole = np.zeros((N, T, K), dtype=np.int64)
last = np.zeros((M + 1, N, T, K), dtype=np.int64)
for t in range(T):
    o = is_open[:, t]
    kinds[rows[o], depth[o, t]] = xin[o, t] - proc.OPEN0
    d = depth[:, t]
    for m in list(range(1, M + 1)) + [D]:
        g = np.broadcast_to(np.arange(K), (N, K)).copy()                  # apply the opens in path order, shallowest of the span first
        for q in range(1, D + 1):
            use = (q <= d) & (q > d - m)
            g = np.where(use[:, None], hard[kinds[:, q][:, None], g], g)
        if m == D:
            whole[:, t] = g
        else:
            last[m, :, t] = g
code = lambda g: (g * K ** np.arange(K)).sum(-1)
test = np.zeros((N, T), dtype=bool)
test[int(0.75 * N):] = True


def accuracy(labels, sel):
    classes = np.unique(labels[sel])
    Y = (labels[..., None] == classes).astype(float)
    out = []
    for l in range(L + 1):
        X = resid[l].astype(float)
        Xf, Yf = X[sel & ~test], Y[sel & ~test]
        mx, my = Xf.mean(0), Yf.mean(0)
        W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
        out.append((((X[sel & test] - mx) @ W + my).argmax(-1) == Y[sel & test].argmax(-1)).mean())
    return out, Y[sel & test].mean(0).max(), len(classes)


lookup = {sum(p[i] * K ** i for i in range(K)): (1 if sign[p] > 0 else 0) for p in perms}
sgn = np.vectorize(lookup.get)(code(whole))
print(f"{run}: {args['config']}, {L} layers; {N} sequences; share of held-out positions where a linear readout names the class")
print(f"{'':52s}{'classes':>8s}" + "".join(f"{('emb' if l == 0 else f'blk{l}'):>7s}" for l in range(L + 1)) + "   chance")
for dmin in (3, 5):
    sel = depth >= dmin
    for name, lab in [("whole relabelling from the root", code(whole)), ("its sign", sgn)] + [(f"relabelling over the last {m} level{'s' if m > 1 else ''}", code(last[m])) for m in range(1, M + 1) if m <= dmin]:
        acc, base, n = accuracy(lab, sel)
        print(f"{f'depth >= {dmin}: ' + name:52s}{n:>8d}" + "".join(f"{v:7.2f}" for v in acc) + f"   {base:6.2f}")
