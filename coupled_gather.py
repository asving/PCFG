"""Coupled brackets: can one gather over the enclosing brackets stand in for the chain of reads?

Exactly, a bracket's type prior is a chain: the parent's posterior at the open, which used the
grandparent's posterior at its open, and so on (one read per nesting level; coupled.py). This prices an
observer that never chains. It adds up the evidence of all enclosing brackets with weights that fall off
geometrically with the number of levels up, and applies the coupling once:

  prior of a bracket = softmax( sum over ancestors m levels up of gamma^(m-1) * that ancestor's evidence ) . PI

gamma = 0 is the one-read observer; gamma = 1 treats every enclosing bracket as sharing one type. Every
such observer needs a single gather, whatever the depth.

  python coupled_gather.py [rho]
"""
import sys
import numpy as np
from coupled_train import CoupledProcess, HOPS
import algorithm

rho = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
proc = CoupledProcess(rho)
toks = proc.sample(1024, 128, np.random.default_rng(12345))      # the evaluation set of coupled_train.py


def gather(gamma):
    B, L = toks.shape
    K, S, D, C = proc.K, proc.S, proc.D, proc.C
    v = np.broadcast_to(proc.eta0, (B, D + 1, K, S)).copy()
    ll = np.zeros((B, D + 1, K))                              # a bracket's own evidence: its tokens and its closed children
    pool = np.zeros((B, D + 1, K))                            # weighted sum of the ancestors' evidence, as logits
    prior = np.full((B, D + 1, K), 1.0 / K)
    depth, rows = np.zeros(B, dtype=np.int64), np.arange(B)
    pred = np.zeros((B, L, proc.VOCAB))
    for t in range(L):
        top = v[rows, depth]
        lw = np.log(prior[rows, depth]) + ll[rows, depth]
        w = np.exp(lw - lw.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        po, pc = proc.struct_probs(depth)
        pred[:, t, :C] = (1 - po - pc)[:, None] * np.einsum("bk,bkx->bx", w, np.einsum("bks,kxs->bkx", top, proc.emit))
        pred[:, t, proc.OPEN0] = po
        pred[:, t, proc.CLOSE0:] = pc[:, None] * w
        x = toks[:, t]
        emt, opn, cls = proc.is_content(x), proc.is_open(x), proc.is_close(x)
        if emt.any():
            new = np.einsum("bks,kbsr->bkr", top[emt], proc.T[:, x[emt]])
            mass = np.maximum((new * proc.r).sum(-1), 1e-300)
            ll[rows[emt], depth[emt]] += np.log(mass)
            v[rows[emt], depth[emt]] = new / mass[..., None]
        if cls.any():
            ll[rows[cls], depth[cls] - 1] += np.log(proc.PI[:, x[cls] - proc.CLOSE0]).T
        if opn.any():
            r, d = rows[opn], depth[opn]
            g = ll[r, d] + gamma * pool[r, d]
            p = np.exp(g - g.max(-1, keepdims=True))
            pool[r, d + 1], prior[r, d + 1] = g, (p / p.sum(-1, keepdims=True)) @ proc.PI
            v[r, d + 1], ll[r, d + 1] = proc.eta0, 0.0
        depth = depth + opn - cls
    return pred


lad = proc.ladder(toks)
exact = lad[-1]
st = algorithm.run(proc, toks)
cells = {"all": np.ones_like(toks, dtype=bool), "new bracket, depth >= 2": (st["depth"] >= 2) & (st["index"] <= 2),
         "new bracket, depth >= 4": (st["depth"] >= 4) & (st["index"] <= 2), "just read a close": proc.is_close(st["read"])}
row = lambda name, p: print(f"{name:44s}" + "".join(f"{1e3 * algorithm.excess(exact, p)[m].mean():26.2f}" for m in cells.values()))
print(f"coupling rho = {rho}: a child keeps its parent's type with probability rho + (1 - rho) / 3; excess loss over exact, mnat/token")
print(f"{'':44s}" + "".join(f"{c:>26s}" for c in cells))
print("chained reads of the parent (coupled.py):")
for i, h in enumerate(HOPS):
    row(f"  {h} read{'s' if h != 1 else ''}", lad[i])
print("one gather over all enclosing brackets, weights gamma^(levels up - 1):")
print(f"  check: gamma = 0 against the one-read observer, max |difference| {np.abs(gather(0.0) - lad[1]).max():.1e}")
for gamma in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0):
    row(f"  gamma = {gamma}", gather(gamma))
