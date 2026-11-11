"""Test a trained network against the algorithm in algorithm.py.

  python validate.py runs/pilot_d128_L4_T128_s0

Four measurements. Each prints what the algorithm predicts next to what the network does.

  1 loss       excess loss over Bayes by position class, beside the cost of each shortcut
  2 probes     linear readout, after each block, of every quantity the algorithm computes
  3 attention  which earlier tokens each head reads, and how recency is counted across a child
  4 patching   after a close, which positions the parent's information is read from
"""
import json, sys
import numpy as np
import torch
from process import BracketProcess, BOS, OPEN, CLOSE0
from model import GPT, trace, load
import algorithm

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
proc = BracketProcess(p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
N, T, L = 768, args["T"], args["layers"]
toks, _ = proc.sample(N, T, np.random.default_rng(2026))
ref = algorithm.run(proc, toks)
xin, depth, own, prev, index = ref["read"], ref["depth"], ref["own"], ref["prev"], ref["index"]
ordinary, is_close = xin < 3, (xin >= CLOSE0) & (xin < BOS)
x = torch.from_numpy(xin)
model = load(run, args)
torch.manual_seed(123)
blank = GPT(T, args["d"], L, args["heads"]).eval()
J = np.arange(T)
CATS = ["self", "own #", "own toks", "own closes", "children", "ancestors", "others", "BOS"]


def categories(sl):
    """cat[b, i, j]: what position j is to position i (see CATS); -1 if j is after i."""
    o, d = own[sl], depth[sl]
    n = len(o)
    same = o[:, :, None] == o[:, None, :]
    anc = np.zeros((n, T, T), dtype=bool)
    tag = o.copy()
    for _ in range(proc.D):                                 # walk up: the parent's tag is the tag at own-1
        tag = np.where(tag >= 1, np.take_along_axis(o, np.maximum(tag - 1, 0), 1), -1)
        anc |= o[:, None, :] == tag[:, :, None]
    c = np.full((n, T, T), 6, dtype=np.int8)
    c[anc] = 5
    c[(J[None, None, :] > o[:, :, None]) & ~same] = 4
    c[same & is_close[sl][:, None, :]] = 3
    c[same & ordinary[sl][:, None, :]] = 2
    c[J[None, None, :] == o[:, :, None]] = 1
    c[(J[None, None, :] == 0) & (o[:, :, None] != 0)] = 7
    c[np.broadcast_to(np.eye(T, dtype=bool), c.shape)] = 0
    c[np.broadcast_to(J[None, :] > J[:, None], c.shape)] = -1
    return c


# --- one pass over the data: residuals, log-probs, attention statistics ---------------------------
conds = {"all positions at depth >= 1": depth >= 1,
         "just read a close, >= 3 own tokens behind the child": is_close & (depth >= 1) & (index >= 3)}
H = args["heads"]
share = {k: np.zeros((L, H, len(CATS))) for k in conds}
on_prev = {k: np.zeros((L, H)) for k in conds}
uniform = {k: np.zeros(len(CATS)) for k in conds}
# recency across a child: positions i, i-1 ordinary in one thread, and third-most-recent own token j3
p1 = np.take_along_axis(prev, np.maximum(prev, 0), 1)       # prev of prev
p2 = np.take_along_axis(prev, np.maximum(p1, 0), 1)
adjacent = ordinary & (prev == J[None] - 1) & np.take_along_axis(ordinary, np.maximum(prev, 0), 1)
ord_at = lambda idx: np.take_along_axis(ordinary, np.maximum(idx, 0), 1) & (idx >= 0)
contig = adjacent & (p1 == J[None] - 2) & ord_at(p1)                          # i-2 is an own token too
across = adjacent & (p1 == J[None] - 2) & np.take_along_axis(is_close, np.maximum(p1, 0), 1) & ord_at(p2)
j3 = np.where(contig, p1, np.where(across, p2, -1))
gap = (J[None] - 1) - j3 - 1                                # positions between j3 and i-1
groups = {"reference: three own tokens in a row": contig,
          "child of span <= 5 in between": across & (gap <= 5),
          "child of span >= 12 in between": across & (gap >= 12)}
a2 = {g: np.zeros((L, H)) for g in groups}
a3 = {g: np.zeros((L, H)) for g in groups}
resid, blank_resid, logp = [[] for _ in range(L + 1)], [[] for _ in range(L + 1)], []
for a in range(0, N, 128):
    sl = slice(a, a + 128)
    logits, rs, attn = trace(model, x[sl])
    logp.append(torch.log_softmax(logits, -1).double().numpy())
    for l in range(L + 1):
        resid[l].append(rs[l].double().numpy())
        blank_resid[l].append(trace(blank, x[sl], want_attn=False)[1][l].double().numpy())
    cat = categories(sl)
    for name, sel in conds.items():
        s = sel[sl]
        uniform[name] += np.stack([((cat == c) / np.maximum((cat >= 0).sum(-1, keepdims=True), 1))[s].sum()
                                   for c in range(len(CATS))])
        for l in range(L):
            at = attn[l].numpy()
            for c in range(len(CATS)):
                share[name][l, :, c] += np.stack([(at[:, h] * (cat == c))[s].sum() for h in range(H)])
            pv = np.maximum(prev[sl], 0)
            on_prev[name][l] += np.stack([np.take_along_axis(at[:, h], pv[..., None], 2)[..., 0][s].sum()
                                          for h in range(H)])
    for g, sel in groups.items():
        s = sel[sl]
        for l in range(L):
            at = attn[l].numpy()
            for h in range(H):
                back1 = np.broadcast_to(np.maximum(J - 1, 0)[None, :, None], (len(s), T, 1))
                a2[g][l, h] += np.take_along_axis(at[:, h], back1, 2)[..., 0][s].sum()
                a3[g][l, h] += np.take_along_axis(at[:, h], np.maximum(j3[sl], 0)[..., None], 2)[..., 0][s].sum()
resid = [np.concatenate(r) for r in resid]
blank_resid = [np.concatenate(r) for r in blank_resid]
logp = np.concatenate(logp)

# --- 1. loss --------------------------------------------------------------------------------------
print(f"=== 1. LOSS  ({run}, {N} x {T} held-out positions)")
print("Prediction: the network's row is near zero in every column. A shortcut would show in its own column.\n")
_, cs = algorithm.loss_table(proc, toks, extra=[("THE NETWORK", logp)])
kl = algorithm.excess(ref["pred"], logp)
print("the network by depth 0..6:   " + "  ".join(f"{1e3 * kl[depth == d].mean():.1f}" for d in range(7)))


# --- 2. probes ------------------------------------------------------------------------------------
def r2(X, Y, sel):
    Y = Y.reshape(N, T, -1).astype(float)
    fit, test = sel.copy(), sel.copy()
    fit[int(0.75 * N):] = False
    test[: int(0.75 * N)] = False
    Xf, Yf, Xt, Yt = X[fit], Y[fit], X[test], Y[test]
    mx, my = Xf.mean(0), Yf.mean(0)
    W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
    return 1 - ((Yt - my - (Xt - mx) @ W) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()


everywhere = np.ones((N, T), dtype=bool)
parent = np.take_along_axis(ref["belief"], np.maximum(own - 1, 0)[..., None], 1)
ev = ref["evidence"] - ref["evidence"].mean(-1, keepdims=True)
lw = np.clip(ref["logw"] - ref["logw"].mean(-1, keepdims=True), -6, 6)
targets = [
    ("1 depth", depth, everywhere, "early"),
    ("2 tag: distance back to own '#'", J[None] - own, depth >= 1, "after depth"),
    ("3 index: own tokens read", index, everywhere, "after tag"),
    ("4 state belief per type (9)", ref["state"], index >= 1, "middle"),
    ("5 evidence of the token just read (3)", ev, ordinary & (index >= 2), "middle"),
    ("6 type log-posterior (3)", lw, index >= 1, "late"),
    ("7 belief of the own bracket (9)", ref["belief"], everywhere, "last"),
    ("  belief of the PARENT bracket (9)", parent, depth >= 1, "absent"),
]
print(f"\n=== 2. PROBES  (R^2 on held-out sequences; residual after embedding, then after each block)")
print("Prediction: each quantity becomes readable in the order of the algorithm's steps; the parent's belief never does.\n")
print(f"{'':40s}{'expected':>12s}" + "".join(f"{('emb' if l == 0 else f'blk{l}'):>7s}" for l in range(L + 1))
      + "   best untrained")
for name, Y, sel, when in targets:
    vals = [r2(resid[l], Y, sel) for l in range(L + 1)]
    ctrl = max(r2(blank_resid[l], Y, sel) for l in range(L + 1))
    print(f"{name:40s}{when:>12s}" + "".join(f"{v:7.2f}" for v in vals) + f"   {ctrl:7.2f}")

# --- 3. attention ---------------------------------------------------------------------------------
print(f"\n=== 3. ATTENTION  (share of each head's attention by what the attended position is)")
print("Prediction: an even-spread head early (the count); later heads that read only the own bracket,\n"
      "also just after a close, when its tokens sit behind the closed child.\n")
for name, sel in conds.items():
    n = sel.sum()
    print(f"{name}  ({n} positions)")
    print(f"{'':12s}" + "".join(f"{c:>11s}" for c in CATS) + f"{'prev in thread':>16s}")
    print(f"{'even spread':12s}" + "".join(f"{u / n:11.2f}" for u in uniform[name]))
    for l in range(L):
        for h in range(H):
            print(f"blk{l + 1} head {h} " + "".join(f"{v / n:11.2f}" for v in share[name][l, h])
                  + f"{on_prev[name][l, h] / n:16.2f}")
    print()
close_cond = list(conds)[1]
readers = [(l, h) for l in range(L) for h in range(H)
           if share[close_cond][l, h, 2] / conds[close_cond].sum() >= 0.5]
print("How do the thread-reading heads count recency? Take a position i whose two most recent own tokens are\n"
      "i and i-1, and whose third is j3. Share = attention on j3 / (attention on i-1 + attention on j3).\n"
      "Counted in own tokens, a child between j3 and i-1 changes nothing: the share equals the reference.\n"
      "Counted in positions, a head whose attention decays with distance would drop the share towards 0.\n")
names = list(groups)
print(f"{'':14s}{'three own tokens in a row':>28s}{'child of span <= 5 between':>30s}{'child of span >= 12 between':>30s}")
print(f"{'positions':14s}" + "".join(f"{int(groups[g].sum()):>{w}d}" for g, w in zip(names, (28, 30, 30))))
for l, h in readers:
    obs = [a3[g][l, h] / max(a2[g][l, h] + a3[g][l, h], 1e-12) for g in names]
    print(f"blk{l + 1} head {h}   " + "".join(f"{v:>{w}.2f}" for v, w in zip(obs, (28, 30, 30))))

# --- 4. patching ----------------------------------------------------------------------------------
print(f"\n=== 4. PATCHING  (after a close, where is the parent's information read from?)")
print("Take a close at position q whose bracket has own tokens behind the child. Change the bracket's earlier\n"
      "own tokens (set R, all but the last one before the child, which sits at p). That changes what should be\n"
      "predicted at q. Then overwrite the residual after one block, at one set of positions, with its value in\n"
      "the changed run, and report the fraction of the full change in q's log-probabilities that this produces.\n"
      "Prediction: nothing travels through the child or through q until the last block. In the algorithm, p holds\n"
      "one thing that depends on R, its own evidence term (step 5), so p should carry a minority share once that\n"
      "step is done, and R the rest. The algorithm's own split is printed for comparison.\n")
rng = np.random.default_rng(7)
ev_b, ev_q = [], []
for b in range(N):
    qs = np.flatnonzero(is_close[b] & (depth[b] >= 1) & (index[b] >= 3) & ordinary[b, np.maximum(prev[b], 0)])
    if len(qs):
        ev_b.append(b)
        ev_q.append(qs[0])
ev_b, ev_q = np.array(ev_b), np.array(ev_q)
E = len(ev_b)
xc = xin[ev_b].copy()
p_pos = prev[ev_b, ev_q]
R = (own[ev_b] == own[ev_b, ev_q][:, None]) & ordinary[ev_b] & (J[None] < p_pos[:, None])
xk = np.where(R, (xc + rng.integers(1, 3, xc.shape)) % 3, xc)
masks = {"R: the changed own tokens": R,
         "p: last own token before the child": J[None] == p_pos[:, None],
         "everything else before q": (J[None] < ev_q[:, None]) & ~R & (J[None] != p_pos[:, None]),
         "q itself": J[None] == ev_q[:, None]}
xc_t, xk_t = torch.from_numpy(xc), torch.from_numpy(xk)
at_q = lambda logits: torch.log_softmax(logits, -1)[torch.arange(E), torch.from_numpy(ev_q)].double().numpy()
clean_logits, _, _ = trace(model, xc_t, want_attn=False)
corr_logits, corr_resid, _ = trace(model, xk_t, want_attn=False)
y0, y1 = at_q(clean_logits), at_q(corr_logits)
full = y1 - y0
toks_k = np.concatenate([xk[:, 1:], toks[ev_b, -1:]], 1)
bayes0 = ref["pred"][ev_b, ev_q]
bayes1 = algorithm.run(proc, toks_k)["pred"][np.arange(E), ev_q]
dbayes = np.log(bayes1) - np.log(bayes0)
print(f"{E} events. The change moves the Bayes law at q by {1e3 * (bayes0 * -dbayes).sum(-1).mean():.0f} mnat (KL); "
      f"the network's change in log-probabilities has cosine {(full * dbayes).sum() / np.linalg.norm(full) / np.linalg.norm(dbayes):.2f} with the Bayes change.\n")
# the same patches applied to the algorithm, after its evidence step
alt = algorithm.run(proc, toks_k)
rows_e = np.arange(E)
d_ev = alt["evidence"][rows_e, p_pos] - ref["evidence"][ev_b, p_pos]


def law(logw, eta):
    w = np.exp(logw - logw.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    po, pc = proc.struct_probs(depth[ev_b, ev_q])
    out = np.zeros((E, 7))
    out[:, :3] = (1 - po - pc)[:, None] * np.einsum("eks,kxs->ex", w[..., None] * eta, proc.T.sum(-1))
    out[:, OPEN] = po
    out[:, CLOSE0:] = pc[:, None] * w
    return np.log(out)


frac = lambda y: ((y - np.log(bayes0)) * dbayes).sum() / (dbayes ** 2).sum()
alg_p = frac(law(ref["logw"][ev_b, ev_q] + d_ev, ref["state"][ev_b, ev_q]))
alg_R = frac(law(alt["logw"][rows_e, ev_q] - d_ev, alt["state"][rows_e, ev_q]))
print(f"the algorithm, patched after its evidence step:   R {alg_R:.2f}   p {alg_p:.2f}   everything else 0   q 0\n")
print(f"{'the network, patched after':38s}" + "".join(f"{('embedding' if l == 0 else f'block {l}'):>11s}" for l in range(L)))
for name, mask in masks.items():
    row = []
    for l in range(L):
        logits, _, _ = trace(model, xc_t, patch={l: (torch.from_numpy(mask), corr_resid[l])}, want_attn=False)
        row.append(((at_q(logits) - y0) * full).sum() / (full ** 2).sum())
    print(f"{name:38s}" + "".join(f"{v:11.2f}" for v in row))
