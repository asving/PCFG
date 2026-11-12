"""Circuit analysis of a network trained on coupled brackets with hidden types (vpa.py).

  python vpa_validate.py runs/NAME

Two pictures predict the same exact loss. In the CHAINED one a bracket reads its prior from its parent's
finished posterior (one read per nesting level) and a parent reads a closed child's finished summary. In the
AT-ONCE one a position gathers the evidence of many enclosing brackets, or of a whole closed subtree,
directly. Four measurements:

  1 loss       by position class, beside the observers that cut the down or the up recursion short
  2 probes     the quantities of the exact filter, after each block; and whether a position holds the
               evidence of each enclosing bracket separately
  3 attention  by how the attended position is related to the reading one
  4 knockouts  remove what is read from content tokens beyond some relation (attention weights kept);
               a chained network keeps its loss, an at-once network falls towards the matching observer

Relation of an attended position j to a reading position i: let A be the innermost bracket that encloses
both. m = how many levels i's bracket is below A; r = how many levels j's bracket is below A.
  m = 0, r = 0   i's own bracket            m >= 1, r = 0   an enclosing bracket's own tokens, m levels up
  m = 0, r >= 1  inside a closed child      m >= 1, r >= 1  inside a closed bracket hanging off an ancestor
"""
import json, sys
import numpy as np
import torch
import vpa
from model import GPT, trace, load
import algorithm
from vpa_train import cut_name

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
proc = vpa.make(args["config"])
K, S, D, C = proc.K, proc.S, proc.D, proc.C
N, T, L, H = 512, args["T"], args["layers"], args["heads"]
toks, _ = proc.sample(N, T, np.random.default_rng(2026))
preds, ref = proc.filter(toks)
st = algorithm.run(proc, toks)
xin, depth, own, index = st["read"], ref["depth"], st["own"], st["index"]
content, is_open, is_close = proc.is_content(xin), proc.is_open(xin), proc.is_close(xin)
model = load(run, args, proc)
torch.manual_seed(123)
blank = GPT(proc.VOCAB, T, args["d"], L, H).eval()
J = np.arange(T)

# --- structure: the path of open positions at each position, and facts about the bracket just closed ---------
path = -np.ones((N, T, D + 1), dtype=np.int64)               # path[b, t, q] = open position of the enclosing bracket at depth q
cur = -np.ones((N, D + 1), dtype=np.int64)
cur[:, 0] = 0
kids = np.zeros((N, D + 2), dtype=np.int64)                   # closed children so far, per open bracket
closed_had_kids = np.zeros((N, T), dtype=bool)                # the position has just read the close of a bracket that had children
rows = np.arange(N)
for t in range(T):
    o, c = is_open[:, t], is_close[:, t]
    cur[rows[o], depth[o, t]] = t
    kids[rows[o], depth[o, t]] = 0
    closed_had_kids[c, t] = kids[rows[c], depth[c, t] + 1] > 0
    kids[rows[c], depth[c, t]] += 1
    path[:, t] = np.where(np.arange(D + 1)[None] <= depth[:, t][:, None], cur, -1)
assert (np.take_along_axis(path, depth[..., None], -1)[..., 0] == own).all()


def relation(sl):
    """m, r [b, T, T] for reading position i (axis 1) and attended position j (axis 2)."""
    p, d = path[sl], depth[sl]
    same = (p[:, :, None, :] == p[:, None, :, :]) & (p[:, :, None, :] >= 0)
    common = np.cumprod(same, -1).sum(-1)
    return d[:, :, None] - (common - 1), d[:, None, :] - (common - 1)


# --- the network ---------------------------------------------------------------------------------------------
resid, blank_resid, lps = [[] for _ in range(L + 1)], [[] for _ in range(L + 1)], []
NA = 192                                                      # sequences used for attention and knockouts
cats = [("own bracket", lambda m, r: (m == 0) & (r == 0)), ("parent's own tokens", lambda m, r: (m == 1) & (r == 0)),
        ("grandparent's own", lambda m, r: (m == 2) & (r == 0)), ("3+ levels up, own", lambda m, r: (m >= 3) & (r == 0)),
        ("inside closed children", lambda m, r: (m == 0) & (r == 1)), ("deeper inside them", lambda m, r: (m == 0) & (r >= 2)),
        ("closed, off the parent", lambda m, r: (m == 1) & (r >= 1)), ("closed, off higher up", lambda m, r: (m >= 2) & (r >= 1))]
att = np.zeros((L, H, len(cats) + 3))                          # + BOS, own '#', the position just before own '#'
att_mod = np.zeros((L, H, K + 1))                              # content tokens by (their depth - mine) mod K; then everything else
avail_mod = np.zeros(K + 1)
avail = np.zeros(len(cats) + 3)
nq = 0
for a in range(0, N, 64):
    sl = slice(a, a + 64)
    lg, rs, A = trace(model, torch.from_numpy(xin[sl]))
    lps.append(torch.log_softmax(lg, -1).double().numpy())
    brs = trace(blank, torch.from_numpy(xin[sl]), want_attn=False)[1]
    for l in range(L + 1):
        resid[l].append(rs[l].numpy())
        blank_resid[l].append(brs[l].numpy())
    if a < NA:
        m, r = relation(sl)
        q = (depth[sl] >= 3)                                  # reading positions at depth >= 3
        nq += q.sum()
        bos = np.broadcast_to(J == 0, m.shape)
        vis = np.broadcast_to(J[None, :] <= J[:, None], m.shape)
        keys = [f(m, r) & ~bos for _, f in cats] + [bos, J[None, None, :] == own[sl][:, :, None], J[None, None, :] == (own[sl] - 1)[:, :, None]]
        dd = (depth[sl][:, None, :] - depth[sl][:, :, None]) % K
        ck = np.broadcast_to(content[sl][:, None, :], dd.shape)
        for ci, k in enumerate([ck & (dd == c_) for c_ in range(K)] + [~ck]):
            kq = (k & q[:, :, None]).astype(np.float32)
            avail_mod[ci] += ((kq * vis).sum(-1) / np.maximum(J, 1)[None, :]).sum()
            for l in range(L):
                att_mod[l, :, ci] += np.einsum("bhij,bij->h", A[l].numpy(), kq)
        for ci, k in enumerate(keys):
            kq = (k & q[:, :, None]).astype(np.float32)
            avail[ci] += ((kq * vis).sum(-1) / np.maximum(J, 1)[None, :]).sum()
            for l in range(L):
                att[l, :, ci] += np.einsum("bhij,bij->h", A[l].numpy(), kq)
att /= nq
att_mod /= nq
lp = np.concatenate(lps)
resid = [np.concatenate(r_) for r_ in resid]
blank_resid = [np.concatenate(r_) for r_ in blank_resid]
exact = preds["exact"]
net = algorithm.excess(exact, lp)
print(f"{run}: {args['config']}, {L} layers, d = {args['d']}; {N} sequences of {T}")

# --- 1. loss ---------------------------------------------------------------------------------------------------
cells = {"all": np.ones_like(depth, dtype=bool), "depth >= 5": depth >= 5,
         "new bracket, depth >= 3": (depth >= 3) & (index <= 1) & ~is_close,
         "just read a close": is_close, "close of a bracket with children": closed_had_kids}
print(f"\n=== 1. LOSS  (excess over exact Bayes, mnat/token)")
print(f"{'':30s}" + "".join(f"{c:>34s}" for c in cells))
print(f"{'share of positions':30s}" + "".join(f"{m_.mean():34.3f}" for m_ in cells.values()))
order = ["prior", "ignore"] + [("down", h) for h in range(6)] + [("up", u) for u in range(4)]
for n in order:
    kl = algorithm.excess(exact, preds[n])
    print(f"{cut_name(n):30s}" + "".join(f"{1e3 * kl[m_].mean():34.1f}" for m_ in cells.values()))
rigid_kl = algorithm.excess(exact, vpa.rigid(args["config"]).filter(toks)[0]["exact"])   # the coupling with its noise removed
print(f"{'coupling taken as exact':30s}" + "".join(f"{1e3 * rigid_kl[m_].mean():34.1f}" for m_ in cells.values())
      + "   (every bracket's type fixed by the root's and its depth; no bracket structure needed beyond depth)")
print(f"{'THE NETWORK':30s}" + "".join(f"{1e3 * net[m_].mean():34.1f}" for m_ in cells.values()))
print("the network by depth 0-8:   " + "  ".join(f"{1e3 * net[depth == d].mean():.1f}" for d in range(D + 1)))
for n in (("down", 2), ("down", 4), ("up", 1)):
    print(f"{cut_name(n) + ' by depth:':28s}" + "  ".join(f"{1e3 * algorithm.excess(exact, preds[n])[depth == d].mean():.1f}" for d in range(D + 1)))

# --- 2. probes -------------------------------------------------------------------------------------------------
test = np.zeros((N, T), dtype=bool)
test[int(0.75 * N):] = True
cen = lambda v: v - v.mean(-1, keepdims=True)
clip = lambda v: np.clip(v, -8, 8)


def r2(X, Y, sel):
    Y = Y.reshape(N, T, -1).astype(float)
    Xf, Yf, Xt, Yt = X[sel & ~test].astype(float), Y[sel & ~test], X[sel & test].astype(float), Y[sel & test]
    mx, my = Xf.mean(0), Yf.mean(0)
    W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
    return 1 - ((Yt - my - (Xt - mx) @ W) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()


at = lambda Y, pos: np.take_along_axis(Y.reshape(N, T, -1), np.maximum(pos, 0)[..., None], 1)
evid = cen(ref["ll"] + ref["msg"])                            # all the evidence a bracket has about its own type
up1 = own - 1                                                 # the parent's last position before this bracket opened
up2 = np.take_along_axis(own, np.maximum(up1, 0), 1) - 1
up3 = np.take_along_axis(own, np.maximum(up2, 0), 1) - 1
everywhere = np.ones((N, T), dtype=bool)
targets = [("depth", depth, everywhere),
           ("distance back to own '#'", J[None] - own, depth >= 1),
           (f"state belief per type ({K * S})", ref["state"], index >= 1),
           ("own-token evidence", clip(cen(ref["ll"])), index >= 1),
           ("message of the child just closed", clip(ref["close_msg"]), is_close),
           ("summed messages of closed children", clip(cen(ref["msg"])), depth >= 0),
           ("type prior of the bracket (log)", clip(cen(np.log(ref["prior"]))), depth >= 1),
           ("type posterior (log)", clip(cen(np.log(ref["w"] + 1e-12))), everywhere),
           ("evidence of the PARENT at the open", clip(at(evid, up1)), depth >= 1),
           ("evidence of the GRANDPARENT", clip(at(evid, up2)), depth >= 2),
           ("evidence 3 levels up", clip(at(evid, up3)), depth >= 3)]
print(f"\n=== 2. PROBES  (R^2 on held-out sequences; residual after the embedding, then after each block)")
print(f"{'':38s}" + "".join(f"{('emb' if l == 0 else f'blk{l}'):>7s}" for l in range(L + 1)) + "   best untrained   from the bracket's own exact quantities")
own_summary = np.concatenate([clip(cen(np.log(ref["prior"]))), clip(cen(ref["ll"])), clip(cen(ref["msg"])), clip(cen(np.log(ref["w"] + 1e-12)))], -1)
for name, Y, sel in targets:
    vals = [r2(resid[l], Y, sel) for l in range(L + 1)]
    ctrl = max(r2(blank_resid[l], Y, sel) for l in range(L + 1))
    implied = r2(own_summary, Y, sel) if "evidence of" in name or "levels up" in name else float("nan")
    print(f"{name:38s}" + "".join(f"{v:7.2f}" for v in vals) + f"   {ctrl:7.2f}" + (f"   {implied:22.2f}" if implied == implied else ""))

# --- 3. attention ----------------------------------------------------------------------------------------------
names = [c for c, _ in cats] + ["BOS", "(own '#')", "(just before own '#')"]
print(f"\n=== 3. ATTENTION  (share of each head's attention, reading positions at depth >= 3; the last two columns are single positions)")
short = ["own", "parent", "grandp", "3+ up", "child", "deeper", "off par", "off high", "BOS", "own #", "pre-#"]
print(f"{'':16s}" + "".join(f"{s_:>9s}" for s_ in short))
print(f"{'if spread evenly':16s}" + "".join(f"{v / nq:9.2f}" for v in avail))
for l in range(L):
    for h in range(H):
        print(f"{f'block {l + 1} head {h}':16s}" + "".join(f"{v:9.2f}" for v in att[l, h]))

print(f"\nthe same attention, split by how much deeper the attended CONTENT token is than the reading position, mod {K}")
print(f"(with this coupling a bracket's likely type is its ancestor's type plus the depth difference, so the classes are type classes)")
print(f"{'':16s}" + "".join(f"{f'{c_} mod {K}':>9s}" for c_ in range(K)) + f"{'not content':>13s}")
print(f"{'if spread evenly':16s}" + "".join(f"{v / nq:9.2f}" for v in avail_mod[:K]) + f"{avail_mod[K] / nq:13.2f}")
for l in range(L):
    for h in range(H):
        print(f"{f'block {l + 1} head {h}':16s}" + "".join(f"{v:9.2f}" for v in att_mod[l, h, :K]) + f"{att_mod[l, h, K]:13.2f}")

# --- 4. knockouts ----------------------------------------------------------------------------------------------
def values(x):
    """What every position offers each head in each block (the value vectors), on an ordinary forward pass."""
    out = []
    with torch.no_grad():
        h, rot = model.embed(x), model.rotation(x)
        for blk in model.blocks:
            out.append(blk.qkv_heads(h, rot)[2])
            h = blk(h, rot)
    return out


# the same sequences with every content token redrawn at random: brackets, depths and positions unchanged
RESAMPLES = 8
gen = torch.Generator().manual_seed(7)
neutral = []                                                  # per chunk, per block: the average value over the redraws
for a in range(0, NA, 64):
    x = torch.from_numpy(xin[a:a + 64])
    cont = torch.from_numpy(content[a:a + 64])
    acc = None
    for _ in range(RESAMPLES):
        vs = values(torch.where(cont, torch.randint(0, C, x.shape, generator=gen), x))
        acc = vs if acc is None else [p + q for p, q in zip(acc, vs)]
    neutral.append([p / RESAMPLES for p in acc])


def ablated(pairs, blocks):
    """Log-probabilities when, in the given blocks, what a reading position i takes from an attended position j
    is replaced, for the pairs [b, T, T] marked True, by what j offers on average when the content tokens of the
    sequence are redrawn. Attention weights are kept. Position, depth and bracket structure survive in the
    replacement; anything that depended on which content tokens were emitted does not."""
    out = []
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    with torch.no_grad():
        for ci, a in enumerate(range(0, NA, 64)):
            x = torch.from_numpy(xin[a:a + 64])
            F_ = torch.from_numpy(pairs[a:a + 64].astype(np.float32))[:, None]
            h, rot = model.embed(x), model.rotation(x)
            for l, blk in enumerate(model.blocks, 1):
                B_, _, Dm = h.shape
                q, k, v = blk.qkv_heads(h, rot)
                w_ = (q @ k.transpose(-1, -2) / (Dm // blk.heads) ** 0.5).masked_fill(future, -1e9).softmax(-1)
                o = w_ @ v
                if l in blocks:
                    o = o + (w_ * F_) @ (neutral[ci][l - 1] - v)
                h = h + blk.o(o.transpose(1, 2).reshape(B_, T, Dm))
                h = h + blk.mlp(blk.ln2(h))
            out.append(torch.log_softmax(model.unembed(model.lnf(h)), -1))
    return torch.cat(out).double().numpy()


M, R = [np.concatenate(v) for v in zip(*[relation(slice(a, a + 64)) for a in range(0, NA, 64)])]
sub = slice(0, NA)
key_content = np.broadcast_to(content[sub][:, None, :], M.shape)
nested = np.broadcast_to((depth[sub] >= 1)[:, :, None], M.shape)
pre = (J[None, None, :] == (own[sub] - 1)[:, :, None]) & nested
ownopen = (J[None, None, :] == own[sub][:, :, None]) & nested
kcells = {"all": np.ones_like(depth[sub], dtype=bool), "depth >= 5": depth[sub] >= 5, "just read a close": is_close[sub]}
everything = np.broadcast_to(J[None, None, :] >= 0, M.shape)
tests = [("everything (all pairs)", everything, "prior"),
         ("content 1+ levels up the path", (M >= 1) & key_content, ("down", 0)),
         ("content 2+ levels up the path", (M >= 2) & key_content, ("down", 1)),
         ("content 3+ levels up the path", (M >= 3) & key_content, ("down", 2)),
         ("content 1+ levels inside closed brackets", (R >= 1) & key_content, ("up", 0)),
         ("content 2+ levels inside closed brackets", (R >= 2) & key_content, ("up", 1)),
         ("the position just before own '#'", pre, None), ("own '#'", ownopen, None)]
allb = set(range(1, L + 1))
print(f"\n=== 4. KNOCKOUTS  ({NA} sequences; excess loss, mnat/token)")
print("For the reading-attended pairs named, what is read is replaced by what the attended position offers on average when the")
print("sequence's content tokens are redrawn (structure kept); attention weights are kept. If the information reaches a position")
print("only level by level, removing the direct read costs nothing. If it is read directly, the loss rises towards the observer")
print("that lacks it.")
print(f"{'':46s}" + "".join(f"{c:>20s}" for c in kcells) + "   | all positions, one block at a time: " + " ".join(f"blk{l}" for l in range(1, L + 1)))
print(f"{'the network, nothing removed':46s}" + "".join(f"{1e3 * net[sub][m_].mean():20.1f}" for m_ in kcells.values()))
print(f"{'check: no pairs marked':46s}" + "".join(f"{1e3 * algorithm.excess(exact[sub], ablated(np.zeros_like(pre), allb))[m_].mean():20.1f}" for m_ in kcells.values()))
for label, pairs, obs in tests:
    v = algorithm.excess(exact[sub], ablated(pairs, allb))
    per = [1e3 * algorithm.excess(exact[sub], ablated(pairs, {l})).mean() for l in range(1, L + 1)]
    print(f"{'remove ' + label:46s}" + "".join(f"{1e3 * v[m_].mean():20.1f}" for m_ in kcells.values()) + "   | " + " ".join(f"{p:6.1f}" for p in per))
    if obs is not None:
        f_ = algorithm.excess(exact[sub], preds[obs][sub])
        print(f"{'   observer: ' + cut_name(obs):46s}" + "".join(f"{1e3 * f_[m_].mean():20.1f}" for m_ in kcells.values()))
