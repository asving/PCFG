"""Which earlier tokens does each attention head read? And which address tags are in the residual?

  python attention.py runs/pilot_d128_L4_T128_s0

Model position i has read tokens 0..i-1 (position 0 is BOS) and predicts token i. Its "own bracket"
is the one that will emit token i. Every earlier position j is put in one category relative to i:

  self        j == i
  own #       the '#' that opened i's bracket
  own toks    ordinary tokens of i's bracket                      <- what the filter needs
  own closes  close tokens of i's already-closed children
  children    anything inside those closed children
  ancestors   tokens (and '#') of the brackets enclosing i's bracket
  others      brackets that closed before i's bracket opened
  BOS

Table 1 averages over all positions at depth >= 1. Table 2 takes only positions that have just read a
close token and whose bracket had at least 3 ordinary tokens before that child opened: there the
needed tokens sit behind the child, as in (AA(BBB(C)  ->  BBB.
"uniform" is the share each category would get if attention were spread evenly over earlier positions.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from process import BracketProcess, OPEN, CLOSE0
from train import GPT, BOS

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
proc = BracketProcess(p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
N, T = 384, args["T"]
toks, _ = proc.sample(N, T, np.random.default_rng(99))
x = torch.cat([torch.full((N, 1), BOS), torch.from_numpy(toks)[:, :-1]], 1)
xin = x.numpy()
CATS = ["self", "own #", "own toks", "own closes", "children", "ancestors", "others", "BOS"]

# --- structure, per model position ---------------------------------------------------------------
cat = np.zeros((N, T, T), dtype=np.int8)
depth = np.zeros((N, T), dtype=np.int64)
own_open = np.full((N, T), -1)
n_own = np.zeros((N, T), dtype=np.int64)        # ordinary tokens of the own bracket read so far
after_close = np.zeros((N, T), dtype=bool)
hard = np.zeros((N, T), dtype=bool)             # just read a close, >= 3 own tokens before that child
for b in range(N):
    stack, nf, frame, opener = [0], 0, np.zeros(T, dtype=np.int64), [-1]
    open_at = np.zeros((T, T), dtype=bool)      # open_at[i, f]: bracket f is open at position i
    for i in range(T):
        tok = xin[b, i]
        if tok == OPEN:
            nf += 1
            stack.append(nf)
            opener.append(i)
        elif CLOSE0 <= tok < BOS:
            stack.pop()
        frame[i] = stack[-1]
        open_at[i, stack] = True
        depth[b, i] = len(stack) - 1
    opener = np.array(opener)
    o = opener[frame]
    own_open[b] = o
    ordinary = xin[b] < 3
    is_close = (xin[b] >= CLOSE0) & (xin[b] < BOS)
    I, J = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    same = frame[I] == frame[J]
    c = np.full((T, T), 6, dtype=np.int8)                               # others
    c[open_at[I, frame[J]] & ~same] = 5                                 # ancestors
    c[(J > o[I]) & ~same] = 4                                           # inside own closed children
    c[same & is_close[J]] = 3
    c[same & ordinary[J]] = 2
    c[J == o[I]] = 1
    c[J == 0] = 7
    c[I == J] = 0
    c[J > I] = -1
    cat[b] = c
    n_own[b] = np.array([(same[i, : i + 1] & ordinary[: i + 1]).sum() for i in range(T)])
    after_close[b] = is_close
    for i in np.flatnonzero(is_close):
        child_open = max(j for j in range(i) if xin[b, j] == OPEN and depth[b, j] == depth[b, i] + 1)
        hard[b, i] = (same[i, :child_open] & ordinary[:child_open]).sum() >= 3

# --- attention weights and residuals, recomputed by hand -----------------------------------------
model = GPT(T, args["d"], args["layers"], args["heads"])
model.load_state_dict(torch.load(run + ".pt", map_location="cpu"))
model.eval()
atts, resid = [], []
with torch.no_grad():
    h = model.tok(x) + model.pos(torch.arange(T))
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for blk in model.blocks:
        B, _, D = h.shape
        q, k, v = blk.qkv(blk.ln1(h)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, blk.heads, D // blk.heads).transpose(1, 2)
        a = (sh(q) @ sh(k).transpose(-1, -2) / (D // blk.heads) ** 0.5).masked_fill(mask, -1e9).softmax(-1)
        atts.append(a.numpy())
        h = h + blk.o((a @ sh(v)).transpose(1, 2).reshape(B, T, D))
        h = h + blk.mlp(blk.ln2(h))
        resid.append(h.numpy().astype(np.float64))
    err = (model.unembed(model.lnf(h)) - model(x)).abs().max().item()
print(f"{run}: hand-computed forward pass differs from the model by {err:.1e}")


def table(title, sel):
    print(f"\n{title}  ({sel.sum()} positions)")
    print(f"{'':10s}" + "".join(f"{c:>11s}" for c in CATS))
    uni = np.stack([((cat == ci) / np.maximum((cat >= 0).sum(-1, keepdims=True), 1))[sel].sum(-1).mean()
                    for ci in range(len(CATS))])
    print(f"{'uniform':10s}" + "".join(f"{u:11.2f}" for u in uni))
    for l, a in enumerate(atts):
        for hd in range(a.shape[1]):
            m = [(a[:, hd] * (cat == ci))[sel].sum(-1).mean() for ci in range(len(CATS))]
            print(f"L{l + 1} head {hd} " + "".join(f"{v:11.2f}" for v in m))


table("Table 1: all positions at depth >= 1", depth >= 1)
table("Table 2: just read a close; own tokens are behind the child", hard & (depth >= 1))


# --- address tags in the residual -----------------------------------------------------------------
def r2(X, y, sel):
    fit = sel.copy(); fit[int(0.75 * N):] = False
    test = sel.copy(); test[: int(0.75 * N)] = False
    Xf, yf, Xt, yt = X[fit], y[fit].astype(float), X[test], y[test].astype(float)
    mx, my = Xf.mean(0), yf.mean()
    w = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (yf - my))
    return 1 - ((yt - my - (Xt - mx) @ w) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


pos = np.tile(np.arange(T), (N, 1))
tags = [("depth", depth, depth >= 0),
        ("position of own '#' (depth >= 1)", own_open, depth >= 1),
        ("distance back to own '#'", pos - own_open, depth >= 1),
        ("own tokens read so far", n_own, depth >= 0)]
torch.manual_seed(123)
rand = GPT(T, args["d"], args["layers"], args["heads"]).eval()
with torch.no_grad():
    rres = [h.numpy().astype(np.float64) for h in rand(x, return_hidden=True)[1][1:]]
print("\nlinear readout of address tags, R^2 on held-out sequences, after block 1..L  (untrained net in brackets)")
for name, y, sel in tags:
    print(f"{name:34s}" + "".join(f"  {r2(X, y, sel):5.2f} ({r2(R, y, sel):5.2f})" for X, R in zip(resid, rres)))
