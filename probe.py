"""Linear probes on a trained run: where does each frame's belief live?

  python probe.py runs/pilot_d128_L4_T128_s0

Targets, per position t (the residual at t is the one that predicts token t):
  active   belief over (type, state) of the frame that emits token t                [9 numbers]
  parent   belief the enclosing frame had when it made the call (stale, suspended)  [9 numbers]
Ridge regression from the residual stream after each block; fit on 75% of the sequences, R^2 on the
rest. A randomly initialised copy of the network is the control for what a probe finds for free.
"""
import json, sys
import numpy as np
import torch
from process import BracketProcess, OPEN, CLOSE0
from train import GPT, BOS

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
proc = BracketProcess(p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
N, T = 1536, args["T"]
toks, deps = proc.sample(N, T, np.random.default_rng(777))
_, bel, resumed = proc.filter(toks, "bayes")

# position of the '#' that opened the frame active at t (-1 for the root)
opened = np.full((N, T), -1)
for b in range(N):
    stack = []
    for t in range(T):
        opened[b, t] = stack[-1] if stack else -1
        if toks[b, t] == OPEN:
            stack.append(t)
        elif toks[b, t] >= CLOSE0:
            stack.pop()
parent = bel[np.arange(N)[:, None], np.maximum(opened, 0)]
first = np.zeros((N, T), dtype=bool)
first[:, 1:] = toks[:, :-1] >= CLOSE0

x = torch.cat([torch.full((N, 1), BOS), torch.from_numpy(toks)[:, :-1]], 1)


def hidden(trained):
    torch.manual_seed(123)
    m = GPT(T, args["d"], args["layers"], args["heads"])
    if trained:
        m.load_state_dict(torch.load(run + ".pt", map_location="cpu"))
    m.eval()
    with torch.no_grad():
        hs = [torch.cat([m(x[i:i + 256], return_hidden=True)[1][l] for i in range(0, N, 256)]).double().numpy()
              for l in range(args["layers"] + 1)]
    return hs


def r2(X, Y, fit, test, recenter=False, lam=1e-4):
    """Fit on rows `fit`, report R^2 on rows `test` (boolean masks over [N, T]).
    recenter=True re-estimates only the intercept on the test rows, so it scores the linear map alone."""
    Xf, Yf, Xt, Yt = X[fit], Y[fit], X[test], Y[test]
    mx, my = Xf.mean(0), Yf.mean(0)
    W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + lam * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
    if recenter:
        mx, my = Xt.mean(0), Yt.mean(0)
    return 1 - ((Yt - my - (Xt - mx) @ W) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()


tr = np.zeros((N, T), dtype=bool)
tr[: int(0.75 * N)] = True
te = ~tr
child = deps >= 1
after_open = np.zeros((N, T), dtype=bool)          # first position of a new frame: its input token is '#'
after_open[:, 1:] = toks[:, :-1] == OPEN
shallow, deep = deps <= 1, deps >= 3
rows = [
    ("active belief: one probe, all positions", bel, tr, te, False),
    ("  same probe scored at depth 0", bel, tr, te & (deps == 0), False),
    ("  same probe scored at depth 1-2", bel, tr, te & (deps >= 1) & (deps <= 2), False),
    ("  same probe scored at depth >= 3", bel, tr, te & deep, False),
    ("  same probe, first token after return", bel, tr, te & first, False),
    ("probe fit at depth >= 3 only", bel, tr & deep, te & deep, False),
    ("fit depth<=1 -> depth>=3, as is", bel, tr & shallow, te & deep, False),
    ("fit depth<=1 -> depth>=3, intercept refit", bel, tr & shallow, te & deep, True),
    ("fit depth>=3 -> depth<=1, intercept refit", bel, tr & deep, te & shallow, True),
    ("PARENT belief, all child positions", parent, tr & child, te & child, False),
    ("  PARENT, first position of a frame", parent, tr & child, te & child & after_open, False),
    ("  PARENT, later positions of a frame", parent, tr & child, te & child & ~after_open, False),
    ("  PARENT, fit on later positions only", parent, tr & child & ~after_open, te & child & ~after_open, False),
]
print(f"{run}: R^2 of linear probes, residual after block 0..{args['layers']} (0 = embedding)")
print(f"share of child positions that are the first of their frame: {after_open[child].mean():.2f}")
for label, hs in (("trained", hidden(True)), ("random init", hidden(False))):
    print(f"\n{label:44s}" + "".join(f"  blk{l}" for l in range(len(hs))))
    for name, Y, fit, test, rc in rows:
        print(f"{name:44s}" + "".join(f" {r2(X, Y, fit, test, rc):5.2f}" for X in hs))


# --- Is it the same subspace, with the same coordinates, at every depth? -------------------------
# Encoding maps: least-squares map belief -> residual, fit separately in each depth band, after
# regressing the input token out of both (the last token alone predicts part of the belief, and its
# embedding is trivially shared across depths). "plane" = mean squared cosine of the principal angles
# between the two 8-dim images (1 = same subspace; random 8-dim subspaces in d dims give 8/d).
# "map" = cosine between the two encoding matrices (1 = same coordinates as well).
def encoding(X, mask):
    tokin = np.concatenate([np.full((N, 1), BOS), toks[:, :-1]], 1)[mask]
    onehot = np.eye(BOS + 1)[tokin]
    Xm, Bm = X[mask], bel[mask]
    Xr = Xm - onehot @ np.linalg.lstsq(onehot, Xm, rcond=None)[0]
    Br = Bm - onehot @ np.linalg.lstsq(onehot, Bm, rcond=None)[0]
    A = np.linalg.lstsq(Br, Xr, rcond=None)[0]                  # [9, d], rank 8
    basis = np.linalg.svd(A, full_matrices=False)[2][:8]
    return A, basis


bands = [("depth 0", deps == 0), ("depth 1-2", (deps >= 1) & (deps <= 2)), ("depth >= 3", deps >= 3)]
print(f"\nencoding maps across depth bands (chance plane overlap {8 / args['d']:.2f})")
for label, hs in (("trained", hidden(True)), ("random init", hidden(False))):
    for l in (len(hs) - 2, len(hs) - 1):
        enc = [encoding(hs[l], m) for _, m in bands]
        cells = []
        for i, j in ((0, 1), (1, 2), (0, 2)):
            plane = (np.linalg.norm(enc[i][1] @ enc[j][1].T) ** 2) / 8
            cos = (enc[i][0] * enc[j][0]).sum() / np.linalg.norm(enc[i][0]) / np.linalg.norm(enc[j][0])
            cells.append(f"{bands[i][0]} vs {bands[j][0]}: plane {plane:.2f} map {cos:.2f}")
        print(f"{label:12s} blk{l}  " + " | ".join(cells))
