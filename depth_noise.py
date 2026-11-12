"""Within the trained depths, where is the depth count least accurate?

Depth is a count (opens minus closes). A transformer gets it from an average over the positions so far,
in which one level is a difference of 1/t at position t, so the count could get noisier late in the
sequence. This reads depth with a linear probe after block 2 and prints the probe's RMS error by true
depth and by position, beside the network's excess loss in the same cells.

  python depth_noise.py runs/NAME
"""
import json, sys
import numpy as np
import torch
from process import make
from model import trace, load
import algorithm

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
proc = make(args.get("process", "mess3"), p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
N, T = 3072, args["T"]
toks, _ = proc.sample(N, T, np.random.default_rng(2029))
ref = algorithm.run(proc, toks)
model = load(run, args, proc)
X, lp = [], []
for a in range(0, N, 128):
    lg, rs, _ = trace(model, torch.from_numpy(ref["read"][a:a + 128]), want_attn=False)
    X.append(rs[2].double().numpy())
    lp.append(torch.log_softmax(lg, -1).double().numpy())
X, lp = np.concatenate(X), np.concatenate(lp)
kl, depth, nf = algorithm.excess(ref["pred"], lp), ref["depth"].astype(float), int(0.75 * N)
Xf, Yf = X[:nf].reshape(-1, X.shape[-1]), depth[:nf].reshape(-1, 1)
mx, my = Xf.mean(0), Yf.mean(0)
W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
err = ((X[nf:] - mx) @ W + my)[..., 0] - depth[nf:]
d, k = depth[nf:], kl[nf:]
pos = np.broadcast_to(np.arange(T), d.shape)
bins = [(1, 16), (16, 32), (32, 64), (64, 96), (96, 128)]
print(f"{run}: depth read after block 2, {N - nf} held-out sequences")
print(f"{'':8s}RMS error of the depth readout, by position" + " " * 12 + "| network excess loss, mnat/token, same cells")
print(f"{'depth':>8s}" + "".join(f"{f'{a}-{b - 1}':>10s}" for a, b in bins) + "     |" + "".join(f"{f'{a}-{b - 1}':>8s}" for a, b in bins))
for dd in range(args["max_depth"] + 1):
    cells = [(d == dd) & (pos >= a) & (pos < b) for a, b in bins]
    print(f"{dd:>8d}" + "".join(f"{np.sqrt((err[m] ** 2).mean()) if m.sum() > 150 else np.nan:10.2f}" for m in cells)
          + "     |" + "".join(f"{1e3 * k[m].mean() if m.sum() > 150 else np.nan:8.1f}" for m in cells))
