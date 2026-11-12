"""How does the network write depth: along one line, or as a separate direction per depth?

Finding a position's own '#' means testing whether two depths are equal. Attention can do that on a
scalar only with extra features, and does it trivially on a one-hot-like code, which has no entry for a
depth never seen. This takes the mean residual at each trained depth (positions 64-127 only, so that
depth is not confounded with being late in the sequence) and asks how many dimensions those means span.

  python depth_code.py runs/NAME

participation ratio = (sum s^2)^2 / sum s^4 over the singular values of the centred depth means:
1 for a line, CAP for one direction per depth. It then prints the means in their two leading directions,
with the unseen depths projected onto the same two.
"""
import json, sys
import numpy as np
import torch
from process import make
from model import trace, load
import algorithm

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
name, CAP, T, L = args.get("process", "mess3"), args["max_depth"], args["T"], args["layers"]
deep = make(name, max_depth=24, p_open=args["p_open"], p_close=args["p_close"])
model = load(run, args, deep)
toks = deep.sample(40000, T, np.random.default_rng(34))[0]
peak = np.cumsum(deep.is_open(toks).astype(int) - deep.is_close(toks).astype(int), 1).max(1)
toks = np.concatenate([toks[peak <= CAP][:1500], toks[peak > CAP][:2500]])
ref = algorithm.run(deep, toks)
res = [[] for _ in range(L + 1)]
for a in range(0, len(toks), 128):
    rs = trace(model, torch.from_numpy(ref["read"][a:a + 128]), want_attn=False)[1]
    for l in range(L + 1):
        res[l].append(rs[l].numpy())
res = [np.concatenate(v) for v in res]
late = np.broadcast_to(np.arange(T) >= 64, ref["depth"].shape)
depth = ref["depth"]
print(f"{run}: {name}, {L} layers; mean residual at each depth, positions 64-127")
print(f"{'':10s}{'participation':>14s}{'share of variance in':>26s}")
print(f"{'':10s}{'ratio':>14s}{'1st, 2nd, 3rd direction':>26s}")
keep = {}
for l in range(L + 1):
    X = res[l].astype(float)
    mu = np.stack([X[late & (depth == d)].mean(0) for d in range(CAP + 5)])
    centre = mu[:CAP + 1].mean(0)
    u, sv, vt = np.linalg.svd(mu[:CAP + 1] - centre, full_matrices=False)
    s = sv ** 2
    print(f"{('embedding' if l == 0 else f'block {l}'):10s}{s.sum() ** 2 / (s ** 2).sum():14.2f}{'':8s}" + ", ".join(f"{v:.2f}" for v in s[:3] / s.sum()))
    keep[l] = (mu - centre) @ vt[:2].T
for l in (2, 3):
    xy = keep[l]
    a, b = xy[:CAP + 1, 0], xy[:CAP + 1, 1]
    quad = np.polyfit(a, b, 2)                             # second coordinate as a parabola in the first, fitted on trained depths
    fit = 1 - ((b - np.polyval(quad, a)) ** 2).sum() / ((b - b.mean()) ** 2).sum()
    print(f"\nblock {l}: depth means in their two leading directions (fitted on depths 0-{CAP}; depths above {CAP} projected onto them)")
    print(f"{'depth':28s}" + "".join(f"{d:>7d}" for d in range(CAP + 5)))
    print(f"{'first direction':28s}" + "".join(f"{v:7.1f}" for v in xy[:, 0]))
    print(f"{'second direction':28s}" + "".join(f"{v:7.1f}" for v in xy[:, 1]))
    print(f"{'  parabola through 0-' + str(CAP):28s}" + "".join(f"{v:7.1f}" for v in np.polyval(quad, xy[:, 0])) + f"   (R^2 {fit:.3f} on the trained depths)")
