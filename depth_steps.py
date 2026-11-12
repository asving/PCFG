"""Which step of the algorithm breaks at depths never shown in training?

depth_ood.py shows the loss rising past the training depth cap. This fits each probe of validate.py on
sequences that stay within the cap, then applies the same probe, unchanged, to sequences that go deeper,
and reports how well it reads each quantity at each depth.

  python depth_steps.py runs/NAME

A quantity that is still read well beyond the cap is still being computed there, in the same place.
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
kw = dict(p_open=args["p_open"], p_close=args["p_close"])
deep = make(name, max_depth=24, **kw)
model = load(run, args, deep)
toks = deep.sample(40000, T, np.random.default_rng(32))[0]
peak = np.cumsum(deep.is_open(toks).astype(int) - deep.is_close(toks).astype(int), 1).max(1)
sets = {"fit": toks[peak <= CAP][:1500], "deep": toks[peak > CAP][:2000]}
ref = {k: algorithm.run(deep, v) for k, v in sets.items()}


def residuals(xin):
    out = [[] for _ in range(L + 1)]
    for a in range(0, len(xin), 128):
        rs = trace(model, torch.from_numpy(xin[a:a + 128]), want_attn=False)[1]
        for l in range(L + 1):
            out[l].append(rs[l].numpy())
    return [np.concatenate(v) for v in out]


resid = {k: residuals(r["read"]) for k, r in ref.items()}
J = np.arange(T)
lw = lambda w: np.clip(np.log(w + 1e-12) - np.log(w + 1e-12).mean(-1, keepdims=True), -6, 6)
targets = [("1 depth", lambda r: r["depth"], lambda r: np.ones_like(r["depth"], dtype=bool)),
           ("2 tag: distance back to own '#'", lambda r: J[None] - r["own"], lambda r: r["depth"] >= 1),
           ("3 index: own tokens read", lambda r: r["index"], lambda r: np.ones_like(r["depth"], dtype=bool)),
           ("4 state belief per type", lambda r: r["state"], lambda r: r["index"] >= 1),
           ("6 type log-posterior", lambda r: lw(r["w"]), lambda r: r["index"] >= 1)]
depths = list(range(0, CAP + 6))
nfit = int(0.75 * len(sets["fit"]))
print(f"{run}: {name}, {L} layers; probes fitted on {nfit} sequences that stay within depth {CAP}, applied unchanged to {len(sets['deep'])} that go deeper")
print(f"R^2 by depth of the position (around the mean of the tested positions at that depth); depths above {CAP} were never shown\n")
print(f"{'':34s}{'block':>6s}{'in range':>10s}" + "".join(f"{d:>7d}" for d in depths))
pred_depth = None
for label, tgt, sel in targets:
    Yf, Yd = [tgt(ref[k]).reshape(len(sets[k]), T, -1).astype(float) for k in ("fit", "deep")]
    sf, sd = sel(ref["fit"]), sel(ref["deep"])
    tr, ho = sf.copy(), sf.copy()
    tr[nfit:], ho[:nfit] = False, False
    best = None
    for l in range(L + 1):
        X = resid["fit"][l].astype(float)
        mx, my = X[tr].mean(0), Yf[tr].mean(0)
        W = np.linalg.solve((X[tr] - mx).T @ (X[tr] - mx) + 1e-4 * tr.sum() * np.eye(X.shape[-1]), (X[tr] - mx).T @ (Yf[tr] - my))
        r2 = 1 - ((Yf[ho] - my - (X[ho] - mx) @ W) ** 2).sum() / ((Yf[ho] - Yf[ho].mean(0)) ** 2).sum()
        if best is None or r2 > best[0]:
            best = (r2, l, mx, my, W)
    r2, l, mx, my, W = best
    P = (resid["deep"][l].astype(float) - mx) @ W + my
    dd = ref["deep"]["depth"]
    if label.startswith("1"):
        pred_depth = (l, [P[dd == d].mean() for d in depths], [P[dd == d].std() for d in depths])
        print(f"{label:34s}{l:>6d}{r2:10.3f}" + "".join(f"{'':>7s}" for _ in depths))
        continue
    row, row_c = [], []
    for d in depths:
        m = sd & (dd == d)
        if m.sum() <= 200:
            row.append(float("nan")), row_c.append(float("nan"))
            continue
        err, tot = Yd[m] - P[m], ((Yd[m] - Yd[m].mean(0)) ** 2).sum()
        row.append(1 - (err ** 2).sum() / tot)
        row_c.append(1 - ((err - err.mean(0)) ** 2).sum() / tot)       # the probe's constant re-estimated at this depth
    print(f"{label:34s}{l:>6d}{r2:10.3f}" + "".join(f"{v:7.2f}" for v in row))
    print(f"{'   constant re-estimated per depth':34s}{'':>6s}{'':>10s}" + "".join(f"{v:7.2f}" for v in row_c))
l, mean, sd_ = pred_depth
print(f"\ndepth as read by the probe (block {l}), by true depth:")
print(f"{'true depth':34s}{'':16s}" + "".join(f"{d:>7d}" for d in depths))
print(f"{'mean readout':34s}{'':16s}" + "".join(f"{v:7.2f}" for v in mean))
print(f"{'spread (sd)':34s}{'':16s}" + "".join(f"{v:7.2f}" for v in sd_))
