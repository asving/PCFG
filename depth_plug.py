"""Does the network fail at unseen depths the way the algorithm would, if the algorithm had the network's depth?

depth_steps.py shows the network's depth count losing resolution past the training cap. This reads the
depth off the network with a linear probe (fitted on sequences within the cap), hands that reading to
algorithm.py in place of the exact count, and compares the algorithm's predictions with the network's.

  python depth_plug.py runs/NAME

In step 2 the algorithm looks for the most recent open at its own depth. With a read depth, two positions
count as the same depth when their readings differ by less than a tolerance (0.5 unless stated).
All losses are within token class (see depth_ood.py), in mnat/token.
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
C, O, Z = deep.C, deep.OPEN0, deep.CLOSE0
model = load(run, args, deep)
toks = deep.sample(40000, T, np.random.default_rng(33))[0]
peak = np.cumsum(deep.is_open(toks).astype(int) - deep.is_close(toks).astype(int), 1).max(1)
sets = {"fit": toks[peak <= CAP][:1500], "within": toks[peak <= CAP][1500:3000], "deep": toks[peak > CAP][:2500]}
ref = {k: algorithm.run(deep, v) for k, v in sets.items()}


def forward(xin):
    res, lps = [[] for _ in range(L + 1)], []
    for a in range(0, len(xin), 128):
        lg, rs, _ = trace(model, torch.from_numpy(xin[a:a + 128]), want_attn=False)
        lps.append(torch.log_softmax(lg, -1).double().numpy())
        for l in range(L + 1):
            res[l].append(rs[l].numpy())
    return [np.concatenate(v) for v in res], np.concatenate(lps)


out = {k: forward(r["read"]) for k, r in ref.items()}

# the depth probe: fitted within the cap, block chosen on held-out sequences within the cap
Y = ref["fit"]["depth"].astype(float)
nf = int(0.75 * len(Y))
best = None
for l in range(L + 1):
    X = out["fit"][0][l].astype(float)
    Xf, Yf = X[:nf].reshape(-1, X.shape[-1]), Y[:nf].reshape(-1)
    mx, my = Xf.mean(0), Yf.mean()
    W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
    r2 = 1 - ((Y[nf:] - my - (X[nf:] - mx) @ W) ** 2).sum() / ((Y[nf:] - Y[nf:].mean()) ** 2).sum()
    if best is None or r2 > best[0]:
        best = (r2, l, mx, my, W)
r2, blk, mx, my, W = best
read = {k: (out[k][0][blk].astype(float) - mx) @ W + my for k in ("within", "deep")}


def within(p, logq):
    """KL(p || q) within token class; p probabilities, logq log-probabilities."""
    tot = np.zeros(p.shape[:-1])
    for a, b in ((0, C), (O, Z), (Z, p.shape[-1])):
        q = p[..., a:b]
        mass = q.sum(-1)
        cond = q / np.where(mass > 0, mass, 1.0)[..., None]
        lq = logq[..., a:b] - np.logaddexp.reduce(logq[..., a:b], -1)[..., None]
        tot += mass * (cond * (np.log(np.where(cond > 0, cond, 1.0)) - lq)).sum(-1)
    return tot


safe = lambda p: np.log(np.maximum(p, 1e-300))
m = lambda v, sel: 1e3 * v[sel].mean() if sel.sum() > 100 else float("nan")
print(f"{run}: {name}, {L} layers.")
print(f"depth is read by a linear probe after block {blk} (R^2 {r2:.3f} within the cap); two positions are at the same depth when their readings are within 0.5")
check = algorithm.run(deep, sets["within"][:64], depth_read=ref["within"]["depth"][:64].astype(float))
print(f"check: the algorithm given the exact depth through the same route differs from the exact law by {np.abs(check['pred'] - ref['within']['pred'][:64]).max():.1e}")

for key, title in (("within", f"sequences that stay within depth {CAP}"), ("deep", f"sequences that go deeper than {CAP}")):
    r, lp = ref[key], out[key][1]
    exact, depth = r["pred"], r["depth"]
    depths = [d for d in range(0, 14) if (depth == d).sum() > 300]
    net_x = within(exact, lp)
    print(f"\n=== {title}  ({len(sets[key])} sequences; within-class, mnat/token)")
    print(f"{'depth of the position':46s}" + "".join(f"{d:>7d}" for d in depths))
    print(f"{'THE NETWORK, excess over exact':46s}" + "".join(f"{m(net_x, depth == d):7.1f}" for d in depths))
    dr = read[key]
    plug = algorithm.run(deep, sets[key], depth_read=dr)
    plug_x, gap, wrong = within(exact, safe(plug["pred"])), within(plug["pred"], lp), plug["own"] != r["own"]
    print(f"{'mean depth reading':46s}" + "".join(f"{dr[depth == d].mean():7.2f}" for d in depths))
    print(f"{'THE ALGORITHM GIVEN THAT READING, excess':46s}" + "".join(f"{m(plug_x, depth == d):7.1f}" for d in depths))
    print(f"{'share of positions where it picks a wrong #':46s}" + "".join(f"{(wrong & (depth == d)).sum() / max((depth == d).sum(), 1):7.2f}" for d in depths))
    print(f"{'network excess where it picks the right #':46s}" + "".join(f"{m(net_x, ~wrong & (depth == d)):7.1f}" for d in depths))
    print(f"{'network excess where it picks a wrong #':46s}" + "".join(f"{m(net_x, wrong & (depth == d)):7.1f}" for d in depths))
    cc = [np.corrcoef(np.log(net_x[depth == d] + 1e-6), np.log(plug_x[depth == d] + 1e-6))[0, 1] for d in depths]
    print(f"{'correlation of log excess, by position':46s}" + "".join(f"{v:7.2f}" for v in cc))
    print(f"{'distance from the network to that algorithm':46s}" + "".join(f"{m(gap, depth == d):7.1f}" for d in depths))
    print(f"{'distance from the network to the exact law':46s}" + "".join(f"{m(net_x, depth == d):7.1f}" for d in depths))
    if key == "deep":
        print(f"\nthe tolerance is a free choice; the algorithm's excess at depths 7 (shown), {CAP + 1}, {CAP + 2}, {CAP + 3}:")
        for tol in (0.3, 0.5, 0.75, 1.0):
            p = algorithm.run(deep, sets[key], depth_read=read[key], tol=tol)
            x = within(exact, safe(p["pred"]))
            print(f"  tolerance {tol:4.2f}: " + ", ".join(f"{m(x, depth == d):6.1f}" for d in (7, CAP + 1, CAP + 2, CAP + 3)))
