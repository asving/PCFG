"""Depth as an angle, and what the reader heads attend to at depths never shown.

depth_code.py shows the depth means lying in two dimensions. This prints the angle of each depth's mean in
that plane, then follows the heads that read a position's own bracket (found within the trained depths)
out to unseen depths: how much of their attention stays on the own bracket, and how much goes to positions
at depth 0 (the root), at the deepest trained depth, or in other brackets at unseen depths.

  python depth_alias.py runs/NAME
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
toks = deep.sample(40000, T, np.random.default_rng(35))[0]
peak = np.cumsum(deep.is_open(toks).astype(int) - deep.is_close(toks).astype(int), 1).max(1)
toks = toks[peak > CAP][:1500]
ref = algorithm.run(deep, toks)
depth, own = ref["depth"], ref["own"]
N, H = len(toks), args["heads"]
Q = list(range(4, CAP + 5))
cats = ["own bracket", "depth 0 (root)", f"depth {CAP}", f"other, depth > {CAP}", f"other, depth 1-{CAP - 1}"]
mass = np.zeros((L, H, len(Q), len(cats)))
count = np.zeros(len(Q))
sums = [np.zeros((CAP + 5, args["d"])) for _ in range(L + 1)]
cnt = np.zeros(CAP + 5)
late = np.arange(T) >= 64
for a in range(0, N, 100):
    sl = slice(a, a + 100)
    _, rs, attn = trace(model, torch.from_numpy(ref["read"][sl]))
    d, o = depth[sl], own[sl]
    same = o[:, :, None] == o[:, None, :]
    key = [same, ~same & (d[:, None, :] == 0), ~same & (d[:, None, :] == CAP), ~same & (d[:, None, :] > CAP)]
    key.append(~(key[0] | key[1] | key[2] | key[3]))
    for qi, q in enumerate(Q):
        m = (d == q) & late[None]
        count[qi] += m.sum()
        for l in range(L):
            A = attn[l].numpy()                              # [B, H, T, T]
            for ci, k in enumerate(key):
                mass[l, :, qi, ci] += np.einsum("bhij,bij->h", A, (k & m[:, :, None]).astype(np.float32))
    for dd in range(CAP + 5):
        m = (d == dd) & late[None]
        cnt[dd] += m.sum()
        for l in range(L + 1):
            sums[l][dd] += rs[l].numpy()[m].sum(0)
mass /= np.maximum(count, 1)[None, None, :, None]

print(f"{run}: {name}, {L} layers; {N} sequences that go deeper than {CAP}; positions 64-127")
print("\nANGLE of each depth's mean residual in the plane of the two leading directions of the trained depths' means (degrees)")
print(f"{'':10s}" + "".join(f"{d:>7d}" for d in range(CAP + 5)) + "    step per level, trained depths")
for l in range(1, L + 1):
    mu = sums[l] / np.maximum(cnt, 1)[:, None]
    c = mu[:CAP + 1].mean(0)
    vt = np.linalg.svd(mu[:CAP + 1] - c, full_matrices=False)[2][:2]
    xy = (mu - c) @ vt.T
    ang = np.degrees(np.unwrap(np.arctan2(xy[:, 1], xy[:, 0])))
    ang = (ang - ang[0]) * np.sign(ang[CAP] - ang[0])
    step = np.diff(ang[:CAP + 1])
    print(f"{f'block {l}':10s}" + "".join(f"{v:7.0f}" for v in ang) + f"    {step.mean():.0f} (from {step.min():.0f} to {step.max():.0f})")

inr = [Q.index(q) for q in (5, 6, 7)]
score = mass[:, :, inr, 0].mean(-1)
readers = [(l, h) for l in range(L) for h in range(H) if score[l, h] > 0.5]
print(f"\nREADER HEADS (more than half of their attention on the own bracket at depths 5-7): "
      + ", ".join(f"block {l + 1} head {h}" for l, h in readers))
print(f"share of attention, mean over those heads, by depth of the reading position")
print(f"{'':22s}" + "".join(f"{q:>7d}" for q in Q))
for ci, cname in enumerate(cats):
    print(f"{cname:22s}" + "".join(f"{np.mean([mass[l, h, qi, ci] for l, h in readers]):7.2f}" for qi in range(len(Q))))
