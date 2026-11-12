"""Depths beyond training: does a trained network still predict well deeper than it was ever shown?

Training capped the nesting depth (8 in every run so far). This replays the run's own training sampler to
count how often each depth was shown, then tests the network on the same process with the cap lifted, on
sequences that go deeper than the cap.

  python depth_ood.py runs/NAME [n_sampled]

The score is the excess loss WITHIN token class: KL of the content token given that the next token is
content, plus KL of the close type given that it is a close, each weighted by the exact probability of
that class. It leaves out the split between open, close and content, which training fixed per depth (at
the cap it taught that opens never happen) and never taught beyond it. The total is printed beside it.
"""
import json, sys
import numpy as np
import torch
from process import make
from model import load
import algorithm

run = sys.argv[1]
n_sampled = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
args = json.load(open(run + ".json"))["args"]
name, CAP, T = args.get("process", "mess3"), args["max_depth"], args["T"]
kw = dict(p_open=args["p_open"], p_close=args["p_close"])
proc, deep = make(name, max_depth=CAP, **kw), make(name, max_depth=24, **kw)
C, O, Z = proc.C, proc.OPEN0, proc.CLOSE0
model = load(run, args, proc)
depth_after = lambda p, toks: np.cumsum(p.is_open(toks).astype(int) - p.is_close(toks).astype(int), 1)

# --- what training showed -------------------------------------------------------------------------
rng = np.random.default_rng(args["seed"] + 1)                # train.py draws one batch per step from this generator
shown = np.zeros(32, dtype=np.int64)
for _ in range(args["steps"]):
    toks, _ = proc.sample(args["batch"], T, rng)
    shown += np.bincount(depth_after(proc, toks).ravel(), minlength=32)
print(f"{run}: {name}, {args['layers']} layers, d = {args['d']}, trained {args['steps']} steps with depth capped at {CAP}")
print(f"deepest depth in any training sequence: {np.flatnonzero(shown).max()}")
print("training positions by depth: " + ", ".join(f"{d}: {shown[d]:,}" for d in range(np.flatnonzero(shown).max() + 1)))


# --- scoring --------------------------------------------------------------------------------------
def within(bayes, logq):
    """Excess loss within token class, and in total. bayes: probabilities; logq: log-probabilities."""
    tot = algorithm.excess(bayes, logq)
    out = np.zeros(bayes.shape[:-1])
    for a, b in ((0, C), (O, Z), (Z, bayes.shape[-1])):
        p = bayes[..., a:b]
        mass = p.sum(-1)
        cond = p / np.where(mass > 0, mass, 1.0)[..., None]
        lq = logq[..., a:b] - np.logaddexp.reduce(logq[..., a:b], -1)[..., None]
        out += mass * (cond * (np.log(np.where(cond > 0, cond, 1.0)) - lq)).sum(-1)
    return out, tot


def network(xin):
    with torch.no_grad():
        return torch.cat([torch.log_softmax(model(torch.from_numpy(xin[a:a + 256])), -1) for a in range(0, len(xin), 256)]).double().numpy()


def score(toks):
    ref = algorithm.run(deep, toks)
    lp = network(ref["read"])
    out = {"network": within(ref["pred"], lp)}
    safe = lambda p: np.log(np.maximum(p, 1e-300))
    for label, pred in [("no bracket tag (depth only)", algorithm.run(deep, toks, tag="depth")["pred"]),
                        ("no reset at an open", algorithm.run(deep, toks, tag="open")["pred"]),
                        ("forgets the parent at a close", deep.filter(toks, "amnesic")[0]),
                        ("no inference at all", deep.filter(toks, "prior")[0])]:
        out[label] = within(ref["pred"], safe(pred))
    return ref, out


# --- test sequences -------------------------------------------------------------------------------
toks = deep.sample(n_sampled, T, np.random.default_rng(31))[0]
peak = depth_after(deep, toks).max(1)
over, inside = toks[peak > CAP][:6000], toks[peak <= CAP][:3000]
print(f"\ntest: {n_sampled} sequences from the same process with the cap lifted; {int((peak > CAP).sum())} go deeper than {CAP} "
      f"({100 * (peak > CAP).mean():.1f}%). Scored: {len(over)} that do, and {len(inside)} that do not, for comparison.")
ref, res = score(over)
ref_in, res_in = score(inside)
depth, index, just_closed = ref["depth"], ref["index"], deep.is_close(ref["read"])
depths = [d for d in range(0, 20) if (depth == d).sum() >= 400]
m = lambda v, sel: 1e3 * v[sel].mean() if sel.any() else float("nan")

print(f"\n=== A. EXCESS LOSS WITHIN TOKEN CLASS, by depth of the predicting position (mnat/token); depths above {CAP} were never shown")
print(f"{'':34s}" + "".join(f"{d:>8d}" for d in depths))
print(f"{'positions tested':34s}" + "".join(f"{int((depth == d).sum()):>8d}" for d in depths))
print(f"{'training positions, millions':34s}" + "".join(f"{shown[d] / 1e6:8.2f}" for d in depths))
print(f"{'THE NETWORK':34s}" + "".join(f"{m(res['network'][0], depth == d):8.1f}" for d in depths))
print(f"{'  sequences that stay within cap':34s}" + "".join(f"{m(res_in['network'][0], ref_in['depth'] == d):8.1f}" for d in depths))
for label in list(res)[1:]:
    print(f"{label:34s}" + "".join(f"{m(res[label][0], depth == d):8.1f}" for d in depths))
print(f"{'the network, total excess':34s}" + "".join(f"{m(res['network'][1], depth == d):8.1f}" for d in depths))

print(f"\n=== B. THE NETWORK BY WHAT THE POSITION HAS TO DO (within class, mnat/token)")
kinds = [("new bracket, <= 2 own tokens", (index <= 2) & ~just_closed),
         ("in a bracket, >= 3 own tokens", (index >= 3) & ~just_closed),
         ("just read a close (resume parent)", just_closed)]
print(f"{'':34s}" + "".join(f"{d:>8d}" for d in depths))
for label, sel in kinds:
    print(f"{label:34s}" + "".join(f"{m(res['network'][0], sel & (depth == d)):8.1f}" for d in depths))

print(f"\n=== C. DOES GOING DEEPER DAMAGE THE BRACKETS AROUND IT?  (the network, within class, at depths <= {CAP})")
went = np.maximum.accumulate(depth, 1) > CAP                 # this position comes after the sequence first exceeded the cap
shallow = [d for d in depths if d <= CAP]
print(f"{'':34s}" + "".join(f"{d:>8d}" for d in shallow))
for label, sel in [("before first going deeper", ~went), ("after having been deeper", went)]:
    print(f"{label:34s}" + "".join(f"{m(res['network'][0], sel & (depth == d)):8.1f}" for d in shallow))

lo = m(res["network"][0], (depth >= CAP - 1) & (depth <= CAP))
clean = m(res["network"][0], (depth >= CAP - 1) & (depth <= CAP) & ~went)
hi = [m(res["network"][0], depth == d) for d in range(CAP + 1, CAP + 4)]
print(f"\nsummary: depths {CAP + 1}, {CAP + 2}, {CAP + 3} (never shown) " + ", ".join(f"{h:.1f}" for h in hi) + " mnat")
print(f"  against depths {CAP - 1}-{CAP} on the same sequences, {lo:.1f}: " + ", ".join(f"{h / lo:.1f}x" for h in hi)
      + "   (the baseline fixed before the test; it includes positions that have come back from an unseen depth)")
print(f"  against depths {CAP - 1}-{CAP} before the sequence first goes deeper, {clean:.1f}: " + ", ".join(f"{h / clean:.0f}x" for h in hi))
