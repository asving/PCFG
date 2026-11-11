"""Check the address rules a parallel (attention-style) filter would use.

D[t] = depth after token t. The residual at t predicts token t+1, which is emitted by the frame
that is active after token t. Claims, for every position t:
  A  that frame's own '#' is the most recent position o <= t with token '#' and D[o] == D[t]
     (no such position when D[t] == 0: the root);
  B  the earlier positions whose active frame is the same one are exactly
     {s : o <= s <= t, D[s] == D[t]}  (for the root: all s <= t with D[s] == 0);
  C  the previous position in the same frame is the most recent s < t with D[s] == D[t];
     for a close token this is the position just before the matching '#'.
"""
import numpy as np
from process import BracketProcess, OPEN, CLOSE0

proc = BracketProcess(p_open=0.2, p_close=0.22, max_depth=8)
toks, _ = proc.sample(400, 160, np.random.default_rng(3))
bad = dict(A=0, B=0, C=0)
n = 0
for seq in toks:
    stack, fid, frame, opener, D = [0], 0, [], {0: -1}, []
    for t, x in enumerate(seq):                      # ground truth by explicit stack
        if x == OPEN:
            fid += 1
            stack.append(fid)
            opener[fid] = t
        elif x >= CLOSE0:
            stack.pop()
        frame.append(stack[-1])
        D.append(len(stack) - 1)
    frame, D = np.array(frame), np.array(D)
    for t in range(len(seq)):
        n += 1
        cand = [o for o in range(t + 1) if seq[o] == OPEN and D[o] == D[t]]
        o = cand[-1] if cand else -1
        bad["A"] += o != opener[frame[t]]
        same = {s for s in range(max(o, 0), t + 1) if D[s] == D[t]}
        bad["B"] += same != set(np.flatnonzero(frame[: t + 1] == frame[t]))
        prev = [s for s in range(t) if D[s] == D[t]]
        truth = [s for s in range(t) if frame[s] == frame[t]]
        if truth:
            bad["C"] += (not prev) or prev[-1] != truth[-1]
        if seq[t] >= CLOSE0:                         # close: predecessor is just before the matching '#'
            child_open = max(s for s in range(t) if seq[s] == OPEN and D[s] == D[t] + 1)
            bad["C"] += (child_open - 1 >= 0) and prev[-1] != child_open - 1
print(f"{n} positions checked: violations {bad}")
