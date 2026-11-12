"""Lengths beyond training: does a network trained on 128 tokens still predict well further into a sequence?

Only for networks without a learned position table (train.py --pos_rotary). The process is the one the
network was trained on, depth cap included; only the sequences are longer. Excess loss over exact Bayes
(the stack filter), mnat/token, by position band and by the depth of the predicting position.

  python length_ood.py runs/NAME [length]
"""
import json, sys
import numpy as np
import torch
from process import make
from model import load
import algorithm

run = sys.argv[1]
LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 512
args = json.load(open(run + ".json"))["args"]
name, T = args.get("process", "mess3"), args["T"]
proc = make(name, p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
model = load(run, args, proc)
N = 1500
toks = proc.sample(N, LEN, np.random.default_rng(41))[0]
exact = proc.filter(toks, "bayes")[0]
small = algorithm.run(proc, toks[:24, :256], chunk=8)
print(f"{run}: {name}, trained on {T} tokens; tested on {N} sequences of {LEN}")
print(f"check: stack filter vs algorithm.py on 24 sequences of 256: max |difference| {np.abs(small['pred'] - exact[:24, :256]).max():.1e}")
xin = algorithm.read_tokens(proc, toks)
with torch.no_grad():
    lp = torch.cat([torch.log_softmax(model(torch.from_numpy(xin[a:a + 64])), -1) for a in range(0, N, 64)]).double().numpy()
depth = np.cumsum(proc.is_open(xin).astype(int) - proc.is_close(xin).astype(int), 1)
closed = proc.is_close(xin)
J = np.arange(LEN)
since_open = np.zeros((N, LEN), dtype=np.int64)              # positions since the own bracket opened (BOS for the root)
for a in range(0, N, 50):
    d = depth[a:a + 50]
    cand = (d[:, :, None] == d[:, None, :]) & (proc.is_open(xin[a:a + 50]) | (J == 0))[:, None, :] & (J[None, :] <= J[:, None])
    since_open[a:a + 50] = J[None] - np.where(cand, J, -1).max(-1)
kl = {"THE NETWORK": algorithm.excess(exact, lp),
      "forgets the parent at a close": algorithm.excess(exact, proc.filter(toks, "amnesic")[0]),
      "no inference at all": algorithm.excess(exact, proc.filter(toks, "prior")[0])}
bands = [(a, min(a + T, LEN)) for a in range(0, LEN, T)]
band = lambda a, b: np.broadcast_to((J >= a) & (J < b), depth.shape)
cols = [("all", np.ones_like(depth, dtype=bool)), ("root", depth == 0), ("depth 1-3", (depth >= 1) & (depth <= 3)),
        ("depth 4+", depth >= 4), ("just read a close", closed),
        (f"own open <= {T} back", (depth >= 1) & (since_open <= T)), (f"own open > {T} back", (depth >= 1) & (since_open > T))]
m = lambda v, sel: f"{1e3 * v[sel].mean():22.2f}" if sel.sum() >= 200 else f"{'-':>22s}"
for label, v in kl.items():
    print(f"\n{label}: excess loss, mnat/token" + ("" if label == "THE NETWORK" else "   (for scale)"))
    print(f"{'positions':12s}" + "".join(f"{c:>22s}" for c, _ in cols))
    for a, b in bands:
        print(f"{f'{a}-{b - 1}':12s}" + "".join(m(v, sel & band(a, b)) for _, sel in cols))
print(f"\nshare of positions by depth group, per band (root, 1-3, 4+): "
      + "; ".join(f"{a}-{b - 1}: " + ", ".join(f"{(sel & band(a, b)).sum() / band(a, b).sum():.2f}" for _, sel in cols[1:4]) for a, b in bands))
first = 1e3 * kl["THE NETWORK"][band(*bands[0])].mean()
print("summary: the network by band, relative to the trained band: "
      + ", ".join(f"{a}-{b - 1}: {1e3 * kl['THE NETWORK'][band(a, b)].mean() / first:.1f}x" for a, b in bands))
