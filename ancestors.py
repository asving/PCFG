"""Inside a nested bracket, does the network still hold the beliefs of the brackets around it?

Two pictures agree on the loss. In one, a position inside a child carries only the child's belief, and the
parent's is rebuilt from the parent's own tokens when the child closes (the algorithm in algorithm.py). In
the other, every position carries the running beliefs of all enclosing brackets and uses one of them. This
probes for the second: from positions inside a child, read the parent's belief as it stood when the child
opened, which is exactly what will be needed at the close.

  python ancestors.py runs/NAME

R^2 of a ridge probe on held-out sequences, after the embedding and after each block.
"""
import json, sys
import numpy as np
import torch
from process import make
from model import GPT, trace, load
import algorithm

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
proc = make(args.get("process", "mess3"), p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
N, T, L = 1536, args["T"], args["layers"]
toks, _ = proc.sample(N, T, np.random.default_rng(2028))
ref = algorithm.run(proc, toks)
depth, own, index = ref["depth"], ref["own"], ref["index"]
model = load(run, args, proc)
torch.manual_seed(123)
blank = GPT(proc.VOCAB, T, args["d"], L, args["heads"]).eval()


def residuals(net):
    out = [[] for _ in range(L + 1)]
    for a in range(0, N, 128):
        rs = trace(net, torch.from_numpy(ref["read"][a:a + 128]), want_attn=False)[1]
        for l in range(L + 1):
            out[l].append(rs[l].double().numpy())
    return [np.concatenate(v) for v in out]


resid, blank_resid = residuals(model), residuals(blank)
test = np.zeros((N, T), dtype=bool)
test[int(0.75 * N):] = True


def r2(X, Y, sel):
    Y = Y.reshape(N, T, -1).astype(float)
    Xf, Yf, Xt, Yt = X[sel & ~test], Y[sel & ~test], X[sel & test], Y[sel & test]
    mx, my = Xf.mean(0), Yf.mean(0)
    W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
    return 1 - ((Yt - my - (Xt - mx) @ W) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()


at = lambda Y, pos: np.take_along_axis(Y.reshape(N, T, -1), np.maximum(pos, 0)[..., None], 1)
before = own - 1                                             # the parent's last position before this bracket opened
before2 = np.take_along_axis(own, np.maximum(before, 0), 1) - 1          # the grandparent's last position before the parent opened
lw = lambda w: np.clip(np.log(w + 1e-12) - np.log(w + 1e-12).mean(-1, keepdims=True), -6, 6)
inside, inside2 = depth >= 1, depth >= 2


def ancestor(j):
    """For each position deeper than j: the last position of its enclosing depth-j bracket before the chain of opens."""
    p = np.broadcast_to(np.arange(T), (N, T)).copy()
    for _ in range(proc.D):
        deeper = np.take_along_axis(depth, p, 1) > j
        p = np.where(deeper, np.take_along_axis(own, p, 1) - 1, p)
    return p


root, first = ancestor(0), ancestor(1)
rows = [
    ("OWN bracket: state per type", ref["state"], inside & (index >= 1)),
    ("OWN bracket: type log-posterior", lw(ref["w"]), inside & (index >= 1)),
    ("PARENT: state per type", at(ref["state"], before), inside),
    ("PARENT: type log-posterior", lw(at(ref["w"], before)), inside),
    ("PARENT: belief (type x state)", at(ref["belief"], before), inside),
    ("  parent state, just read the open", at(ref["state"], before), inside & (index == 0)),
    ("  parent state, 1-3 own tokens in", at(ref["state"], before), inside & (index >= 1) & (index <= 3)),
    ("  parent state, >= 4 own tokens in", at(ref["state"], before), inside & (index >= 4)),
    ("GRANDPARENT: state per type", at(ref["state"], before2), inside2),
    ("GRANDPARENT: type log-posterior", lw(at(ref["w"], before2)), inside2),
    # a stack with one slot per depth would not show in the rows above, where the parent's depth varies; these fix the depth
    ("ROOT bracket: state, from depth >= 1", at(ref["state"], root), inside),
    ("ROOT bracket: type, from depth >= 1", lw(at(ref["w"], root)), inside),
    ("ROOT bracket: state, from depth >= 2", at(ref["state"], root), inside2),
    ("ROOT bracket: type, from depth >= 2", lw(at(ref["w"], root)), inside2),
    ("DEPTH-1 bracket: state, from depth >= 2", at(ref["state"], first), inside2),
    ("DEPTH-1 bracket: type, from depth >= 2", lw(at(ref["w"], first)), inside2),
]
print(f"{run}: {args.get('process', 'mess3')}, {L} layers, d = {args['d']}; {N} sequences, probes fitted on the first {int(0.75 * N)}")
print("Probed from positions inside a bracket at depth >= 1 (>= 2 for the grandparent). The parent's values are those it held when")
print("this bracket opened; they do not change while the bracket is open, and they are what a close has to restore.\n")
print(f"{'':40s}{'positions':>10s}" + "".join(f"{('emb' if l == 0 else f'blk{l}'):>7s}" for l in range(L + 1)) + "   best untrained")
for name, Y, sel in rows:
    vals = [r2(resid[l], Y, sel) for l in range(L + 1)]
    ctrl = max(r2(blank_resid[l], Y, sel) for l in range(L + 1))
    print(f"{name:40s}{int(sel.sum()):>10d}" + "".join(f"{v:7.2f}" for v in vals) + f"   {ctrl:7.2f}")
