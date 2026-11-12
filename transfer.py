"""Is the state belief read by ONE linear map at every depth and for every bracket type?

validate.py fits each probe on all selected positions at once, so a high R^2 there could hide a map that
works at some depths and not others. This fits the same ridge probe on a subset of positions (one depth,
or brackets where a type is likely) and tests it on the others. If the belief sits in one subspace with
one meaning, a probe fitted anywhere works everywhere.

  python transfer.py runs/NAME

Target: the active bracket's state belief under each type hypothesis (K * S numbers), at positions with at
least one own token read. R^2 is on held-out sequences, around the mean of the tested positions.
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
K, S = proc.K, proc.S
toks, _ = proc.sample(N, T, np.random.default_rng(2027))
ref = algorithm.run(proc, toks)
depth, index, eta, w = ref["depth"], ref["index"], ref["state"], ref["w"]
model = load(run, args, proc)
torch.manual_seed(123)
blank = GPT(proc.VOCAB, T, args["d"], L, args["heads"]).eval()


def residuals(net, layers):
    out = {l: [] for l in layers}
    for a in range(0, N, 128):
        rs = trace(net, torch.from_numpy(ref["read"][a:a + 128]), want_attn=False)[1]
        for l in layers:
            out[l].append(rs[l].double().numpy())
    return {l: np.concatenate(v) for l, v in out.items()}


resid = residuals(model, (L - 1, L))
blank_resid = residuals(blank, (L,))
test = np.zeros((N, T), dtype=bool)
test[int(0.75 * N):] = True
base = index >= 1


def r2(X, Y, fit, tst, refit_offset=False):
    """refit_offset: keep the fitted directions but let the constant be re-estimated on the tested positions."""
    Y = Y.reshape(N, T, -1)
    fit, tst = fit & base & ~test, tst & base & test
    Xf, Yf, Xt, Yt = X[fit], Y[fit], X[tst], Y[tst]
    mx, my = Xf.mean(0), Yf.mean(0)
    W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + 1e-4 * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
    err = Yt - my - (Xt - mx) @ W
    if refit_offset:
        err = err - err.mean(0)
    return 1 - (err ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()


everywhere = np.ones((N, T), dtype=bool)
bins = [("depth 0", depth == 0), ("depth 1", depth == 1), ("depth 2", depth == 2), ("depth 3", depth == 3), ("depth >= 4", depth >= 4)]
print(f"{run}: {args.get('process', 'mess3')}, {L} layers, d = {args['d']}; {N} sequences, probes fitted on the first {int(0.75 * N)}")
print("fitting positions per depth: " + ", ".join(f"{n} {int((m & base & ~test).sum())}" for n, m in bins))

print("\n=== A. ACROSS DEPTH  (state belief per type; rows: where the probe was fitted; columns: where it is tested)")
for l, X, tag in [(L - 1, resid[L - 1], f"after block {L - 1}"), (L, resid[L], f"after block {L}")]:
    print(f"\n{tag:26s}" + "".join(f"{n:>12s}" for n, _ in bins))
    rows = [("all depths", lambda m: everywhere), ("the tested depth only", lambda m: m)] + [(f"{n} only", lambda m, f=f: f) for n, f in bins[:3]] \
        + [("depth >= 2 only", lambda m: depth >= 2)]
    for name, fit in rows:
        print(f"{name:26s}" + "".join(f"{r2(X, eta, fit(m), m):12.3f}" for _, m in bins))
    print("  same directions, constant re-estimated at the tested depth:")
    for name, fit in rows[2:]:
        print(f"{name:26s}" + "".join(f"{r2(X, eta, fit(m), m, True):12.3f}" for _, m in bins))
print(f"\n{'untrained, all depths':26s}" + "".join(f"{r2(blank_resid[L], eta, everywhere, m):12.3f}" for _, m in bins))

print("\n=== B. ACROSS TYPE  (state belief of type k; fitted and tested where k is likely (posterior > 0.5) or unlikely (< 0.1); mean over k)")
likely, unlikely = (lambda k: w[..., k] > 0.5), (lambda k: w[..., k] < 0.1)
for l, X, tag in [(L - 1, resid[L - 1], f"after block {L - 1}"), (L, resid[L], f"after block {L}")]:
    print(f"\n{tag:26s}{'k likely':>12s}{'k unlikely':>12s}")
    for name, fit in [("all positions", lambda k: everywhere), ("k likely only", likely), ("k unlikely only", unlikely)]:
        print(f"{name:26s}" + "".join(f"{np.mean([r2(X, eta[:, :, k], fit(k), c(k)) for k in range(K)]):12.3f}" for c in (likely, unlikely)))
print(f"\n{'untrained, all positions':26s}" + "".join(
    f"{np.mean([r2(blank_resid[L], eta[:, :, k], everywhere, c(k)) for k in range(K)]):12.3f}" for c in (likely, unlikely)))

print("\n=== C. SUBSPACES FITTED SEPARATELY AT EACH DEPTH  (where a type's state belief is written, as in validate.py section 5)")
print("Regress the residual on all types' state beliefs and the type posterior, using one depth's positions only; keep each")
print("type's S - 1 directions. Overlap = mean squared cosine of principal angles (1 = same subspace), mean over types.\n")
busy = index >= 2


def type_planes(X, mask):
    Z = np.concatenate([eta[mask].reshape(-1, K * S), w[mask]], 1)
    B = np.linalg.lstsq(Z - Z.mean(0), X[mask] - X[mask].mean(0), rcond=None)[0]
    return [np.linalg.svd(B[k * S:(k + 1) * S] - B[k * S:(k + 1) * S].mean(0), full_matrices=False)[2][:S - 1]
            for k in range(K)]


overlap = lambda P, Q: float(np.linalg.norm(P @ Q.T) ** 2 / len(P))
half = np.arange(N)[:, None] % 2 == 0
for l, X, tag in [(L - 1, resid[L - 1], f"after block {L - 1}"), (L, resid[L], f"after block {L}")]:
    P = [type_planes(X, busy & m) for _, m in bins]
    print(f"{tag:26s}" + "".join(f"{n:>12s}" for n, _ in bins))
    for i, (n, _) in enumerate(bins):
        print(f"{n:26s}" + "".join(f"{np.mean([overlap(P[i][k], P[j][k]) for k in range(K)]):12.2f}" for j in range(len(bins))))
    A, B_ = type_planes(X, busy & half), type_planes(X, busy & ~half)
    Pall = type_planes(X, busy)
    print(f"for scale: same type, two halves of the sequences {np.mean([overlap(A[k], B_[k]) for k in range(K)]):.2f}; "
          f"different types, all depths {np.mean([overlap(Pall[j], Pall[k]) for j in range(K) for k in range(j + 1, K)]):.2f}; "
          f"chance {(S - 1) / args['d']:.2f}\n")
