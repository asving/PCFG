"""How big is each piece of the belief, in nats? Compares the exact filter with ablated observers.

usage: python scope_signal.py [n_seq] [length]
"""
import sys
import numpy as np
from process import BracketProcess, nll, show, OPEN, CLOSE0

n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
length = int(sys.argv[2]) if len(sys.argv) > 2 else 128

SETTINGS = {
    "classic  (.05,.85)(.15,.6)(.45,.85)": ((0.05, 0.85), (0.15, 0.6), (0.45, 0.85)),
    "sharp    (.03,.9)(.2,.8)(.45,.9)": ((0.03, 0.9), (0.2, 0.8), (0.45, 0.9)),
}

for name, params in SETTINGS.items():
    proc = BracketProcess(params=params)
    rng = np.random.default_rng(0)
    toks, deps = proc.sample(n_seq, length, rng)
    out = {m: proc.filter(toks, m) for m in ["bayes", "type_only", "amnesic", "blind", "prior"]}
    loss = {m: nll(o[0], toks) for m, o in out.items()}
    resumed = out["bayes"][2]
    content, close = toks < 3, toks >= CLOSE0
    print(f"\n=== {name}   p_open {proc.p_open} p_close {proc.p_close} max_depth {proc.D}  "
          f"{n_seq} x {length} tokens")
    print("sample:", show(toks[0][:90]))
    frac = np.bincount(deps.ravel(), minlength=proc.D + 1) / deps.size
    print("token share by depth 0..D:", " ".join(f"{f:.3f}" for f in frac))
    print(f"share of tokens: content {content.mean():.3f}  open {(toks == OPEN).mean():.3f}  "
          f"close {close.mean():.3f};  in a resumed frame {(resumed > 0).mean():.3f}")
    base = loss["bayes"]
    print(f"Bayes loss {base.mean():.4f} nats/token   (prior-only observer: +{(loss['prior'] - base).mean():.4f})")
    print(f"{'observer':10s} {'all tokens':>11s} {'resumed':>9s} {'1st after ret':>14s} "
          f"{'close in resumed':>17s}   (excess nats/token over Bayes, +- s.e.)")
    first = np.zeros_like(toks, dtype=bool)
    first[:, 1:] = toks[:, :-1] >= CLOSE0
    for m in ["type_only", "amnesic", "blind"]:
        d = loss[m] - base
        cells = [d, d[resumed > 0], d[first], d[(resumed > 0) & close]]
        se = d.reshape(n_seq, -1).mean(1).std() / np.sqrt(n_seq)
        print(f"{m:10s} " + " ".join(f"{c.mean():>{w}.4f}" for c, w in zip(cells, [11, 9, 14, 17]))
              + f"   +-{se:.4f}")
    # how fast is the type identified? One uninterrupted frame, by number of content tokens seen.
    flat = BracketProcess(params=params, p_open=0.0, p_close=0.0)
    ft, _ = flat.sample(n_seq, 33, np.random.default_rng(1))
    w = flat.filter(ft)[1].reshape(n_seq, 33, flat.K, 3).sum(-1)
    ent = -(w * np.log(w + 1e-12)).sum(-1).mean(0)
    print("type-posterior entropy after n own tokens (nats; ln 3 = 1.099):",
          " ".join(f"{n}:{ent[n]:.2f}" for n in [2, 4, 8, 16, 32]),
          "| P(max w > .95):", " ".join(f"{n}:{(w[:, n].max(-1) > .95).mean():.2f}" for n in [4, 8, 16, 32]))
