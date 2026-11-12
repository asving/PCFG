"""Which coupled bracket process gives a network a reason to learn both recursions? No training.

Brackets are typed by a hidden HMM as before, but now a child's type is drawn from PI[parent type], and
the close token does not say what the child was. Exact prediction then needs two recursions (coupled.py):

  down   a bracket's prior comes from its parent's posterior at the open, which used the grandparent's ...
  up     when a child closes, the parent learns from the child's content, which includes what the child
         learned from its own children ...

For each candidate this samples sequences from the process itself and prints what it costs, in excess loss
over exact Bayes (mnat/token, all positions), to cut either recursion short. A process is worth training
on when the later levels of BOTH are expensive, not only the first.

  python design_coupled.py [n_seq]

Columns:
  ignore        uses neither recursion (every bracket starts from a uniform prior, a close teaches nothing)
  down 0..3     reads down allowed, with the up recursion complete
  up 0..3       levels up allowed, with the down chain complete
"""
import sys
import numpy as np
from process import Component, mess3, FAMILIES
from coupled import Coupled, INF, kl

K = 3


def biased(x, a, k, beta):
    """A Mess3 that also, with probability beta, emits token k and stays where it is: type k favours token k,
    so that a single token is already some evidence about the type."""
    T = (1 - beta) * mess3(x, a).T
    T[k] += beta * np.eye(3)
    return Component(f"mess3({x},{a})+{beta}*tok{k}", T)


PARAMS = [(0.05, 0.85), (0.15, 0.6), (0.45, 0.85)]
COMPONENTS = {
    "mess3": FAMILIES["mess3"],
    "mess3 bias 0.2": lambda: [biased(x, a, k, 0.2) for k, (x, a) in enumerate(PARAMS)],
    "mess3 bias 0.4": lambda: [biased(x, a, k, 0.4) for k, (x, a) in enumerate(PARAMS)],
    "mess3 bias 0.6": lambda: [biased(x, a, k, 0.6) for k, (x, a) in enumerate(PARAMS)],
}
I3, U3 = np.eye(K), np.full((K, K), 1 / K)
SHIFT = np.roll(I3, 1, axis=1)                                # child type = parent type + 1
COUPLINGS = {
    "same 0.9": 0.9 * I3 + 0.1 * U3,
    "same 0.7": 0.7 * I3 + 0.3 * U3,
    "shift 0.9": 0.9 * SHIFT + 0.1 * U3,
    "lopsided": np.array([[0.85, 0.10, 0.05], [0.10, 0.45, 0.45], [0.45, 0.45, 0.10]]),
}
RATES = [(0.10, 0.12), (0.20, 0.22), (0.30, 0.32), (0.30, 0.40)]
DOWN = [(h, INF) for h in (0, 1, 2, 3)]
UP = [(INF, u) for u in (0, 1, 2, 3)]
OBS = [(0, 0)] + DOWN + UP + [(INF, INF)]


def price(comps, PI, po, pc, n_seq, seed=0, length=128):
    proc = Coupled(comps, PI, False, p_open=po, p_close=pc, max_depth=8)
    rng = np.random.default_rng(seed)
    tot, n, nll, deep, own = {o: 0.0 for o in OBS}, 0, 0.0, 0, []
    for _ in range(n_seq):
        toks = proc.sample(length, rng)
        preds = proc.filter(toks, OBS)
        exact = preds[(INF, INF)]
        for o in OBS[:-1]:
            tot[o] += kl(exact, preds[o]).sum()
        nll -= np.log(exact[np.arange(length), toks]).sum()
        depth = np.concatenate([[0], np.cumsum((toks == proc.OPEN).astype(int) - (toks >= proc.CLOSE0))[:-1]])
        deep += (depth >= 3).sum()
        n += length
    return {o: 1e3 * v / n for o, v in tot.items()}, nll / n, deep / n


if __name__ == "__main__":
    n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    print(f"{n_seq} sequences of 128 per candidate; closes never reveal the type; depth capped at 8; excess loss in mnat/token\n")
    print(f"{'components':16s}{'coupling':11s}{'open,close':>11s}{'bayes':>7s}{'depth>=3':>9s}{'ignore':>8s} |"
          + "".join(f"{'down ' + str(h):>8s}" for h, _ in DOWN) + " |" + "".join(f"{'up ' + str(u):>7s}" for _, u in UP))
    for cname, make_c in COMPONENTS.items():
        for pname, PI in COUPLINGS.items():
            for po, pc in RATES:
                v, nll, deep = price(make_c(), PI, po, pc, n_seq)
                print(f"{cname:16s}{pname:11s}{f'{po:.2f},{pc:.2f}':>11s}{nll:7.3f}{deep:9.2f}{v[(0, 0)]:8.1f} |"
                      + "".join(f"{v[o]:8.1f}" for o in DOWN) + " |" + "".join(f"{v[o]:7.1f}" for o in UP), flush=True)
