"""Coupled brackets: what does truncated inference cost? A design study with no training.

Here a child's type depends on its parent's type through PI[k_parent, k_child]. Two variants:
  revealed   the close token names the child's type (as in the main process)
  hidden     one close token for all types: a child's type is never revealed

The belief of the active bracket is still a local, sufficient statistic (checked below against
brute-force enumeration), but computing it now takes sequential reads:

  down   the child's prior is PI applied to the parent's type posterior at the open, which in turn
         used the grandparent's ... one read per nesting level
  up     (hidden only) when a child closes, the parent learns from the child's content, which
         includes what the child learned from its own children ... one read per level of height

A fixed-depth network can chain only so many reads, so this prints the excess loss of observers that
stop after h reads down and u reads up. Coupling is worth an experiment only where these are large.

  python coupled.py [n_seq]
"""
import itertools, sys
import numpy as np
from process import FAMILIES

INF = 99


class Coupled:
    def __init__(self, comps, PI, reveal, p_open=0.10, p_close=0.12, max_depth=8):
        self.T = np.stack([c.T for c in comps])                      # [K, C, S, S]
        self.eta0, self.r = np.stack([c.eta0 for c in comps]), np.stack([c.r for c in comps])
        self.K, self.C, self.S = self.T.shape[:3]
        self.emit = np.einsum("kxsr,kr->kxs", self.T, self.r)
        self.PI, self.reveal = np.asarray(PI, float), reveal
        self.p_open, self.p_close, self.D = p_open, p_close, max_depth
        self.OPEN, self.CLOSE0 = self.C, self.C + 1
        self.VOCAB = self.C + 1 + (self.K if reveal else 1)

    def rates(self, depth):
        return (self.p_open if depth < self.D else 0.0), (self.p_close if depth > 0 else 0.0)

    def sample(self, length, rng):
        stack = [(int(rng.integers(self.K)), None)]
        stack[0] = (stack[0][0], self.eta0[stack[0][0]].copy())
        out = []
        for _ in range(length):
            k, eta = stack[-1]
            po, pc = self.rates(len(stack) - 1)
            u = rng.random()
            if u < po:
                kc = int(rng.choice(self.K, p=self.PI[k]))
                stack.append((kc, self.eta0[kc].copy()))
                out.append(self.OPEN)
            elif u < po + pc:
                stack.pop()
                out.append(self.CLOSE0 + (k if self.reveal else 0))
            else:
                p = np.clip(eta @ self.emit[k].T, 0, None)
                x = int(rng.choice(self.C, p=p / p.sum()))
                new = eta @ self.T[k, x]
                stack[-1] = (k, new / (new @ self.r[k]))
                out.append(x)
        return np.array(out)

    def filter(self, toks, observers):
        """observers: list of (h, u) = reads allowed down and up (INF = unlimited).
        Returns {(h, u): pred [L, VOCAB]}."""
        K = self.K
        hs = sorted({x for h, _ in observers for x in ([INF] if h == INF else range(h + 1))})
        us = sorted({x for _, u in observers for x in ([INF] if u == INF else range(u + 1))})
        uniform = np.full(K, 1 / K)

        def new_frame(prior):
            return dict(v=self.eta0.copy(), loglik=np.zeros(K), logmsg={u: np.zeros(K) for u in us}, prior=prior)

        def post(f, h, u):
            lw = np.log(f["prior"][(h, u)]) + f["loglik"] + f["logmsg"][u]
            w = np.exp(lw - lw.max())
            return w / w.sum()

        stack = [new_frame({(h, u): uniform for h in hs for u in us})]
        preds = {o: np.zeros((len(toks), self.VOCAB)) for o in observers}
        for t, x in enumerate(toks):
            f = stack[-1]
            po, pc = self.rates(len(stack) - 1)
            for o in observers:
                w = post(f, *o)
                preds[o][t, :self.C] = (1 - po - pc) * np.einsum("k,ks,kxs->x", w, f["v"], self.emit)
                preds[o][t, self.OPEN] = po
                preds[o][t, self.CLOSE0:] = pc * w if self.reveal else pc
            if x < self.C:
                new = np.einsum("ks,ksr->kr", f["v"], self.T[:, x])
                mass = (new * self.r).sum(-1)
                f["loglik"] = f["loglik"] + np.log(np.maximum(mass, 1e-300))
                f["v"] = new / np.maximum(mass, 1e-300)[:, None]
            elif x == self.OPEN:
                prior = {}
                for h in hs:
                    for u in us:
                        prior[(h, u)] = uniform if h == 0 else self.PI.T @ post(f, INF if h == INF else h - 1, u)
                stack.append(new_frame(prior))
            else:
                child = stack.pop()
                parent = stack[-1]
                for u in us:
                    if u == 0:
                        continue                                    # no read up: the close teaches nothing
                    if self.reveal:
                        parent["logmsg"][u] = parent["logmsg"][u] + np.log(self.PI[:, x - self.CLOSE0])
                    else:
                        ll = child["loglik"] + child["logmsg"][INF if u == INF else u - 1]
                        parent["logmsg"][u] = parent["logmsg"][u] + np.log(self.PI @ np.exp(ll - ll.max()))
        return preds

    def brute(self, toks):
        """Exact P(next content token | toks) by enumerating the type of every bracket."""
        parent, own, known, stack = [-1], [[]], [None], [0]
        for x in toks:
            if x == self.OPEN:
                parent.append(stack[-1]); own.append([]); known.append(None); stack.append(len(parent) - 1)
            elif x >= self.CLOSE0:
                f = stack.pop()
                known[f] = x - self.CLOSE0 if self.reveal else None
            else:
                own[stack[-1]].append(x)
        active, F = stack[-1], len(parent)
        fwd = np.zeros((F, self.K, self.S))
        for f in range(F):
            for k in range(self.K):
                v = self.eta0[k].copy()
                for x in own[f]:
                    v = v @ self.T[k, x]
                fwd[f, k] = v
        num, den = np.zeros(self.C), 0.0
        for ks in itertools.product(range(self.K), repeat=F):
            if any(known[f] is not None and known[f] != ks[f] for f in range(F)):
                continue
            w = 1.0
            for f in range(F):
                w *= (1 / self.K if parent[f] < 0 else self.PI[ks[parent[f]], ks[f]])
                if f != active:
                    w *= fwd[f, ks[f]] @ self.r[ks[f]]
            v = fwd[active, ks[active]]
            den += w * (v @ self.r[ks[active]])
            num += w * (v @ self.emit[ks[active]].T)
        return num / den


def kl(p, q):
    return (p * (np.log(np.where(p > 0, p, 1.0)) - np.log(np.where(q > 0, q, 1e-300)))).sum(-1)


def study(label, PI, reveal, n_seq, rng):
    proc = Coupled(FAMILIES["mess3"](), PI, reveal)
    if reveal:
        rows = [("ignores the coupling", (0, 0)), ("uses each child's revealed type", (0, INF)),
                ("+ prior from the parent, 1 read down", (1, INF)), ("+ 2 reads down", (2, INF)),
                ("+ 3 reads down", (3, INF))]
    else:
        rows = [("ignores the coupling", (0, 0)),
                ("child's own tokens inform the parent", (0, 1)), ("child's full evidence, any height", (0, INF)),
                ("1 read down, no reads up", (1, 0)), ("1 read down, 1 up", (1, 1)),
                ("1 read down, all up", (1, INF)), ("2 reads down, all up", (2, INF)),
                ("all down, 1 up", (INF, 1)), ("all down, 2 up", (INF, 2))]
    obs = [o for _, o in rows] + [(INF, INF)]
    tot = {o: np.zeros(3) for o in obs}
    cnt = np.zeros(3)
    for _ in range(n_seq):
        toks = proc.sample(128, rng)
        preds = proc.filter(toks, obs)
        depth = np.concatenate([[0], np.cumsum((toks == proc.OPEN).astype(int) - (toks >= proc.CLOSE0))[:-1]])
        since_open = np.zeros(len(toks), dtype=int)                 # content tokens read since own open
        stack = [0]
        for t, x in enumerate(toks):
            since_open[t] = stack[-1]
            if x == proc.OPEN:
                stack.append(0)
            elif x >= proc.CLOSE0:
                stack.pop()
            else:
                stack[-1] += 1
        after_close = np.concatenate([[False], toks[:-1] >= proc.CLOSE0])
        masks = [np.ones(len(toks), bool), (depth >= 2) & (since_open <= 2), after_close]
        cnt += [m.sum() for m in masks]
        for o in obs:
            d = kl(preds[(INF, INF)], preds[o])
            tot[o] += [d[m].sum() for m in masks]
    print(f"\n{label}   ({'close reveals the type' if reveal else 'type never revealed'})")
    print(f"{'excess loss over exact, mnat/token':42s}{'all':>10s}{'new bracket, depth >= 2':>26s}{'just after a close':>21s}")
    for name, o in rows:
        v = 1e3 * tot[o] / cnt
        print(f"{name:42s}{v[0]:10.1f}{v[1]:26.1f}{v[2]:21.1f}")


if __name__ == "__main__":
    n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(0)
    K = 3
    same = lambda rho: rho * np.eye(K) + (1 - rho) / K          # child has the parent's type w.p. rho, else uniform

    worst, n = 0.0, 0
    for reveal in (True, False):
        chk = Coupled(FAMILIES["mess3"](), same(0.7), reveal, p_open=0.25, p_close=0.25, max_depth=4)
        while n < (150 if reveal else 300):
            toks = chk.sample(int(rng.integers(6, 20)), rng)
            if (toks == chk.OPEN).sum() > 8:
                continue
            depth = (toks == chk.OPEN).sum() - (toks >= chk.CLOSE0).sum()
            po, pc = chk.rates(depth)
            ext = np.concatenate([toks, [0]])
            got = chk.filter(ext, [(INF, INF)])[(INF, INF)][-1, :chk.C] / (1 - po - pc)
            worst = max(worst, np.abs(got - chk.brute(toks)).max())
            n += 1
    print(f"exact coupled filter vs brute-force enumeration on {n} random nested strings: max |difference| {worst:.1e}")

    for rho in (0.5, 0.9):
        for reveal in (True, False):
            study(f"child has its parent's type with probability {rho}, otherwise uniform", same(rho), reveal, n_seq, rng)
