"""Coupled brackets with hidden types: the general visibly pushdown case with HMM-typed frames.

As in process.py, a bracket has a hidden type k and emits content from component k's HMM. Two changes:

  coupling      a child's type is drawn from PI[parent's type]
  hidden close  one close token for every type, so a child's type is never revealed

Exact prediction then needs two recursions (coupled.py has the slow reference and its brute-force check):

  down   a bracket's type prior is its parent's posterior at the open, pushed through PI; that posterior
         used the parent's own prior, which came from the grandparent, and so on
  up     when a child closes, its content is evidence about the parent's type: log(PI . L), where L is the
         likelihood of everything inside the child under each of its types, and "everything" includes what
         the child's own children told it

filter() returns the exact next-token law, the laws of observers that cut either recursion short, and the
quantities a probe can look for. CONFIGS holds the processes chosen by design_coupled.py.

  python vpa.py [config]      checks the batched filter against coupled.py and prints what each cut costs
"""
import sys
import numpy as np
from process import BracketProcess, Component, mess3

INF = 99
K3 = 3
I3, U3 = np.eye(K3), np.full((K3, K3), 1 / K3)
SHIFT = np.roll(I3, 1, axis=1)                               # child type = parent type + 1 (mod 3)
PARAMS = [(0.05, 0.85), (0.15, 0.6), (0.45, 0.85)]


def biased(x, a, k, beta):
    """A Mess3 that also, with probability beta, emits token k and stays where it is: type k favours token k,
    so a single token is already some evidence about the type."""
    T = (1 - beta) * mess3(x, a).T
    T[k] += beta * np.eye(3)
    return Component(f"mess3({x},{a})+{beta}*tok{k}", T)


CONFIGS = {
    # the child is its parent's type plus one with probability 0.95 (+ 0.05 / 3), otherwise uniform
    "vpa_shift": dict(components=lambda: [biased(x, a, k, 0.5) for k, (x, a) in enumerate(PARAMS)],
                      PI=0.95 * SHIFT + 0.05 * U3, p_open=0.35, p_close=0.36, max_depth=8),
    # weaker coupling and stronger token bias: here the shortcut of taking the coupling as exact is also expensive
    "vpa_shift2": dict(components=lambda: [biased(x, a, k, 0.7) for k, (x, a) in enumerate(PARAMS)],
                       PI=0.90 * SHIFT + 0.10 * U3, p_open=0.35, p_close=0.36, max_depth=8),
    # a noisy 3-cycle with unequal noise: no symmetry between the types
    "vpa_cycle": dict(components=lambda: [biased(x, a, k, 0.5) for k, (x, a) in enumerate(PARAMS)],
                      PI=np.array([[0.03, 0.95, 0.02], [0.05, 0.10, 0.85], [0.90, 0.04, 0.06]]),
                      p_open=0.30, p_close=0.31, max_depth=8),
}


def make(config, **over):
    c = dict(CONFIGS[config], **over)
    return VPA(c["components"](), c["PI"], c["p_open"], c["p_close"], c["max_depth"], name=config)


def rigid(config):
    """The same process with the coupling's noise removed: every bracket's type is then fixed by the root's
    type and its depth. Its exact law is the prediction of an observer that needs no bracket structure beyond
    depth, the simplest shortcut for a strongly coupled process."""
    c = CONFIGS[config]
    hard = np.eye(len(c["PI"]))[np.asarray(c["PI"]).argmax(-1)]
    return VPA(c["components"](), 0.999 * hard + 0.001 / len(hard), c["p_open"], c["p_close"], c["max_depth"], name=config + "_rigid")


class VPA(BracketProcess):
    def __init__(self, components, PI, p_open, p_close, max_depth=8, name=""):
        super().__init__(components, p_open=p_open, p_close=p_close, max_depth=max_depth, name=name)
        self.PI = np.asarray(PI, dtype=float)                 # [parent type, child type]
        self.cumPI = self.PI.cumsum(-1)
        self.hidden_close = True
        self.VOCAB = self.CLOSE0 + 1                          # content tokens, '#', one close
        self.BOS = self.VOCAB
        self.syms = "ABCDEFGH"[:self.C] + "#)"

    def sample(self, batch, length, rng):
        K, D, S, C = self.K, self.D, self.S, self.C
        typ = np.zeros((batch, D + 1), dtype=np.int64)
        eta = np.zeros((batch, D + 1, S))
        depth, rows = np.zeros(batch, dtype=np.int64), np.arange(batch)
        typ[:, 0] = rng.integers(0, K, batch)
        eta[:, 0] = self.eta0[typ[:, 0]]
        toks = np.zeros((batch, length), dtype=np.int64)
        deps = np.zeros((batch, length), dtype=np.int64)
        for t in range(length):
            po, pc = self.struct_probs(depth)
            u = rng.random(batch)
            opn, cls = u < po, (u >= po) & (u < po + pc)
            emt = ~(opn | cls)
            k, e = typ[rows, depth], eta[rows, depth]
            deps[:, t] = depth
            probs = np.clip(np.einsum("bs,bxs->bx", e, self.emit[k]), 0, None)
            probs /= probs.sum(-1, keepdims=True)
            tok = (rng.random(batch)[:, None] > probs.cumsum(-1)).sum(-1).clip(0, C - 1)
            new = np.einsum("bs,bsr->br", e, self.T[k, tok])
            new /= (new * self.r[k]).sum(-1, keepdims=True)
            newk = (rng.random(batch)[:, None] > self.cumPI[k]).sum(-1).clip(0, K - 1)
            toks[:, t] = np.where(emt, tok, np.where(opn, self.OPEN0, self.CLOSE0))
            eta[rows, depth] = np.where(emt[:, None], new, e)
            depth = depth + opn - cls
            typ[rows[opn], depth[opn]] = newk[opn]
            eta[rows[opn], depth[opn]] = self.eta0[newk[opn]]
        return toks, deps

    def filter(self, toks, downs=(0, 1, 2, 3, 4, 5), ups=(0, 1, 2, 3)):
        """Returns (preds, ref).
        preds: next-token laws [B, L, VOCAB] of "exact"; "ignore" (neither recursion); ("down", h): at most h
               reads down, the up recursion complete; ("up", u): at most u levels up, the down chain complete;
               "prior": no inference at all.
        ref:   for the exact observer at each position: depth; prior, ll, msg, w [B, L, K] (the active bracket's
               type prior, the log-likelihood of its own tokens, the summed log-messages of its closed children,
               and its type posterior); state [B, L, K, S]; close_msg [B, L, K], the log-message of the child
               that the position has just seen close (zero elsewhere)."""
        assert list(downs) == list(range(len(downs))) and list(ups) == list(range(len(ups)))
        B, L = toks.shape
        K, S, D, C = self.K, self.S, self.D, self.C
        nH, nU = len(downs) + 1, len(ups) + 1                 # the last index of each is "unlimited"
        v = np.broadcast_to(self.eta0, (B, D + 1, K, S)).copy()
        ll = np.zeros((B, D + 1, K))
        msg = np.zeros((B, D + 1, nU, K))
        prior_h = np.full((B, D + 1, nH, K), 1.0 / K)         # down limit h, up complete
        prior_u = np.full((B, D + 1, nU, K), 1.0 / K)         # up limit u, down complete
        depth, rows = np.zeros(B, dtype=np.int64), np.arange(B)
        names = ["exact", "ignore", "prior"] + [("down", h) for h in downs] + [("up", u) for u in ups]
        preds = {n: np.zeros((B, L, self.VOCAB)) for n in names}
        ref = dict(depth=np.zeros((B, L), dtype=np.int64), prior=np.zeros((B, L, K)), ll=np.zeros((B, L, K)),
                   msg=np.zeros((B, L, K)), w=np.zeros((B, L, K)), state=np.zeros((B, L, K, S)), close_msg=np.zeros((B, L, K)))
        pending = np.zeros((B, K))

        def post(prior, l, m):
            lw = np.log(prior) + l + m
            w = np.exp(lw - lw.max(-1, keepdims=True))
            return w / w.sum(-1, keepdims=True)

        flat_emit = np.einsum("ks,kxs->kx", self.eta0, self.emit).mean(0)
        for t in range(L):
            top, l, m = v[rows, depth], ll[rows, depth], msg[rows, depth]
            ph, pu = prior_h[rows, depth], prior_u[rows, depth]
            po, pc = self.struct_probs(depth)
            emit_k = np.einsum("bks,kxs->bkx", top, self.emit)
            ws = {"exact": post(ph[:, -1], l, m[:, -1]), "ignore": post(np.full((B, K), 1.0 / K), l, 0.0)}
            for i, h in enumerate(downs):
                ws[("down", h)] = post(ph[:, i], l, m[:, -1])
            for i, u in enumerate(ups):
                ws[("up", u)] = post(pu[:, i], l, m[:, i])
            for n, w in ws.items():
                preds[n][:, t, :C] = (1 - po - pc)[:, None] * np.einsum("bk,bkx->bx", w, emit_k)
            preds["prior"][:, t, :C] = (1 - po - pc)[:, None] * flat_emit
            for n in names:
                preds[n][:, t, self.OPEN0] = po
                preds[n][:, t, self.CLOSE0] = pc
            ref["depth"][:, t], ref["prior"][:, t], ref["ll"][:, t], ref["msg"][:, t] = depth, ph[:, -1], l, m[:, -1]
            ref["w"][:, t], ref["state"][:, t], ref["close_msg"][:, t] = ws["exact"], top, pending
            pending = np.zeros((B, K))

            x = toks[:, t]
            emt, opn, cls = self.is_content(x), self.is_open(x), self.is_close(x)
            if emt.any():
                new = np.einsum("bks,kbsr->bkr", top[emt], self.T[:, x[emt]])
                mass = np.maximum((new * self.r).sum(-1), 1e-300)
                ll[rows[emt], depth[emt]] += np.log(mass)
                v[rows[emt], depth[emt]] = new / mass[..., None]
            if cls.any():                                     # the child at depth d tells its parent at d - 1
                r, d = rows[cls], depth[cls]
                for i in range(1, nU):                        # u = 0 learns nothing from a close
                    src = m[cls][:, -1] if i == nU - 1 else m[cls][:, i - 1]
                    tot = l[cls] + src
                    mx = tot.max(-1, keepdims=True)
                    message = np.log(np.exp(tot - mx) @ self.PI.T) + mx
                    msg[r, d - 1, i] += message
                    if i == nU - 1:
                        pending[cls] = message - message.mean(-1, keepdims=True)
            if opn.any():
                r, d = rows[opn], depth[opn]
                for i in range(nH):
                    if i == 0:
                        child = np.full((opn.sum(), K), 1.0 / K)
                    else:
                        j = nH - 1 if i == nH - 1 else i - 1
                        child = post(ph[opn][:, j], l[opn], m[opn][:, -1]) @ self.PI
                    prior_h[r, d + 1, i] = child
                for i in range(nU):
                    prior_u[r, d + 1, i] = post(pu[opn][:, i], l[opn], m[opn][:, i]) @ self.PI
                v[r, d + 1], ll[r, d + 1], msg[r, d + 1] = self.eta0, 0.0, 0.0
            depth = depth + opn - cls
        return preds, ref


if __name__ == "__main__":
    import algorithm
    from coupled import Coupled, INF as CINF
    config = sys.argv[1] if len(sys.argv) > 1 else "vpa_shift"
    proc = make(config)
    toks, _ = proc.sample(1024, 128, np.random.default_rng(12345))
    preds, ref = proc.filter(toks)
    slow = Coupled(CONFIGS[config]["components"](), proc.PI, False, p_open=proc.p_open, p_close=proc.p_close, max_depth=proc.D)
    pairs = [("exact", (CINF, CINF)), ("ignore", (0, 0)), (("down", 1), (1, CINF)), (("down", 3), (3, CINF)), (("up", 1), (CINF, 1)), (("up", 2), (CINF, 2))]
    worst = 0.0
    for b in range(40):
        s = slow.filter(toks[b], [o for _, o in pairs])
        worst = max(worst, max(np.abs(preds[n][b] - s[o]).max() for n, o in pairs))
    print(f"{config}: batched filter vs coupled.py on 40 sequences, six observers: max |difference| {worst:.1e}")
    st = algorithm.run(proc, toks)
    print(f"algorithm.py's structure on this process: depth agrees {bool((st['depth'] == ref['depth']).all())}; "
          f"own-token evidence agrees to {np.abs((st['logw'] - st['logw'].mean(-1, keepdims=True)) - (ref['ll'] - ref['ll'].mean(-1, keepdims=True))).max():.1e}; "
          f"per-type state to {np.abs(st['state'] - ref['state']).max():.1e}")
    depth = ref["depth"]
    cells = {"all": np.ones_like(depth, dtype=bool), "depth >= 3": depth >= 3, "depth >= 5": depth >= 5,
             "just read a close": proc.is_close(algorithm.read_tokens(proc, toks))}
    nll = -np.log(np.take_along_axis(preds["exact"], toks[..., None], -1)).mean()
    print(f"\n1024 sequences of 128; exact Bayes {nll:.3f} nats/token; content {proc.is_content(toks).mean():.2f}, opens {proc.is_open(toks).mean():.2f}, closes {proc.is_close(toks).mean():.2f} of tokens")
    print("share of positions by depth 0-8: " + " ".join(f"{(depth == d).mean():.2f}" for d in range(9)))
    print(f"\n{'excess loss over exact, mnat/token':38s}" + "".join(f"{c:>20s}" for c in cells))
    preds["rigid"] = rigid(config).filter(toks)[0]["exact"]
    label = {"prior": "no inference at all", "ignore": "neither recursion", "rigid": "coupling taken as exact"}
    for n in ["prior", "ignore", "rigid"] + [("down", h) for h in range(6)] + [("up", u) for u in range(4)]:
        name = label.get(n) or (f"{n[1]} read{'s' if n[1] != 1 else ''} down, all up" if n[0] == "down" else f"all down, {n[1]} level{'s' if n[1] != 1 else ''} up")
        kl = algorithm.excess(preds["exact"], preds[n])
        print(f"{name:38s}" + "".join(f"{1e3 * kl[m].mean():20.1f}" for m in cells.values()))
