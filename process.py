"""Bracketed processes: batched sampler, exact Bayes filter, and forgetful observers.

This file is the ground truth. The filter here keeps an explicit stack, which a transformer cannot do;
algorithm.py computes the same beliefs with operations attention can do, and is checked against it.

A process is a set of K component processes that share a content vocabulary (K bracket types).
Each open bracket has a hidden type; the content tokens directly inside it are emitted by the
component of that type; a close token names the type. Brackets nest, and the depth-0 root never closes.

Components are generalized hidden Markov models: matrices T[token] with word probability
eta0 T[x1] ... T[xn] r. For an ordinary HMM r is all ones and eta0 the stationary distribution.
Tom Quantum is not an HMM (its matrices have negative entries), so sampling uses the belief vector.

Token ids: content 0..C-1, then the open token(s), then K close tokens. BOS (model input only) is last.

Observers (all know the grammar; they differ in what they keep across a close):
  bayes      exact filter: a returning bracket resumes with the belief it had before the child
  type_only  on a close keeps the parent's type posterior, forgets its component state
  amnesic    on a close resets the parent to the prior
  blind      one belief for the whole sequence, never pushed or popped
  prior      no inference at all (structure probabilities only)
"""
import numpy as np


class Component:
    def __init__(self, name, T):
        self.name, self.T = name, np.asarray(T, dtype=float)        # [C, S, S]
        A = self.T.sum(0)
        vals, right = np.linalg.eig(A)
        r = np.real(right[:, np.argmin(np.abs(vals - 1))])
        valsl, left = np.linalg.eig(A.T)
        eta0 = np.real(left[:, np.argmin(np.abs(valsl - 1))])
        r = r / r[np.argmax(np.abs(r))]                              # largest entry = 1 (ones for an HMM)
        self.r, self.eta0 = r, eta0 / (eta0 @ r)


def mess3(x, a, shift=0):
    """Mess3. T[tok, s, s'] = P(emit tok, go to s' | s); the emission depends on the destination state.
    shift != 0 gives a clock: the state advances by `shift` (mod 3) with probability 1 - 2x."""
    b = (1 - a) / 2
    trans = np.full((3, 3), float(x))
    for s in range(3):
        trans[s, (s + shift) % 3] = 1 - 2 * x
    T = np.zeros((3, 3, 3))
    for tok in range(3):
        emit = np.full(3, b)
        emit[tok] = a
        T[tok] = trans * emit[None, :]
    return Component(f"mess3({x},{a}" + (f",shift {shift})" if shift else ")"), T)


def tom_quantum(alpha, beta):
    """Tom Quantum, as in simplexity's transition_matrices.py. Four tokens, three-dimensional state."""
    g2 = 1 / (4 * (alpha ** 2 + beta ** 2))
    c, m, o = 0.25, (alpha ** 2 - beta ** 2) * g2, 2 * alpha * beta * g2
    T = [[[c, 0, o], [0, m, 0], [o, 0, c]],
         [[c, 0, -o], [0, m, 0], [-o, 0, c]],
         [[c, o, 0], [o, c, 0], [0, 0, m]],
         [[c, -o, 0], [-o, c, 0], [0, 0, m]]]
    return Component(f"tomq({alpha},{beta})", T)


def fanizza(alpha, lamb):
    """FRDN ("fern"), as `fanizza` in simplexity's transition_matrices.py. Two tokens, four-dimensional
    state. Token 0 resets the belief; token 1 rotates and shrinks it, so the belief is a function of how
    many 1s have come since the last 0."""
    c, s = np.cos(alpha), np.sin(alpha)
    den = 1 - 2 * lamb * c + lamb ** 2
    a_la, b_la = (1 - lamb * c + lamb * s) / den, (1 - lamb * c - lamb * s) / den
    pi0 = np.array([1 - (2 / (1 - lamb) - a_la - b_la) / 4, 1 / (2 * (1 - lamb)), -a_la / 4, -b_la / 4])
    w = np.array([1, 1 - lamb, 1 + lamb * (s - c), 1 - lamb * (s + c)])
    db = lamb * np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]])
    return Component(f"fanizza({alpha},{lamb})", np.stack([np.outer(w, pi0), db]))


def rrxor(p1=0.5, p2=0.5):
    """RRXOR, as in simplexity: two random bits, then their XOR. Two tokens, five states."""
    S, Z, O, Tr, Fa = range(5)
    T = np.zeros((2, 5, 5))
    T[0, S, Z], T[1, S, O] = p1, 1 - p1
    T[0, Z, Fa], T[1, Z, Tr] = p2, 1 - p2
    T[0, O, Tr], T[1, O, Fa] = p2, 1 - p2
    T[1, Tr, S], T[0, Fa, S] = 1.0, 1.0
    return Component("rrxor", T)


FAMILIES = {
    "mess3": lambda: [mess3(0.05, 0.85), mess3(0.15, 0.6), mess3(0.45, 0.85)],
    "mess3_sharp": lambda: [mess3(0.03, 0.9), mess3(0.2, 0.8), mess3(0.45, 0.9)],
    "mess3_slow": lambda: [mess3(0.02, 0.6), mess3(0.08, 0.6), mess3(0.25, 0.6)],
    "mess3_single": lambda: [mess3(0.05, 0.85)],
    "clock3": lambda: [mess3(0.05, 0.75), mess3(0.05, 0.75, shift=1), mess3(0.05, 0.75, shift=2)],
    # beta/alpha = tan(phi/2) for phi = 40, 80, 140 degrees; alpha = beta (phi = 90) has forbidden words
    "fern": lambda: [fanizza(2000, 0.49), fanizza(1.0, 0.49), fanizza(3.0, 0.45)],
    "tomq": lambda: [tom_quantum(1, 0.364), tom_quantum(1, 0.839), tom_quantum(1, 2.747)],
}


def make(name="mess3", **kw):
    """`name` is a family in FAMILIES; the suffix '+open' makes the open token reveal the type."""
    reveal = name.endswith("+open")
    return BracketProcess(FAMILIES[name[:-5] if reveal else name](), reveal_at_open=reveal, name=name, **kw)


class BracketProcess:
    def __init__(self, components, p_open=0.10, p_close=0.12, max_depth=8, reveal_at_open=False, name=""):
        self.comps, self.name = components, name
        self.K, self.reveal = len(components), reveal_at_open
        self.p_open, self.p_close, self.D = p_open, p_close, max_depth
        self.T = np.stack([c.T for c in components])                 # [K, C, S, S]
        self.eta0 = np.stack([c.eta0 for c in components])           # [K, S]
        self.r = np.stack([c.r for c in components])                 # [K, S]
        K, C, S = self.T.shape[:3]
        self.C, self.S = C, S
        self.n_open = K if reveal_at_open else 1
        self.OPEN0, self.CLOSE0 = C, C + self.n_open
        self.VOCAB = self.CLOSE0 + K
        self.BOS = self.VOCAB
        self.syms = "ABCDEFGH"[:C] + ("([{<"[:K] if reveal_at_open else "#") + ")]}>"[:K]
        self.emit = np.einsum("kxsr,kr->kxs", self.T, self.r)        # P(x | type k) as a weight on the state
        self.A = self.T.sum(1)                                       # [K, S, S] state map, token unseen
        # joint over (type, state), flattened to K*S
        self.M = np.zeros((C, K * S, K * S))
        for k in range(K):
            self.M[:, S * k:S * k + S, S * k:S * k + S] = self.T[k]
        self.E = self.emit.transpose(1, 0, 2).reshape(C, K * S)
        self.R = self.r.reshape(K * S)
        self.prior = (self.eta0 / K).reshape(K * S)
        self.typed_prior = np.zeros((K, K * S))                      # prior once the type is known
        for k in range(K):
            self.typed_prior[k, S * k:S * k + S] = self.eta0[k]

    # token classes
    def is_content(self, x):
        return x < self.C

    def is_open(self, x):
        return (x >= self.OPEN0) & (x < self.CLOSE0)

    def is_close(self, x):
        return (x >= self.CLOSE0) & (x < self.VOCAB)

    def struct_probs(self, depth):
        po = np.where(depth < self.D, self.p_open, 0.0)
        pc = np.where(depth > 0, self.p_close, 0.0)
        return po, pc

    def type_weights(self, belief):
        """Type posterior from joint beliefs [..., K*S]."""
        return (belief.reshape(belief.shape[:-1] + (self.K, self.S)) * self.r).sum(-1)

    def sample(self, batch, length, rng):
        """Returns tokens [B, L] and depth [B, L] = depth of the bracket that emitted each token
        (for an open, the depth of the parent; for a close, the depth of the bracket being closed)."""
        K, D, S, C = self.K, self.D, self.S, self.C
        typ = np.zeros((batch, D + 1), dtype=np.int64)
        eta = np.zeros((batch, D + 1, S))
        depth = np.zeros(batch, dtype=np.int64)
        rows = np.arange(batch)
        typ[:, 0] = rng.integers(0, K, batch)
        eta[:, 0] = self.eta0[typ[:, 0]]
        toks = np.zeros((batch, length), dtype=np.int64)
        deps = np.zeros((batch, length), dtype=np.int64)
        for t in range(length):
            po, pc = self.struct_probs(depth)
            u = rng.random(batch)
            opn = u < po
            cls = (u >= po) & (u < po + pc)
            emt = ~(opn | cls)
            k, e = typ[rows, depth], eta[rows, depth]
            deps[:, t] = depth
            probs = np.clip(np.einsum("bs,bxs->bx", e, self.emit[k]), 0, None)
            probs /= probs.sum(-1, keepdims=True)
            tok = (rng.random(batch)[:, None] > probs.cumsum(-1)).sum(-1).clip(0, C - 1)
            new = np.einsum("bs,bsr->br", e, self.T[k, tok])
            new /= (new * self.r[k]).sum(-1, keepdims=True)
            newk = rng.integers(0, K, batch)
            toks[:, t] = np.where(emt, tok, np.where(opn, self.OPEN0 + (newk if self.reveal else 0),
                                                    self.CLOSE0 + k))
            eta[rows, depth] = np.where(emt[:, None], new, e)
            depth = depth + opn - cls
            typ[rows[opn], depth[opn]] = newk[opn]
            eta[rows[opn], depth[opn]] = self.eta0[newk[opn]]
        return toks, deps

    def filter(self, toks, mode="bayes"):
        """Returns pred [B, L, VOCAB] with pred[:, t] = P(tok_t | tok_<t) under the observer,
        belief [B, L, K*S] = the active-bracket belief used for that prediction,
        resumed [B, L] = number of children of the active bracket that have closed so far."""
        batch, length = toks.shape
        K, S, D, C = self.K, self.S, self.D, self.C
        bel = np.tile(self.prior, (batch, D + 1, 1))
        nret = np.zeros((batch, D + 1), dtype=np.int64)
        depth = np.zeros(batch, dtype=np.int64)
        rows = np.arange(batch)
        pred = np.zeros((batch, length, self.VOCAB))
        beliefs = np.zeros((batch, length, K * S))
        resumed = np.zeros((batch, length), dtype=np.int64)
        for t in range(length):
            slot = np.zeros(batch, dtype=np.int64) if mode == "blind" else depth
            top = np.tile(self.prior, (batch, 1)) if mode == "prior" else bel[rows, slot]
            po, pc = self.struct_probs(depth)
            pred[:, t, :C] = (1 - po - pc)[:, None] * np.einsum("bi,xi->bx", top, self.E)
            pred[:, t, self.OPEN0:self.CLOSE0] = (po / self.n_open)[:, None]
            pred[:, t, self.CLOSE0:] = pc[:, None] * self.type_weights(top)
            beliefs[:, t] = top
            resumed[:, t] = nret[rows, depth]
            x = toks[:, t]
            emt, opn, cls = self.is_content(x), self.is_open(x), self.is_close(x)
            if emt.any():
                new = np.einsum("bi,bij->bj", top[emt], self.M[x[emt]])
                bel[rows[emt], slot[emt]] = new / (new @ self.R)[:, None]
            depth = depth + opn - cls
            if mode != "blind":
                r = rows[opn]
                bel[r, depth[opn]] = self.typed_prior[x[opn] - self.OPEN0] if self.reveal else self.prior
                nret[r, depth[opn]] = 0
                r = rows[cls]
                nret[r, depth[cls]] += 1
                if mode == "amnesic":
                    bel[r, depth[cls]] = self.prior
                elif mode == "type_only":
                    w = self.type_weights(bel[r, depth[cls]])
                    bel[r, depth[cls]] = (w[:, :, None] * self.eta0[None]).reshape(-1, K * S)
        return pred, beliefs, resumed

    def show(self, toks):
        return "".join("^" if t == self.BOS else self.syms[t] for t in toks)

    def encode(self, text):
        return np.array([self.syms.index(c) for c in text])


def nll(pred, toks):
    return -np.log(np.take_along_axis(pred, toks[..., None], -1)[..., 0])
