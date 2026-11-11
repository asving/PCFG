"""Bracketed-Mess3 process, with a batched sampler and exact Bayes filter.

Tokens: 0,1,2 = A,B,C   3 = '#' (open)   4,5,6 = close of type 0,1,2.

Each '#' opens a frame of hidden type k (uniform prior). The content tokens directly inside
a frame are emitted by Mess3 number k. Frames nest; the depth-0 root frame never closes.
The bracket skeleton is visible, the frames are independent given the skeleton, so the exact
posterior is a product over open frames of a belief over (type, Mess3 state): 9 numbers each.

Observers (all know the grammar; they differ in what they keep across a return):
  bayes      exact filter: a returning frame resumes with the belief it had at the call
  type_only  on return keeps the type posterior, forgets the Mess3 state
  amnesic    on return resets the parent to the prior (no memory of suspended frames)
  blind      one belief for the whole sequence, never pushed or popped
  prior      no inference at all (structure probabilities only)
"""
import numpy as np

A, B, C, OPEN = 0, 1, 2, 3
CLOSE0 = 4
VOCAB = 7
SYMS = "ABC#)]}"


def mess3(x, a):
    """T[tok, s, s'] = P(emit tok, go to s' | s). Emission depends on the destination state."""
    b = (1 - a) / 2
    trans = np.full((3, 3), x) + np.eye(3) * (1 - 3 * x)
    T = np.zeros((3, 3, 3))
    for tok in range(3):
        emit = np.full(3, b)
        emit[tok] = a
        T[tok] = trans * emit[None, :]
    return T


class BracketProcess:
    def __init__(self, params=((0.05, 0.85), (0.15, 0.6), (0.45, 0.85)),
                 p_open=0.10, p_close=0.12, max_depth=8):
        self.K = len(params)
        self.p_open, self.p_close, self.D = p_open, p_close, max_depth
        self.T = np.stack([mess3(x, a) for x, a in params])        # [K, tok, s, s']
        S = self.K * 3
        self.S = S
        # block-diagonal joint transition on (k, s): M[tok, (k,s), (k,s')]
        self.M = np.zeros((3, S, S))
        for k in range(self.K):
            self.M[:, 3 * k:3 * k + 3, 3 * k:3 * k + 3] = self.T[k]
        self.E = self.M.sum(-1)                                     # [tok, (k,s)] emission prob
        self.prior = np.full(S, 1.0 / S)

    def struct_probs(self, depth):
        po = np.where(depth < self.D, self.p_open, 0.0)
        pc = np.where(depth > 0, self.p_close, 0.0)
        return po, pc

    def sample(self, batch, length, rng):
        """Returns tokens [B, L] and depth [B, L] = depth of the frame that emitted each token
        (for '#', the depth of the parent; for a close, the depth of the frame being closed)."""
        K, D = self.K, self.D
        typ = np.zeros((batch, D + 1), dtype=np.int64)
        st = np.zeros((batch, D + 1), dtype=np.int64)
        depth = np.zeros(batch, dtype=np.int64)
        rows = np.arange(batch)
        typ[:, 0] = rng.integers(0, K, batch)
        st[:, 0] = rng.integers(0, 3, batch)
        toks = np.zeros((batch, length), dtype=np.int64)
        deps = np.zeros((batch, length), dtype=np.int64)
        # cumulative joint over (tok, s') for each (k, s)
        cum = self.T.transpose(0, 2, 1, 3).reshape(K, 3, 9).cumsum(-1)   # [K, s, tok*3+s']
        for t in range(length):
            po, pc = self.struct_probs(depth)
            u = rng.random(batch)
            is_open = u < po
            is_close = (u >= po) & (u < po + pc)
            is_emit = ~(is_open | is_close)
            k, s = typ[rows, depth], st[rows, depth]
            deps[:, t] = depth
            v = rng.random(batch)
            idx = (v[:, None] > cum[k, s]).sum(-1).clip(0, 8)
            tok, s2 = idx // 3, idx % 3
            toks[:, t] = np.where(is_emit, tok, np.where(is_open, OPEN, CLOSE0 + k))
            st[rows, depth] = np.where(is_emit, s2, s)
            depth = depth + is_open - is_close
            new = is_open
            typ[rows[new], depth[new]] = rng.integers(0, K, new.sum())
            st[rows[new], depth[new]] = rng.integers(0, 3, new.sum())
        return toks, deps

    def filter(self, toks, mode="bayes"):
        """Returns pred [B, L, VOCAB] with pred[:, t] = P(tok_t | tok_<t) under the observer,
        belief [B, L, S] = the active-frame belief used for that prediction,
        resumed [B, L] = number of returns the active frame has had so far."""
        batch, length = toks.shape
        K, S, D = self.K, self.S, self.D
        bel = np.tile(self.prior, (batch, D + 1, 1))
        nret = np.zeros((batch, D + 1), dtype=np.int64)
        depth = np.zeros(batch, dtype=np.int64)
        rows = np.arange(batch)
        pred = np.zeros((batch, length, VOCAB))
        beliefs = np.zeros((batch, length, S))
        resumed = np.zeros((batch, length), dtype=np.int64)
        for t in range(length):
            slot = np.zeros(batch, dtype=np.int64) if mode == "blind" else depth
            top = np.tile(self.prior, (batch, 1)) if mode == "prior" else bel[rows, slot]
            po, pc = self.struct_probs(depth)
            pred[:, t, :3] = (1 - po - pc)[:, None] * np.einsum("bi,xi->bx", top, self.E)
            pred[:, t, OPEN] = po
            pred[:, t, CLOSE0:] = pc[:, None] * top.reshape(batch, K, 3).sum(-1)
            beliefs[:, t] = top
            resumed[:, t] = nret[rows, depth]
            x = toks[:, t]
            emit, opn, cls = x < 3, x == OPEN, x >= CLOSE0
            if emit.any():
                r = rows[emit]
                new = np.einsum("bi,bij->bj", top[emit], self.M[x[emit]])
                bel[r, slot[emit]] = new / new.sum(-1, keepdims=True)
            depth = depth + opn - cls
            if mode != "blind":
                bel[rows[opn], depth[opn]] = self.prior
                nret[rows[opn], depth[opn]] = 0
                r = rows[cls]
                nret[r, depth[cls]] += 1
                if mode == "amnesic":
                    bel[r, depth[cls]] = self.prior
                elif mode == "type_only":
                    w = bel[r, depth[cls]].reshape(-1, K, 3).sum(-1, keepdims=True)
                    bel[r, depth[cls]] = np.repeat(w / 3, 3, axis=-1).reshape(-1, S)
        return pred, beliefs, resumed


def nll(pred, toks):
    return -np.log(np.take_along_axis(pred, toks[..., None], -1)[..., 0])


def show(toks):
    return "".join(SYMS[t] for t in toks)
