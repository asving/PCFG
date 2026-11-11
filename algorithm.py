"""The proposed algorithm: exact next-token prediction with no stack and no recurrence.

Position i has read tokens 0..i-1 (position 0 has read only BOS) and predicts token i. Every quantity
below is computed from the tokens with three operations that attention can perform:

  count        a running sum over earlier positions                 (even attention)
  most recent  the latest earlier position matching a key           (content match + recency)
  gather       a sum over the earlier positions that share a tag    (content match)

Steps, and what each leaves at position i:

  1 depth      opens minus closes among the tokens read
  2 tag        position of the '#' that opened i's bracket: the most recent open at depth[i]
               (BOS plays the '#' of the root). Positions with equal tags form one bracket's thread.
  3 index      how many ordinary tokens of its own bracket i has read
  4 state      for each type k, the Mess3-k state belief from the thread's ordinary tokens in order
  5 evidence   at each ordinary token, log P(that token | its bracket so far, type k)
  6 type       log-posterior of the type = sum of the thread's evidence (no decay)
  7 belief     type posterior x state belief, and from it the next-token law

Nothing is done at a close: the depth drops, so step 2 returns the parent's tag and steps 3-7 pick up
the parent's tokens again.

run() also implements the shortcuts a network might take instead, to predict what each would cost:

  tag="depth"        no bracket tag: all earlier positions at the same depth are treated as one thread
  clock="position"   recency counted in positions, not thread tokens: the state belief relaxes across
                     the positions that separate two consecutive own tokens (a child in between)
  window=m           the state belief uses only the last m own tokens

  python algorithm.py        checks run() against the stack filter, prints a worked example, and the
                             excess loss of each shortcut by position class
"""
import numpy as np
from process import BracketProcess, BOS, OPEN, CLOSE0, VOCAB, SYMS, encode

STEPS = ["depth", "tag", "index", "state", "evidence", "type", "belief"]


def read_tokens(toks):
    """Token read at each position: BOS, then tokens 0..T-2."""
    return np.concatenate([np.full((len(toks), 1), BOS), toks[:, :-1]], 1)


def run(proc, toks, tag="bracket", clock="thread", window=None, chunk=256):
    outs = [_run(proc, toks[a:a + chunk], tag, clock, window) for a in range(0, len(toks), chunk)]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _run(proc, toks, tag, clock, window):
    B, T = toks.shape
    K = proc.K
    xin = read_tokens(toks)
    J = np.arange(T)
    visible = J[None, :] <= J[:, None]                      # [i, j]: j is at or before i
    is_open, is_close, ordinary = xin == OPEN, (xin >= CLOSE0) & (xin < BOS), xin < 3

    # 1. depth: a running count
    depth = np.cumsum(is_open.astype(np.int64) - is_close, 1)
    same_depth = depth[:, :, None] == depth[:, None, :]     # [b, i, j]

    # 2. tag: the most recent open at my depth; equal tags = same bracket
    openlike = is_open | (J == 0)[None]
    own = np.where(openlike[:, None, :] & same_depth & visible, J, -1).max(-1)
    member = (same_depth if tag == "depth" else own[:, :, None] == own[:, None, :]) & visible
    prev = np.where(member & (J[None, :] < J[:, None]), J, -1).max(-1)      # previous position in thread

    # 3. index: a count over the thread
    own_tok = member & ordinary[:, None, :]
    index = own_tok.sum(-1)

    # 4. state belief per type, from the thread's ordinary tokens, most recent first in `order`
    kmax = max(int(index.max()), 1)
    order = -np.sort(-np.where(own_tok, J, -1), axis=-1)[..., :kmax + 1]
    if order.shape[-1] < kmax + 1:
        order = np.concatenate([order, np.full((B, T, kmax + 1 - order.shape[-1]), -1)], -1)
    m = kmax if window is None else min(window, kmax)
    Tm = proc.T                                             # [K, tok, s, s']
    A = Tm.sum(1)                                           # [K, s, s'] state transition, token unseen
    Apow = np.stack([np.stack([np.linalg.matrix_power(A[k], g) for g in range(T + 1)]) for k in range(K)])

    def state(cols, then=None):
        """Belief after the tokens at positions cols[..., c], oldest (last column) to newest."""
        v = np.full((B, T, K, 3), 1 / 3)
        for c in range(cols.shape[-1] - 1, -1, -1):
            j = cols[..., c]
            ok = j >= 0
            tok = np.where(ok, np.take_along_axis(xin, np.maximum(j, 0), 1), 0)
            step = np.einsum("btks,btksr->btkr", v, Tm[:, tok].transpose(1, 2, 0, 3, 4))
            nxt = cols[..., c - 1] if c > 0 else then
            if clock == "position" and nxt is not None:
                gap = np.where(ok & (nxt >= 0), nxt - j - 1, 0)
                step = np.einsum("btks,btksr->btkr", step, Apow[:, gap].transpose(1, 2, 0, 3, 4))
            step = step / step.sum(-1, keepdims=True)
            v = np.where(ok[..., None, None], step, v)
        return v

    eta = state(order[..., :m])                             # after the last m own tokens
    before = state(order[..., 1:m + 1], then=order[..., 0])  # what the thread believed before token i

    # 5. evidence carried by each ordinary token, per type
    emit = Tm.sum(-1)                                       # [K, tok, s]
    e_tok = emit[:, np.where(ordinary, xin, 0)].transpose(1, 2, 0, 3)
    evidence = np.where(ordinary[..., None], np.log((before * e_tok).sum(-1)), 0.0)

    # 6. type posterior: sum the thread's evidence
    logw = np.einsum("bij,bjk->bik", own_tok.astype(float), evidence) + np.log(1 / K)
    w = np.exp(logw - logw.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)

    # 7. belief and next-token law
    joint = w[..., None] * eta
    po, pc = proc.struct_probs(depth)
    pred = np.zeros((B, T, VOCAB))
    pred[..., :3] = (1 - po - pc)[..., None] * np.einsum("btks,kxs->btx", joint, emit)
    pred[..., OPEN] = po
    pred[..., CLOSE0:] = pc[..., None] * w
    return dict(read=xin, depth=depth, own=own, prev=prev, index=index, state=eta, evidence=evidence,
                logw=logw, w=w, belief=joint.reshape(B, T, K * 3), pred=pred)


def position_classes(xin):
    """Where each shortcut should show. Per position (after reading its token):
    closed      number of children of the own bracket that have closed
    since       own ordinary tokens read since the last child closed (-1 if none has)
    child_span  positions spanned by that child, '#' to close inclusive
    sibling     an earlier bracket was opened at this depth before the own bracket
    """
    B, T = xin.shape
    out = {k: np.zeros((B, T), dtype=np.int64) for k in ["closed", "since", "child_span", "sibling"]}
    for b in range(B):
        stack, seen = [dict(start=0, closed=0, since=-1, span=0, sib=0)], set()
        for i in range(T):
            x = xin[b, i]
            if x == OPEN:
                d = len(stack)
                stack.append(dict(start=i, closed=0, since=-1, span=0, sib=int(d in seen)))
                seen.add(d)
            elif CLOSE0 <= x < BOS:
                child = stack.pop()
                stack[-1].update(closed=stack[-1]["closed"] + 1, since=0, span=i - child["start"] + 1)
            elif x < 3 and stack[-1]["closed"]:
                stack[-1]["since"] += 1
            top = stack[-1]
            out["closed"][b, i], out["since"][b, i] = top["closed"], top["since"]
            out["child_span"][b, i], out["sibling"][b, i] = top["span"], top["sib"]
    return out


def cells(ref, cls):
    """Position classes used in the loss table."""
    xin, depth, index = ref["read"], ref["depth"], ref["index"]
    is_close = (xin >= CLOSE0) & (xin < BOS)
    after = (xin < 3) & (cls["since"] >= 1) & (cls["since"] <= 3)
    return {
        "all": np.ones_like(xin, dtype=bool),
        "no child yet": cls["closed"] == 0,                 # own bracket has had no child close
        "new, has sibling": (cls["sibling"] == 1) & (cls["closed"] == 0) & (index <= 2),
        "at a close": is_close,                             # has just read a close token
        "after short child": after & (cls["child_span"] <= 5),    # 1-3 own tokens after a child closed
        "after long child": after & (cls["child_span"] >= 12),
    }


def excess(bayes, other):
    """KL(bayes || other) per position, nats."""
    lb = np.log(np.where(bayes > 0, bayes, 1.0))
    lo = other if other.min() < 0 else np.log(np.where(other > 0, other, 1e-300))
    return (bayes * (lb - lo)).sum(-1)


SHORTCUTS = [
    ("the algorithm (exact)", dict()),
    ("state from last 8 own tokens", dict(window=8)),
    ("state from last 4 own tokens", dict(window=4)),
    ("state from last 2 own tokens", dict(window=2)),
    ("recency counted in positions", dict(clock="position")),
    ("depth tag only, no bracket tag", dict(tag="depth")),
]
FORGETFUL = [("keeps type, forgets state at a close", "type_only"), ("forgets the parent at a close", "amnesic")]


def loss_table(proc, toks, extra=()):
    """Excess loss over Bayes, mnat/token, of each shortcut (and of `extra` (name, log-probs) rows)."""
    ref = run(proc, toks)
    cs = cells(ref, position_classes(ref["read"]))
    rows = [(name, run(proc, toks, **kw)["pred"]) for name, kw in SHORTCUTS]
    rows += [(name, proc.filter(toks, mode)[0]) for name, mode in FORGETFUL]
    rows += list(extra)
    names = list(cs)
    print(f"{'excess loss over Bayes, mnat/token':40s}" + "".join(f"{n:>18s}" for n in names))
    print(f"{'share of positions':40s}" + "".join(f"{cs[n].mean():18.3f}" for n in names))
    for name, pred in rows:
        kl = excess(ref["pred"], pred)
        print(f"{name:40s}" + "".join(f"{1e3 * kl[cs[n]].mean():18.1f}" for n in names))
    return ref, cs


if __name__ == "__main__":
    proc = BracketProcess()
    toks, _ = proc.sample(1024, 128, np.random.default_rng(0))
    ref = run(proc, toks)
    exact, beliefs, _ = proc.filter(toks, "bayes")
    print(f"algorithm vs stack filter on {toks.size} positions: max |difference| in next-token law "
          f"{np.abs(ref['pred'] - exact).max():.1e}, in belief {np.abs(ref['belief'] - beliefs).max():.1e}\n")

    text = "#AA#BBB#C}B]A)"
    ex = run(proc, encode(text)[None])
    print("worked example, one column per position (it has read the token shown):")
    print("read           " + "".join(f"{('BOS' if t == BOS else SYMS[t]):>5s}" for t in ex["read"][0]))
    print("position       " + "".join(f"{i:5d}" for i in range(len(text))))
    print("1 depth        " + "".join(f"{d:5d}" for d in ex["depth"][0]))
    print("2 tag (own #)  " + "".join(f"{o:5d}" for o in ex["own"][0]))
    print("3 index        " + "".join(f"{n:5d}" for n in ex["index"][0]))
    for k in range(proc.K):
        print(f"6 P(type {SYMS[CLOSE0 + k]})    " + "".join(f"{v:5.2f}" for v in ex["w"][0, :, k]))
    print("  (the token after the last one shown is not read; the tag at each position names the thread\n"
          "   whose tokens steps 3-7 gather: position 10 has read the '}' and is back on tag 4, i.e. BBB)\n")

    loss_table(proc, toks)
