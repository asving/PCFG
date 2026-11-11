"""The proposed algorithm: exact next-token prediction with no stack and no recurrence.

Position i has read tokens 0..i-1 (position 0 has read only BOS) and predicts token i. Every quantity
below is computed from the tokens with three operations that attention can perform:

  count        a running sum over earlier positions                 (even attention)
  most recent  the latest earlier position matching a key           (content match + recency)
  gather       a sum over the earlier positions that share a tag    (content match)

Steps, and what each leaves at position i:

  1 depth      opens minus closes among the tokens read
  2 tag        position of the open token that opened i's bracket: the most recent open at depth[i]
               (BOS plays the open of the root). Positions with equal tags form one bracket's thread.
  3 index      how many content tokens of its own bracket i has read
  4 state      for each type k, component k's state belief from the thread's content tokens in order
  5 evidence   at each content token, log P(that token | its bracket so far, type k)
  6 type       log-posterior of the type = sum of the thread's evidence (no decay)
  7 belief     type posterior x state belief, and from it the next-token law

Nothing is done at a close: the depth drops, so step 2 returns the parent's tag and steps 3-7 pick up
the parent's tokens again.

run() also implements the shortcuts a network might take instead, to predict what each would cost:

  window=m           the state belief uses only the last m own tokens
  clock="position"   recency counted in positions, not thread tokens: the state belief relaxes across
                     the positions that separate two consecutive own tokens (a child in between)
  tag="depth"        no bracket tag: all earlier positions at the same depth are treated as one thread
  tag="open"         no reset at an open: the thread also includes every enclosing bracket's tokens
  infer_type=False   no type inference: the type posterior stays at its prior

  python algorithm.py [process]     checks run() against the stack filter, prints a worked example,
                                    and the excess loss of each shortcut by position class
"""
import sys
import numpy as np
from process import make

STEPS = ["depth", "tag", "index", "state", "evidence", "type", "belief"]


def read_tokens(proc, toks):
    """Token read at each position: BOS, then tokens 0..T-2."""
    return np.concatenate([np.full((len(toks), 1), proc.BOS), toks[:, :-1]], 1)


def next_token_law(proc, depth, w, eta):
    """Step 7. depth [...], w [..., K], eta [..., K, S] -> probabilities [..., VOCAB]."""
    po, pc = proc.struct_probs(depth)
    pred = np.zeros(depth.shape + (proc.VOCAB,))
    pred[..., :proc.C] = (1 - po - pc)[..., None] * np.einsum("...ks,kxs->...x", w[..., None] * eta, proc.emit)
    pred[..., proc.OPEN0:proc.CLOSE0] = (po / proc.n_open)[..., None]
    pred[..., proc.CLOSE0:] = pc[..., None] * w
    return pred


def run(proc, toks, tag="bracket", clock="thread", window=None, infer_type=True, chunk=256):
    outs = [_run(proc, toks[a:a + chunk], tag, clock, window, infer_type) for a in range(0, len(toks), chunk)]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _run(proc, toks, tag, clock, window, infer_type):
    B, T = toks.shape
    K, S = proc.K, proc.S
    xin = read_tokens(proc, toks)
    J = np.arange(T)
    visible = J[None, :] <= J[:, None]                      # [i, j]: j is at or before i
    is_open, is_close, content = proc.is_open(xin), proc.is_close(xin), proc.is_content(xin)

    # 1. depth: a running count
    depth = np.cumsum(is_open.astype(np.int64) - is_close, 1)
    same_depth = depth[:, :, None] == depth[:, None, :]     # [b, i, j]

    # 2. tag: the most recent open at my depth; equal tags = same bracket
    openlike = is_open | (J == 0)[None]
    own = np.where(openlike[:, None, :] & same_depth & visible, J, -1).max(-1)
    same_tag = own[:, :, None] == own[:, None, :]
    if tag == "bracket":
        member = same_tag & visible
    elif tag == "depth":
        member = same_depth & visible
    else:                                                   # "open": own bracket and all enclosing ones
        member, up = same_tag.copy(), own.copy()
        for _ in range(proc.D):                             # the parent's tag is the tag at own - 1
            up = np.where(up >= 1, np.take_along_axis(own, np.maximum(up - 1, 0), 1), -1)
            member |= own[:, None, :] == up[:, :, None]
        member &= visible
    prev = np.where(same_tag & (J[None, :] < J[:, None]), J, -1).max(-1)    # previous position in thread

    # 3. index: a count over the thread
    own_tok = member & content[:, None, :]
    index = own_tok.sum(-1)

    # 4. state belief per type, from the thread's content tokens, most recent first in `order`
    kmax = max(int(index.max()), 1)
    order = -np.sort(-np.where(own_tok, J, -1), axis=-1)[..., :kmax + 1]
    if order.shape[-1] < kmax + 1:
        order = np.concatenate([order, np.full((B, T, kmax + 1 - order.shape[-1]), -1)], -1)
    m = kmax if window is None else min(window, kmax)
    Apow = np.stack([np.stack([np.linalg.matrix_power(proc.A[k], g) for g in range(T + 1)]) for k in range(K)])

    def state(cols, then=None):
        """Belief after the tokens at positions cols[..., c], oldest (last column) to newest."""
        v = np.broadcast_to(proc.eta0, (B, T, K, S)).copy()
        for c in range(cols.shape[-1] - 1, -1, -1):
            j = cols[..., c]
            ok = j >= 0
            tok = np.where(ok, np.take_along_axis(xin, np.maximum(j, 0), 1), 0)
            step = np.einsum("btks,btksr->btkr", v, proc.T[:, tok].transpose(1, 2, 0, 3, 4))
            nxt = cols[..., c - 1] if c > 0 else then
            if clock == "position" and nxt is not None:
                gap = np.where(ok & (nxt >= 0), nxt - j - 1, 0)
                step = np.einsum("btks,btksr->btkr", step, Apow[:, gap].transpose(1, 2, 0, 3, 4))
            mass = (step * proc.r).sum(-1, keepdims=True)
            # a token that is impossible under type k leaves no belief for k; step 5 then rules k out
            step = np.where(mass > 1e-12, step / np.where(mass > 1e-12, mass, 1.0), proc.eta0)
            v = np.where(ok[..., None, None], step, v)
        return v

    eta = state(order[..., :m])                             # after the last m own tokens
    before = state(order[..., 1:m + 1], then=order[..., 0])  # what the thread believed before token i

    # 5. evidence carried by each content token, per type
    e_tok = proc.emit[:, np.where(content, xin, 0)].transpose(1, 2, 0, 3)       # [B, T, K, S]
    evidence = np.where(content[..., None], np.log(np.maximum((before * e_tok).sum(-1), 1e-300)), 0.0)

    # 6. type posterior: sum the thread's evidence onto the prior (known type if the open revealed it)
    base = np.full((B, T, K), np.log(1 / K))
    if proc.reveal:
        shown = np.take_along_axis(xin, own, 1) - proc.OPEN0                     # type named by the own open
        known = own >= 1
        base = np.where(known[..., None], np.where(np.arange(K) == shown[..., None], 0.0, -1e9), base)
    logw = base + (np.einsum("bij,bjk->bik", own_tok.astype(float), evidence) if infer_type else 0.0)
    w = np.exp(logw - logw.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)

    # 7. belief and next-token law
    pred = next_token_law(proc, depth, w, eta)
    return dict(read=xin, depth=depth, own=own, prev=prev, index=index, state=eta, evidence=evidence,
                logw=logw, w=w, belief=(w[..., None] * eta).reshape(B, T, K * S), pred=pred)


def position_classes(proc, xin):
    """Where each shortcut should show. Per position (after reading its token):
    closed      number of children of the own bracket that have closed
    since       own content tokens read since the last child closed (-1 if none has)
    child_span  positions spanned by that child, open to close inclusive
    sibling     an earlier bracket was opened at this depth before the own bracket
    """
    B, T = xin.shape
    out = {k: np.zeros((B, T), dtype=np.int64) for k in ["closed", "since", "child_span", "sibling"]}
    opens, closes = proc.is_open(xin), proc.is_close(xin)
    for b in range(B):
        stack, seen = [dict(start=0, closed=0, since=-1, span=0, sib=0)], set()
        for i in range(T):
            if opens[b, i]:
                d = len(stack)
                stack.append(dict(start=i, closed=0, since=-1, span=0, sib=int(d in seen)))
                seen.add(d)
            elif closes[b, i]:
                child = stack.pop()
                stack[-1].update(closed=stack[-1]["closed"] + 1, since=0, span=i - child["start"] + 1)
            elif xin[b, i] < proc.C and stack[-1]["closed"]:
                stack[-1]["since"] += 1
            top = stack[-1]
            out["closed"][b, i], out["since"][b, i] = top["closed"], top["since"]
            out["child_span"][b, i], out["sibling"][b, i] = top["span"], top["sib"]
    return out


def cells(proc, ref, cls):
    """Position classes used in the loss table."""
    xin, index = ref["read"], ref["index"]
    after = proc.is_content(xin) & (cls["since"] >= 1) & (cls["since"] <= 3)
    return {
        "all": np.ones_like(xin, dtype=bool),
        "no child yet": cls["closed"] == 0,                 # own bracket has had no child close
        "new, has sibling": (cls["sibling"] == 1) & (cls["closed"] == 0) & (index <= 2),
        "new, depth >= 1": (ref["depth"] >= 1) & (cls["closed"] == 0) & (index <= 2),
        "at a close": proc.is_close(xin),                   # has just read a close token
        "after short child": after & (cls["child_span"] <= 5),    # 1-3 own tokens after a child closed
        "after long child": after & (cls["child_span"] >= 12),
    }


def excess(bayes, other):
    """KL(bayes || other) per position, nats. `other` may be probabilities or log-probabilities."""
    lb = np.log(np.where(bayes > 0, bayes, 1.0))
    lo = other if other.min() < 0 else np.log(np.where(other > 0, other, 1e-300))
    return (bayes * (lb - lo)).sum(-1)


SHORTCUTS = [
    ("the algorithm (exact)", dict()),
    ("state from last 4 own tokens", dict(window=4)),
    ("state from last 2 own tokens", dict(window=2)),
    ("state from last own token", dict(window=1)),
    ("recency counted in positions", dict(clock="position")),
    ("depth tag only, no bracket tag", dict(tag="depth")),
    ("no reset at an open", dict(tag="open")),
    ("no type inference", dict(infer_type=False)),
]
FORGETFUL = [("keeps type, forgets state at a close", "type_only"), ("forgets the parent at a close", "amnesic")]


def shortcut_losses(proc, toks):
    """{shortcut name: excess loss per position}, plus the exact reference and the position classes."""
    ref = run(proc, toks)
    cs = cells(proc, ref, position_classes(proc, ref["read"]))
    out = {name: excess(ref["pred"], run(proc, toks, **kw)["pred"]) for name, kw in SHORTCUTS}
    out.update({name: excess(ref["pred"], proc.filter(toks, mode)[0]) for name, mode in FORGETFUL})
    return out, ref, cs


def loss_table(proc, toks, extra=()):
    """Print excess loss over Bayes, mnat/token, of each shortcut and of `extra` (name, log-probs) rows."""
    losses, ref, cs = shortcut_losses(proc, toks)
    losses.update({name: excess(ref["pred"], lp) for name, lp in extra})
    print(f"{'excess loss over Bayes, mnat/token':40s}" + "".join(f"{n:>19s}" for n in cs))
    print(f"{'share of positions':40s}" + "".join(f"{m.mean():19.3f}" for m in cs.values()))
    for name, kl in losses.items():
        print(f"{name:40s}" + "".join(f"{1e3 * kl[m].mean():19.1f}" for m in cs.values()))
    return ref, cs


if __name__ == "__main__":
    proc = make(sys.argv[1] if len(sys.argv) > 1 else "mess3")
    toks, _ = proc.sample(1024, 128, np.random.default_rng(0))
    ref = run(proc, toks)
    exact, beliefs, _ = proc.filter(toks, "bayes")
    print(f"{proc.name}: algorithm vs stack filter on {toks.size} positions: max |difference| in next-token law "
          f"{np.abs(ref['pred'] - exact).max():.1e}, in belief {np.abs(ref['belief'] - beliefs).max():.1e}\n")

    o, cl = proc.syms[proc.OPEN0:proc.CLOSE0], proc.syms[proc.CLOSE0:]
    third = proc.syms[min(2, proc.C - 1)]
    text = f"{o[0]}AA{o[-1]}BBB{o[0]}{third}{cl[0]}B{cl[-1]}A{cl[0]}"
    ex = run(proc, proc.encode(text)[None])
    print("worked example, one column per position (it has read the token shown; ^ is BOS):")
    print("read           " + "".join(f"{c:>5s}" for c in proc.show(ex["read"][0])))
    print("position       " + "".join(f"{i:5d}" for i in range(len(text))))
    print("1 depth        " + "".join(f"{d:5d}" for d in ex["depth"][0]))
    print("2 tag (own #)  " + "".join(f"{t:5d}" for t in ex["own"][0]))
    print("3 index        " + "".join(f"{n:5d}" for n in ex["index"][0]))
    for k in range(proc.K):
        print(f"6 P(type {cl[k]})    " + "".join(f"{v:5.2f}" for v in ex["w"][0, :, k]))
    print()
    loss_table(proc, toks)
