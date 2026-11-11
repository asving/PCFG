"""Check: when frames are coupled, the active frame's marginal belief is still a closed, local statistic.

Variant: the child's type depends on the parent's type through PI[k_parent, k_child], and there is a
single close token, so a child's type is never revealed. The joint posterior over the open frames is
then a Markov chain along the stack, not a product. Claim: the marginal belief b of the active frame
still (i) gives the next-token law, and (ii) updates with one read of another position:
    scan    b <- normalize(b T^x)
    call    child prior = PI^T w_parent ; child b = prior x uniform state        (reads the parent at the call)
    return  parent b <- normalize(parent b at the call * sum_kc PI[k, kc] * w_child[kc] / prior_child[kc])
The check compares this recursion with brute-force enumeration over all type assignments.
"""
import itertools
import numpy as np
from process import mess3

rng = np.random.default_rng(0)
K = 3
T = np.stack([mess3(x, a) for x, a in ((0.05, 0.85), (0.15, 0.6), (0.45, 0.85))])   # [K, tok, s, s']
PI = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
ROOT = np.full(K, 1 / K)


def random_string(n):
    out, depth = [], 0
    for _ in range(n):
        u = rng.random()
        if u < 0.22:
            out.append("#"); depth += 1
        elif u < 0.44 and depth > 0:
            out.append(")"); depth -= 1
        else:
            out.append("ABC"[rng.integers(3)])
    return "".join(out)


def local_filter(s):
    """Frame-local recursion. Returns P(next content token | s) up to the structural constant."""
    stack = [dict(b=np.outer(ROOT, np.full(3, 1 / 3)), prior=ROOT)]
    for ch in s:
        top = stack[-1]
        if ch == "#":
            prior = PI.T @ top["b"].sum(1)
            stack.append(dict(b=np.outer(prior, np.full(3, 1 / 3)), prior=prior))
        elif ch == ")":
            child = stack.pop()
            ratio = child["b"].sum(1) / child["prior"]
            b = stack[-1]["b"] * (PI @ ratio)[:, None]
            stack[-1]["b"] = b / b.sum()
        else:
            b = np.einsum("ks,kst->kt", top["b"], T[:, "ABC".index(ch)])
            top["b"] = b / b.sum()
    b = stack[-1]["b"]
    return np.array([np.einsum("ks,kst->", b, T[:, x]) for x in range(3)])


def brute(s):
    """Enumerate the type of every frame; frames are HMM runs given their types."""
    parent, own, stack = [-1], [[]], [0]
    for ch in s:
        if ch == "#":
            parent.append(stack[-1]); own.append([]); stack.append(len(parent) - 1)
        elif ch == ")":
            stack.pop()
        else:
            own[stack[-1]].append("ABC".index(ch))
    active, F = stack[-1], len(parent)
    fwd = np.zeros((F, K, 3))                       # forward vector of each frame's own tokens, per type
    for f in range(F):
        for k in range(K):
            v = np.full(3, 1 / 3)
            for x in own[f]:
                v = v @ T[k, x]
            fwd[f, k] = v
    num, den = np.zeros(3), 0.0
    for ks in itertools.product(range(K), repeat=F):
        w = 1.0
        for f in range(F):
            w *= (ROOT[ks[f]] if parent[f] < 0 else PI[ks[parent[f]], ks[f]])
            if f != active:
                w *= fwd[f, ks[f]].sum()
        v = fwd[active, ks[active]]
        den += w * v.sum()
        num += w * np.array([(v @ T[ks[active], x]).sum() for x in range(3)])
    return num / den


worst, n = 0.0, 0
for _ in range(300):
    s = random_string(int(rng.integers(6, 22)))
    if s.count("#") > 9:
        continue
    worst = max(worst, np.abs(local_filter(s) - brute(s)).max())
    n += 1
print(f"{n} random nested strings, coupled types, hidden close: max |local - brute force| = {worst:.2e}")
s = "AAB#CC#B)A)AA#BCB"
print(s, "local", np.round(local_filter(s), 4), "brute", np.round(brute(s), 4))
