"""One sequence followed belief by belief: exact, and as read from a trained network.

  python example.py RUN_HIDDEN RUN_OPEN [seed]
  python example.py runs/night_d128_L4_T128_s1 "runs/mess3+open_night_d128_L4_T128_s0"

The two processes differ only in the open token ('#' for every type, or one open token per type), so
both networks are shown the SAME sequence: same brackets, same content. Each column is one position,
after it has read the token above it. Rows, exact first and then the network's version:

  type posterior of the current bracket   exact; read linearly from the last block; and implied by the
                                          network's own close-token probabilities (inside brackets only)
  state belief under each type            exact; read linearly from the last block

Writes figures/example.html and figures/example_hidden.html, example_open.html (for export.sh).
"""
import json, os, sys
import numpy as np
import torch
from process import make
from model import trace, load
import algorithm

run_h, run_o = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 4242
W = 60                                                       # positions shown


def setup(run):
    args = json.load(open(run + ".json"))["args"]
    proc = make(args.get("process", "mess3"), p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
    return args, proc, load(run, args, proc)


(args_h, proc_h, model_h), (args_o, proc_o, model_o) = setup(run_h), setup(run_o)
assert proc_o.reveal and not proc_h.reveal and proc_h.C == proc_o.C and proc_h.K == proc_o.K
K, S, T = proc_h.K, proc_h.S, args_h["T"]

# ---- pick a sequence with several brackets opened inside brackets, and closes that return into brackets
cand, _ = proc_o.sample(6000, T, np.random.default_rng(seed))
st = algorithm.run(proc_o, cand)
xin, depth, index = st["read"], st["depth"], st["index"]
content, opens, closes = proc_o.is_content(xin), proc_o.is_open(xin), proc_o.is_close(xin)
best, pick = -1e9, 0
for b in range(len(cand)):
    nested = returns = differ = 0
    typ_of = {int(p): int(xin[b, p] - proc_o.OPEN0) for p in np.flatnonzero(opens[b, :W])}
    for i in range(2, W - 4):
        if opens[b, i] and depth[b, i - 1] >= 1 and index[b, i - 1] >= 3 and content[b, i + 1:i + 4].all():
            nested += 1
            differ += typ_of[i] != typ_of[int(st["own"][b, i - 1])]
        if closes[b, i] and depth[b, i] >= 1 and index[b, i] >= 3 and content[b, i + 1:i + 3].all():
            returns += 1
    n_struct = int((opens[b, :W] | closes[b, :W]).sum())
    first_open = int(np.argmax(opens[b, :W])) if opens[b, :W].any() else W
    score = 3 * min(nested, 3) + 2 * min(returns, 3) + 2 * min(differ, 3) + 2 * (len(set(typ_of.values())) == K) + (depth[b, :W].max() >= 3) \
        - 0.5 * max(n_struct - 14, 0) - 0.5 * max(first_open - 7, 0)
    if score > best:
        best, pick = score, b
tok_o = cand[pick:pick + 1]
tok_h = np.where(proc_o.is_open(tok_o), proc_h.OPEN0, np.where(proc_o.is_close(tok_o), tok_o - proc_o.CLOSE0 + proc_h.CLOSE0, tok_o))


typed = algorithm.read_tokens(proc_o, tok_o)[0] - proc_o.OPEN0      # at an open position: the bracket's type


def follow(args, proc, model, toks):
    """Exact beliefs and the network's, along one sequence."""
    N = 768
    fit_toks, _ = proc.sample(N, T, np.random.default_rng(2026))
    fit = algorithm.run(proc, fit_toks)
    X = np.concatenate([trace(model, torch.from_numpy(fit["read"][a:a + 128]), want_attn=False)[1][-1].double().numpy()
                        for a in range(0, N, 128)]).reshape(N * T, -1)
    mx = X.mean(0)

    def probe(Y):
        Y = Y.reshape(N * T, -1)
        my = Y.mean(0)
        Wt = np.linalg.solve((X - mx).T @ (X - mx) + 1e-4 * len(X) * np.eye(X.shape[1]), (X - mx).T @ (Y - my))
        return lambda Z: (Z - mx) @ Wt + my
    read_state, read_type = probe(fit["state"]), probe(fit["w"])
    ex = algorithm.run(proc, toks)
    logits, resid, _ = trace(model, torch.from_numpy(ex["read"]), want_attn=False)
    R = resid[-1][0].double().numpy()
    lp = torch.log_softmax(logits[0], -1).double().numpy()
    close_p = np.exp(lp[:, proc.CLOSE0:proc.CLOSE0 + K])
    implied = np.where((ex["depth"][0] >= 1)[:, None], close_p / close_p.sum(-1, keepdims=True), np.nan)
    return dict(read=ex["read"][0], depth=ex["depth"][0], own=ex["own"][0], w=ex["w"][0], state=ex["state"][0], typed=typed,
                w_read=read_type(R).clip(0, 1), w_implied=implied, state_read=read_state(R).reshape(T, K, S).clip(0, 1),
                excess=1e3 * algorithm.excess(ex["pred"][0], lp), proc=proc)


runs = [("hidden", "The open token is the same for every type: the type has to be inferred", follow(args_h, proc_h, model_h, tok_h)),
        ("open", "The open token names the type: the posterior collapses at once", follow(args_o, proc_o, model_o, tok_o))]

# ---- drawing ------------------------------------------------------------------------------------
SURF, PAGE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TYPE_COL = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in
                ["#f3f2ee", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]], dtype=float)
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'
esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def shade(p):
    if np.isnan(p):
        return None
    t = float(np.clip(p, 0, 1)) * (len(SEQ) - 1)
    i = min(int(t), len(SEQ) - 2)
    c = SEQ[i] + (SEQ[i + 1] - SEQ[i]) * (t - i)
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def text(x, y, s, size=11, fill=INK2, anchor="start", weight="400"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" {FONT}>{esc(s)}</text>'


def draw(title, d):
    proc = d["proc"]
    cw, ch, x0 = 15.6, 10, 196
    xs = lambda i: x0 + i * cw
    out, y = [], 78
    is_open, is_close = proc.is_open(d["read"]), proc.is_close(d["read"])
    close_names = proc.syms[proc.CLOSE0:]
    # bracket bars, one lane per depth, coloured by the bracket's type once it is known
    spans, stack = [], []
    for i in range(W):
        if is_open[i]:
            stack.append([i, None, len(stack) + 1, int(d["typed"][i])])
        elif is_close[i]:
            sp = stack.pop()
            sp[1] = i
            spans.append(sp)
    spans += [[o, None, lane, typ] for o, _, lane, typ in stack]
    lanes = max([s[2] for s in spans] + [1])
    out.append(text(24, y + lanes * 9 - 2, "brackets, open to close", 11, INK2))
    for k in range(K):
        out.append(f'<rect x="{24 + k * 50}" y="{y - 20}" width="10" height="6" rx="3" fill="{TYPE_COL[k]}"/>')
        out.append(text(38 + k * 50, y - 13, f"type {close_names[k]}", 10, INK2))
    for o, c, lane, typ in spans:
        yy = y + (lanes - lane) * 9
        x1, x2 = xs(o) + 1, (xs(c) + cw - 2) if c is not None else xs(W)
        out.append(f'<rect x="{x1:.1f}" y="{yy}" width="{x2 - x1:.1f}" height="6" rx="3" fill="{TYPE_COL[typ] if typ is not None else AXIS}">'
                   f'<title>bracket opened at {o}{"" if c is None else f", closed at {c}"}</title></rect>')
    y += lanes * 9 + 14
    for i in range(W):
        ch_ = "^" if d["read"][i] == proc.BOS else proc.syms[d["read"][i]]
        strong = is_open[i] or is_close[i]
        out.append(text(xs(i) + cw / 2 - 0.5, y, ch_, 11, INK if strong else MUTED, "middle", "700" if strong else "400"))
    out.append(text(24, y, "token read", 11, INK2))
    y += 12
    top = y

    def block(label, groups, row_names):
        nonlocal y
        out.append(text(24, y + 9, label, 11, INK, weight="600"))
        y += 14
        for gname, vals in groups:
            out.append(text(34, y + len(row_names) * ch / 2 + 3, gname, 10, INK2))
            for r_, rn in enumerate(row_names):
                out.append(text(x0 - 6, y + r_ * ch + 8, rn, 9, MUTED, "end"))
                for i in range(W):
                    col = shade(vals[i, r_])
                    if col:
                        out.append(f'<rect x="{xs(i):.1f}" y="{y + r_ * ch}" width="{cw - 1:.1f}" height="{ch - 1}" fill="{col}">'
                                   f'<title>{esc(gname)}, {esc(rn)}, position {i}: {vals[i, r_]:.2f}</title></rect>')
            y += len(row_names) * ch + 3
        y += 7

    block("Type of the current bracket", [("exact", d["w"]), ("network, probe", d["w_read"]),
                                          ("network, close odds", d["w_implied"])], [f"type {c}" for c in close_names])
    for k in range(K):
        block(f"State, if the bracket is type {close_names[k]}", [("exact", d["state"][:, k]), ("network, probe", d["state_read"][:, k])],
              [f"state {s + 1}" for s in range(S)])
    for i in range(W):                                      # guides at every open and close
        if is_open[i] or is_close[i]:
            xg = xs(i) - 0.5 if is_open[i] else xs(i) + cw - 0.5
            out.append(f'<line x1="{xg:.1f}" y1="{top - 2}" x2="{xg:.1f}" y2="{y - 6}" stroke="{INK}" stroke-opacity="0.35" stroke-width="1"/>')
    for j in range(len(SEQ)):
        out.append(f'<rect x="{x0 + j * 22}" y="{y + 4}" width="21" height="8" fill="{shade(j / (len(SEQ) - 1))}"/>')
    out += [text(x0, y + 24, "probability 0", 10, MUTED), text(x0 + len(SEQ) * 22, y + 24, "1", 10, MUTED, "end"),
            text(x0 + len(SEQ) * 22 + 14, y + 12, f"network's excess loss over exact Bayes on these {W} positions: {d['excess'][:W].mean():.1f} mnat/token", 10, MUTED)]
    h = y + 36
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="{h}" viewBox="0 0 1140 {h}"><rect width="1140" height="{h}" fill="{SURF}"/>'
            + text(24, 34, title, 17, INK, weight="600")
            + text(24, 54, "Each column is one position, after reading the token above it. A vertical line marks every open (to its left) and close (to its right).", 12, INK2)
            + "".join(out) + "</svg>")


def draw_simplex(title, d, per_row=20):
    """The same sequence with every belief drawn as a point in a triangle, one triangle per position."""
    proc = d["proc"]
    cw, x0, side = 46.0, 150, 34.0
    hgt = side * np.sqrt(3) / 2
    close_names = proc.syms[proc.CLOSE0:]
    is_open, is_close = proc.is_open(d["read"]), proc.is_close(d["read"])
    cur_type = [int(d["typed"][d["own"][i]]) if d["own"][i] >= 1 else None for i in range(W)]
    spans, stack = [], []
    for i in range(W):
        if is_open[i]:
            stack.append([i, W - 1, len(stack) + 1, int(d["typed"][i])])
        elif is_close[i]:
            sp = stack.pop()
            sp[1] = i
            spans.append(sp)
    spans += stack
    lanes = max([sp[2] for sp in spans] + [1])
    rows = [("type", d["w"], d["w_read"])] + [(f"state, if type {close_names[k]}", d["state"][:, k], d["state_read"][:, k]) for k in range(K)]

    def point(p, cx, top):
        p = np.clip(p, 0, None)
        p = p / max(p.sum(), 1e-9)
        return cx - side / 2 + (p[1] + 0.5 * p[2]) * side, top + hgt - p[2] * hgt

    out, y = [], 96
    for r0 in range(0, W, per_row):
        cols = range(r0, min(r0 + per_row, W))
        xs = lambda i: x0 + (i - r0) * cw
        for o, c, lane, typ in spans:                       # the part of each bracket bar that falls in this row
            lo, hi = max(o, r0), min(c, r0 + per_row - 1)
            if lo <= hi:
                x1 = xs(lo) + (4 if lo == o else 0)
                x2 = xs(hi) + cw - (4 if hi == c else 0)
                out.append(f'<rect x="{x1:.1f}" y="{y + (lanes - lane) * 8}" width="{x2 - x1:.1f}" height="5" rx="2.5" fill="{TYPE_COL[typ]}"/>')
        y += lanes * 8 + 14
        for i in cols:
            ch_ = "^" if d["read"][i] == proc.BOS else proc.syms[d["read"][i]]
            strong = is_open[i] or is_close[i]
            out.append(text(xs(i) + cw / 2, y, ch_, 13, INK if strong else MUTED, "middle", "700" if strong else "400"))
        out.append(text(24, y, "token read", 11, INK2))
        y += 8
        top = y
        for label, exact, net in rows:
            out.append(text(24, y + hgt / 2 + 6, label, 11, INK2))
            for i in cols:
                cx = xs(i) + cw / 2
                tri = f"{cx - side / 2:.1f},{y + hgt:.1f} {cx + side / 2:.1f},{y + hgt:.1f} {cx:.1f},{y:.1f}"
                out.append(f'<polygon points="{tri}" fill="none" stroke="{AXIS}" stroke-width="1"/>')
                ex_, ey_ = point(exact[i], cx, y)
                nx_, ny_ = point(net[i], cx, y)
                col = TYPE_COL[cur_type[i]] if cur_type[i] is not None else INK2
                out.append(f'<circle cx="{nx_:.1f}" cy="{ny_:.1f}" r="3" fill="{col}"><title>{esc(label)}, position {i}: network {np.round(net[i], 2)}, exact {np.round(exact[i], 2)}</title></circle>')
                out.append(f'<circle cx="{ex_:.1f}" cy="{ey_:.1f}" r="5" fill="none" stroke="{INK}" stroke-width="1.1"/>')
            y += hgt + 17
        for i in cols:                                      # guides at every open and close
            if is_open[i] or is_close[i]:
                xg = xs(i) + (0.5 if is_open[i] else cw - 0.5)
                out.append(f'<line x1="{xg:.1f}" y1="{top - 2}" x2="{xg:.1f}" y2="{y - 14}" stroke="{INK}" stroke-opacity="0.3" stroke-width="1"/>')
        y += 10
    # key
    kx, ky, ks = 800, 40, 46.0
    kh = ks * np.sqrt(3) / 2
    for j, (names, lab) in enumerate(((["type " + c for c in close_names], "type row"), ([f"state {s_ + 1}" for s_ in range(S)], "state rows"))):
        cx = kx + j * 170
        out.append(f'<polygon points="{cx - ks / 2:.1f},{ky + kh:.1f} {cx + ks / 2:.1f},{ky + kh:.1f} {cx:.1f},{ky:.1f}" fill="none" stroke="{AXIS}"/>')
        out += [text(cx - ks / 2 - 4, ky + kh + 4, names[0], 10, MUTED, "end"), text(cx + ks / 2 + 4, ky + kh + 4, names[1], 10, MUTED),
                text(cx + 6, ky + 4, names[2], 10, MUTED)]
    out.append(f'<circle cx="24" cy="70" r="5" fill="none" stroke="{INK}" stroke-width="1.1"/>' + text(34, 74, "exact", 11, INK2))
    out.append(f'<circle cx="92" cy="70" r="3" fill="{INK2}"/>' + text(100, 74, "network (linear readout); its colour is the current bracket's type:", 11, INK2))
    for k in range(K):
        out.append(f'<circle cx="{456 + k * 62}" cy="70" r="3" fill="{TYPE_COL[k]}"/>' + text(464 + k * 62, 74, f"type {close_names[k]}", 11, INK2))
    out.append(f'<circle cx="{456 + K * 62}" cy="70" r="3" fill="{INK2}"/>' + text(464 + K * 62, 74, "root", 11, INK2))
    h = int(y + 6)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="{h}" viewBox="0 0 1140 {h}"><rect width="1140" height="{h}" fill="{SURF}"/>'
            + text(24, 34, title, 17, INK, weight="600")
            + text(24, 54, "One triangle per position, after reading the token above it. A dot in a ring means the network agrees with exact Bayes.", 12, INK2)
            + "".join(out) + "</svg>")


css = f"""body{{margin:0;background:{PAGE};color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color-scheme:light}}
.wrap{{width:1140px;margin:0 auto;padding:28px 0 40px}} h1{{font-size:22px;margin:0 0 6px}} p{{font-size:14px;line-height:1.5;color:{INK2};margin:6px 0}}
.card{{background:{SURF};border:1px solid rgba(11,11,11,0.10);border-radius:10px;margin:16px 0;overflow:hidden}} svg{{display:block}}
.cap{{font-size:13px;color:{INK2};padding:0 24px 16px;line-height:1.5}}"""
page = lambda inner: (f'<!doctype html><html data-theme="light"><head><meta charset="utf-8"><title>One sequence, belief by belief</title>'
                      f'<style>{css}</style></head><body><div class="wrap">{inner}</div></body></html>')
cap = ("The same sequence is shown to two networks; only the open token differs. Follow any bar: at its left end every belief returns to its "
       "starting value, and just past its right end the columns repeat what the enclosing bracket held just before the bar began. "
       "'Probe' rows are linear readouts of the last block, fitted on other sequences. 'Close odds' needs no probe: it is the network's own "
       "probability for each close token, renormalised, and exists only inside brackets.")
cap_s = ("The same sequence is shown to two networks; only the open token differs. At every open the three state points jump to the centre of "
         "their triangle (the starting belief); the type point jumps to the centre when the open is #, and straight to a corner when the open "
         "names the type. Just past every close, all four points jump back to where they were in the last triangle before that bracket opened. "
         "Bars above the tokens mark each bracket from its open to its close.")
simplex = {name: draw_simplex(title, d) for name, title, d in runs}
figs = {name: draw(title, d) for name, title, d in runs}
os.makedirs("figures", exist_ok=True)
open("figures/example.html", "w").write(page(
    "<h1>One sequence, belief by belief</h1><p>" + esc(cap) + "</p>" + "".join(f'<div class="card">{f}</div>' for f in figs.values())))
for name, f in figs.items():
    open(f"figures/example_{name}.html", "w").write(page(f'<div class="card" style="margin:0">{f}<div class="cap">{esc(cap)}</div></div>'))
open("figures/example_simplex.html", "w").write(page(
    "<h1>One sequence, belief by belief, on the simplex</h1><p>" + esc(cap_s) + "</p>" + "".join(f'<div class="card">{f}</div>' for f in simplex.values())))
for name, f in simplex.items():
    open(f"figures/example_simplex_{name}.html", "w").write(page(f'<div class="card" style="margin:0">{f}<div class="cap">{esc(cap_s)}</div></div>'))
print("sequence (hidden opens):", proc_h.show(tok_h[0, :W - 1]))
print("sequence (typed opens): ", proc_o.show(tok_o[0, :W - 1]))
for name, _, d in runs:
    print(f"{name}: excess loss on the window {d['excess'][:W].mean():.1f} mnat/token; "
          f"type readout error (mean abs) {np.abs(d['w_read'][:W] - d['w'][:W]).mean():.3f}, state readout error {np.abs(d['state_read'][:W] - d['state'][:W]).mean():.3f}")
