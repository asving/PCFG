"""Figures from a trained run, for sharing: what the network holds at each position, and how.

  python figures.py runs/pilot_d128_L4_T128_s0

Writes figures/report.html (self-contained) and one small HTML page per figure in figures/, which
export.sh turns into PNGs with headless Chrome. No plotting library is needed: charts are SVG written
by hand, and the dense scatter panels are PNGs written with zlib.

  Figure 1  three state beliefs at once, one plane each           (question 0)
  Figure 2  the belief is refreshed at an open and returns at a close   (question 1)
  Figure 3  positions are tagged by depth and by bracket, and attention uses the tags   (question 2)
"""
import base64, json, os, struct, sys, zlib
import numpy as np
import torch
from process import make
from model import GPT, trace, load
import algorithm

run = sys.argv[1]
args = json.load(open(run + ".json"))["args"]
proc = make(args.get("process", "mess3"), p_open=args["p_open"], p_close=args["p_close"], max_depth=args["max_depth"])
N, T, L, H = 768, args["T"], args["layers"], args["heads"]
K, S = proc.K, proc.S
toks, _ = proc.sample(N, T, np.random.default_rng(2026))
ref = algorithm.run(proc, toks)
xin, depth, own, prev, index = ref["read"], ref["depth"], ref["own"], ref["prev"], ref["index"]
content, is_open, is_close = proc.is_content(xin), proc.is_open(xin), proc.is_close(xin)
x = torch.from_numpy(xin)
model = load(run, args, proc)
torch.manual_seed(123)
blank = GPT(proc.VOCAB, T, args["d"], L, H).eval()
J = np.arange(T)
NTR = int(0.75 * N)                                         # probes are fit on the first 75% of sequences
test = np.zeros((N, T), dtype=bool)
test[NTR:] = True
TYPES = [f"type {proc.syms[proc.CLOSE0 + k]}" for k in range(K)]
KIND = ["sticky", "noisy", "anti-persistent"] if args.get("process", "mess3") == "mess3" else [""] * K

# ---------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------
resid, blank_resid, logp = [[] for _ in range(L + 1)], [[] for _ in range(L + 1)], []
CATS = ["own tokens", "own # / closes / self", "inside closed children", "enclosing brackets", "earlier brackets"]
hard = is_close & (depth >= 1) & (index >= 3)               # just read a close, own tokens behind the child


def categories(sl):
    o = own[sl]
    n = len(o)
    same = o[:, :, None] == o[:, None, :]
    anc = np.zeros((n, T, T), dtype=bool)
    tag = o.copy()
    for _ in range(proc.D):
        tag = np.where(tag >= 1, np.take_along_axis(o, np.maximum(tag - 1, 0), 1), -1)
        anc |= o[:, None, :] == tag[:, :, None]
    c = np.full((n, T, T), 4, dtype=np.int8)                # earlier brackets (and BOS)
    c[anc] = 3
    c[(J[None, None, :] > o[:, :, None]) & ~same] = 2
    c[same] = 1
    c[same & content[sl][:, None, :]] = 0
    c[np.broadcast_to(np.eye(T, dtype=bool), c.shape)] = 1
    c[np.broadcast_to(J[None, :] > J[:, None], c.shape)] = -1
    return c


share = np.zeros((L, H, len(CATS)))
even = np.zeros(len(CATS))
attn_keep = {}
for a in range(0, N, 128):
    sl = slice(a, a + 128)
    logits, rs, attn = trace(model, x[sl])
    logp.append(torch.log_softmax(logits, -1).double().numpy())
    for l in range(L + 1):
        resid[l].append(rs[l].double().numpy())
        blank_resid[l].append(trace(blank, x[sl], want_attn=False)[1][l].double().numpy())
    cat, s = categories(sl), hard[sl]
    even += np.stack([((cat == c) / np.maximum((cat >= 0).sum(-1, keepdims=True), 1))[s].sum() for c in range(len(CATS))])
    for l in range(L):
        at = attn[l].numpy()
        for c in range(len(CATS)):
            share[l, :, c] += np.stack([(at[:, h] * (cat == c))[s].sum() for h in range(H)])
    if a + 128 >= N:
        attn_keep = {l: attn[l].numpy() for l in range(L)}   # last chunk (held-out sequences), for the example
        keep_base = a
resid = [np.concatenate(r) for r in resid]
blank_resid = [np.concatenate(r) for r in blank_resid]
logp = np.concatenate(logp)
share /= hard.sum()
even /= hard.sum()


def ridge(X, Y, fit, lam=1e-4):
    Y = Y.reshape(N, T, -1).astype(float)
    Xf, Yf = X[fit], Y[fit]
    mx, my = Xf.mean(0), Yf.mean(0)
    W = np.linalg.solve((Xf - mx).T @ (Xf - mx) + lam * len(Xf) * np.eye(X.shape[-1]), (Xf - mx).T @ (Yf - my))
    return lambda Z: (Z - mx) @ W + my


def r2(X, Y, sel):
    Y = Y.reshape(N, T, -1).astype(float)
    f = ridge(X, Y, sel & ~test)
    Yt = Y[sel & test]
    return float(1 - ((Yt - f(X[sel & test])) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum())


everywhere = np.ones((N, T), dtype=bool)
busy = index >= 2
eta, w = ref["state"], ref["w"]

# question 0
state_read = {l: ridge(resid[l], eta, (index >= 1) & ~test)(resid[l]).reshape(N, T, K, S) for l in (L - 1, L)}


def type_planes(X, mask):
    Z = np.concatenate([eta[mask].reshape(-1, K * S), w[mask]], 1)
    B = np.linalg.lstsq(Z - Z.mean(0), X[mask] - X[mask].mean(0), rcond=None)[0]
    return [np.linalg.svd(B[k * S:(k + 1) * S] - B[k * S:(k + 1) * S].mean(0), full_matrices=False)[2][:S - 1]
            for k in range(K)]


overlap = lambda P, Q: float(np.linalg.norm(P @ Q.T) ** 2 / len(P))
pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
planes = {"untrained": type_planes(blank_resid[L], busy), f"after block {L - 1}": type_planes(resid[L - 1], busy),
          f"after block {L}": type_planes(resid[L], busy)}
plane_overlap = {name: [overlap(P[j], P[k]) for j, k in pairs] for name, P in planes.items()}
likely = [np.mean([r2(resid[l], eta[:, :, k], busy & (w[..., k] > 0.5)) for k in range(K)]) for l in range(L + 1)]
unlikely = [np.mean([r2(resid[l], eta[:, :, k], busy & (w[..., k] < 0.1)) for k in range(K)]) for l in range(L + 1)]

# question 1
w_read = ridge(resid[L], w, ~test)(resid[L]).reshape(N, T, K)
close_p = np.exp(logp[..., proc.CLOSE0:proc.CLOSE0 + K])
w_implied = np.where((depth >= 1)[..., None], close_p / close_p.sum(-1, keepdims=True), np.nan)   # defined inside brackets


def aligned(events, lo, hi):
    """Mean posterior of a reference type at positions lo..hi around each event: exact, linear readout,
    and implied by the network's own close-token probabilities (NaN where the position is in the root)."""
    cols = [np.mean([v[b, i + lo:i + hi + 1, k] for b, i, k in events], 0) for v in (w, w_read, w_implied)]
    return cols + [len(events)]


opens_root, opens_in, closes = [], [], []
for b in range(NTR, N):
    for i in range(5, T - 6):
        if is_open[b, i] and content[b, i - 4:i].all() and content[b, i + 1:i + 6].all() and index[b, i - 1] >= 4:
            (opens_root if depth[b, i - 1] == 0 else opens_in).append((b, i, int(w[b, i - 1].argmax())))
        if is_close[b, i] and depth[b, i] >= 1 and content[b, i - 3:i].all() and content[b, i + 1:i + 4].all() and index[b, i] >= 4:
            closes.append((b, i, int(w[b, i].argmax())))
open_root, open_in, close_ev = aligned(opens_root, -4, 5), aligned(opens_in, -4, 5), aligned(closes, -3, 3)

rng = np.random.default_rng(11)
first = np.zeros((N, T), dtype=bool)
after_it = np.zeros((N, T), dtype=bool)
for b in range(N):
    for o in np.flatnonzero(is_open[b] & (depth[b] == 1)):
        later = np.flatnonzero(is_close[b] & (depth[b] == 0) & (J > o))
        if len(later) and later[0] <= T - 10 and content[b, o + 1:later[0]].sum() >= 2:
            first[b, o + 1:later[0]] = True
            after_it[b, later[0]:] = True
            break
invariance = []
for name, change, still, moved, where_still, where_moved in [
        ("the root bracket's tokens", content & (depth == 0), depth >= 1, (depth == 0) & (index >= 1),
         "inside brackets", "in the root"),
        ("all tokens at depth 0 and 1", content & (depth <= 1), depth >= 2, (depth <= 1) & (index >= 1),
         "at depth 2 and deeper", "at depth 0 and 1"),
        ("the content of a closed bracket", content & first, after_it, first & (np.cumsum(content & first, 1) >= 1),
         "after it closed", "inside it")]:
    x2 = np.where(change, (xin + rng.integers(1, proc.C, xin.shape)) % proc.C, xin)
    lp2 = np.concatenate([torch.log_softmax(trace(model, torch.from_numpy(x2[a:a + 128]), want_attn=False)[0], -1)
                          .double().numpy() for a in range(0, N, 128)])
    net = (np.exp(logp) * (logp - lp2)).sum(-1)
    exact = algorithm.excess(ref["pred"], algorithm.run(proc, np.concatenate([x2[:, 1:], toks[:, -1:]], 1))["pred"])
    invariance.append(dict(name=name, where_still=where_still, where_moved=where_moved,
                           net_still=1e3 * net[still].mean(), net_moved=1e3 * net[moved].mean(),
                           exact_moved=1e3 * exact[moved].mean()))

# question 2
tags = [("depth", depth, everywhere), ("distance back to own #", J[None] - own, depth >= 1),
        ("own tokens read so far", index, everywhere)]
tag_r2 = {n: [r2(resid[l], y, s) for l in range(L + 1)] for n, y, s in tags}
tag_blank = {n: max(r2(blank_resid[l], y, s) for l in range(L + 1)) for n, y, s in tags}

# an example sequence from the held-out chunk: brackets early, nested, several closes, in 56 positions
W = 56
best, ex_b = -1e9, None
for b in range(keep_base, N):
    first_open = int(np.argmax(is_open[b, :W])) if is_open[b, :W].any() else W
    score = 3 * (depth[b, :W].max() >= 2) + min(int(is_close[b, :W].sum()), 5) - abs(int(is_open[b, :W].sum()) - 6) \
        - abs(first_open - 5) + 2 * (index[b, :W][is_close[b, :W]] >= 3).sum().clip(0, 2)
    if score > best:
        best, ex_b = score, b
reader = int(np.argmax(share[L - 1, :, 0]))                  # the last-block head that reads own tokens most
ex_attn = attn_keep[L - 1][ex_b - keep_base, reader, :W, :W]
ex_own = (own[ex_b, :W, None] == own[ex_b, None, :W]) & content[ex_b, None, :W] & (J[None, :W] <= J[:W, None])

# ---------------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------------
SURF, PAGE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#b0afa8"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'
esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=INK2, anchor="start", weight="400", rotate=None):
    tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {FONT}{tr}>{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke=GRID, width=1):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def polyline(pts, stroke, width=2):
    return (f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')


def dot(x, y, fill, r=4, tip=""):
    t = f"<title>{esc(tip)}</title>" if tip else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{SURF}" stroke-width="2">{t}</circle>'


def bar(x, y0, w_, h, fill, tip=""):
    """Column from baseline y0 upwards, rounded at the data end only."""
    r = min(4, w_ / 2, max(h, 0.1))
    t = f"<title>{esc(tip)}</title>" if tip else ""
    return (f'<path d="M{x:.1f},{y0:.1f} v{-(h - r):.1f} a{r},{r} 0 0 1 {r},{-r} h{w_ - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 {r},{r} v{h - r:.1f} z" fill="{fill}">{t}</path>')


def yaxis(x0, y0, h, ticks, lo, hi, w_, fmt="{:g}"):
    out = []
    for t in ticks:
        y = y0 - (t - lo) / (hi - lo) * h
        out += [line(x0, y, x0 + w_, y, GRID if t != lo else AXIS), text(x0 - 6, y + 4, fmt.format(t), 11, MUTED, "end")]
    return "".join(out)


def legend(x, y, items, gap=18):
    out = []
    for i, (label, color, kind) in enumerate(items):
        yy = y + i * gap
        out.append(line(x, yy - 4, x + 16, yy - 4, color, 2) if kind == "line" else
                   f'<rect x="{x}" y="{yy - 9}" width="10" height="10" rx="2" fill="{color}"/>' if kind == "box" else
                   dot(x + 8, yy - 4, color))
        out.append(text(x + (22 if kind != "box" else 16), yy, label, 12, INK2))
    return "".join(out)


def png(img):
    """img [h, w, 3] uint8 -> data URI."""
    h, w_ = img.shape[:2]
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w_, h, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


hexrgb = lambda h_: np.array([int(h_[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def simplex_png(P, colors, wpx, m=0.3, alpha=0.55):
    """Scatter of beliefs P [n, 3] in the triangle, each point tinted by `colors` [n, 3].
    The frame extends a margin m (in units of the triangle's side) beyond it. Returns the image, its
    height in pixels, and the triangle's corners in pixels."""
    scale = wpx / (1 + 2 * m)
    hpx = int(round((np.sqrt(3) / 2 + 1.5 * m) * scale))
    to_px = lambda xy: ((xy[..., 0] + m) * scale, hpx - (xy[..., 1] + 0.75 * m) * scale)
    u, v = to_px(np.stack([P[:, 1] + 0.5 * P[:, 2], P[:, 2] * np.sqrt(3) / 2], 1))
    acc, cnt = np.zeros((hpx, wpx, 3)), np.zeros((hpx, wpx))
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            if abs(du) + abs(dv) == 2:
                continue
            iu, iv = np.round(u + du).astype(int), np.round(v + dv).astype(int)
            ok = (iu >= 0) & (iu < wpx) & (iv >= 0) & (iv < hpx)
            np.add.at(acc, (iv[ok], iu[ok]), colors[ok])
            np.add.at(cnt, (iv[ok], iu[ok]), 1)
    cover = 1 - (1 - alpha) ** cnt
    img = hexrgb(SURF) * (1 - cover[..., None]) + acc / np.maximum(cnt, 1)[..., None] * cover[..., None]
    corners = [to_px(np.array(c)) for c in ((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2))]
    return png(img.clip(0, 255).astype(np.uint8)), hpx, corners


def figure(width, height, body, title, subtitle):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{SURF}"/>'
            + text(24, 34, title, 18, INK, weight="600") + text(24, 56, subtitle, 13, INK2) + body + "</svg>")


STAGES = ["emb"] + [f"blk {l}" for l in range(1, L + 1)]


def linechart(x0, y0, w_, h, series, ticks=(0, 0.25, 0.5, 0.75, 1), ylabel="R²", refs=()):
    """series: [(label, color, values)], x = STAGES. Direct labels at the right end."""
    out = [yaxis(x0, y0, h, ticks, ticks[0], ticks[-1], w_)]
    xs = [x0 + 18 + i * (w_ - 36) / (len(STAGES) - 1) for i in range(len(STAGES))]
    ypos = lambda v: y0 - (v - ticks[0]) / (ticks[-1] - ticks[0]) * h
    out += [text(xx, y0 + 18, s, 11, MUTED, "middle") for xx, s in zip(xs, STAGES)]
    out.append(text(x0 - 34, y0 - h / 2, ylabel, 11, MUTED, "middle", rotate=-90))
    for label, val in refs:
        out += [line(x0, ypos(val), x0 + w_, ypos(val), AXIS), text(x0 + w_ - 4, ypos(val) - 5, label, 11, MUTED, "end")]
    for label, color, vals in series:
        out.append(polyline([(xx, ypos(v)) for xx, v in zip(xs, vals)], color))
        out += [dot(xx, ypos(v), color, tip=f"{label}, {s}: {v:.2f}") for xx, v, s in zip(xs, vals, STAGES)]
    return "".join(out), xs, ypos


# ---- Figure 1 -------------------------------------------------------------------------------------
anchors = np.stack([hexrgb(C1), hexrgb(C2), hexrgb(C3)])
sel = test & (index >= 1)
body, px, gx, gy = [], 232, 24, 100
rows = [("exact", None)] + [(f"read from the residual after block {l}", l) for l in (L - 1, L)]
for k in range(K):
    xx = gx + 112 + k * (px + 4)
    body.append(text(xx + px / 2, gy - 6, f"{TYPES[k]}" + (f", {KIND[k]}" if KIND[k] else ""), 12, INK, "middle", "600"))
    cols = eta[sel][:, k] @ anchors
    for r_, (label, l) in enumerate(rows):
        P = eta[sel][:, k] if l is None else state_read[l][sel][:, k]
        uri, hpx, corners = simplex_png(P, cols, px * 2)
        pitch = hpx / 2 + 2
        yy = gy + r_ * pitch
        body.append(f'<image x="{xx}" y="{yy:.1f}" width="{px}" height="{hpx / 2:.1f}" xlink:href="{uri}" href="{uri}"/>')
        body.append(f'<polygon points="{" ".join(f"{xx + a_ / 2:.1f},{yy + b_ / 2:.1f}" for a_, b_ in corners)}" fill="none" stroke="{AXIS}"/>')
        if k == 0:
            for j, piece in enumerate(label.replace(" after ", "\nafter ").replace("the residual", "\nthe residual").split("\n")):
                body.append(text(gx, yy + hpx / 4 + j * 16, piece, 12, INK2))
foot = gy + 3 * pitch + 16
body.append(text(gx, foot, "Each dot is one held-out position. The corners are the three Mess3 states. A dot's colour is its exact belief under that type,", 11, MUTED))
body.append(text(gx, foot + 15, "so the same colour in one column marks the same positions. All nine panels show the same positions.", 11, MUTED))
bx, by, bh, bw = 896, 330, 180, 226
body.append(text(bx - 40, 104, "Do the three beliefs share a plane?", 13, INK, weight="600"))
body.append(text(bx - 40, 122, f"Plane overlap (1 = same plane; chance {(S - 1) / args['d']:.2f})", 11, INK2))
body.append(yaxis(bx, by, bh, (0, 0.25, 0.5, 0.75, 1), 0, 1, bw))
stage_cols = [GRAY, C1, C2]
for g, (j, k) in enumerate(pairs):
    cx = bx + 4 + g * 75
    for s_, (name, vals) in enumerate(plane_overlap.items()):
        hgt = vals[g] * bh
        body.append(bar(cx + s_ * 24, by, 22, hgt, stage_cols[s_], f"{name}: {vals[g]:.2f}"))
        body.append(text(cx + s_ * 24 + 11, by - hgt - 5, f"{vals[g]:.2f}", 10, INK2, "middle"))
    body.append(text(cx + 35, by + 17, f"{proc.syms[proc.CLOSE0 + j]} and {proc.syms[proc.CLOSE0 + k]}", 12, INK2, "middle"))
body.append(legend(bx + 4, by + 44, [(n, c, "box") for n, c in zip(plane_overlap, stage_cols)]))
lx, ly, lh, lw = 896, 610, 110, 226
body.append(text(lx - 40, ly - lh - 44, "Is a type's belief kept when it is unlikely?", 13, INK, weight="600"))
body.append(text(lx - 40, ly - lh - 26, "Linear readout of a type's state belief", 11, INK2))
chart, xs, ypos = linechart(lx, ly, lw, lh, [("type likely", C1, likely), ("type unlikely", C2, unlikely)], ticks=(0.5, 0.75, 1))
body.append(chart)
body.append(legend(lx + 4, ly + 44, [("type likely (posterior > 0.5)", C1, "line"), ("type unlikely (posterior < 0.1)", C2, "line")]))
fig1 = figure(1140, int(foot + 40), "".join(body), "1. Three state beliefs at once, one plane each",
              "For every bracket the network holds a Mess3 belief for each possible type, whichever type the bracket turns out to be.")

# ---- Figure 2 -------------------------------------------------------------------------------------
body = []


def event_panel(x0, y0, w_, h, rel, ev, marker, title_, sub, refline):
    ex, rd, im, n = ev
    out = [text(x0 - 40, y0 - h - 40, title_, 13, INK, weight="600"), text(x0 - 40, y0 - h - 22, sub, 11, INK2),
           yaxis(x0, y0, h, (0.2, 0.4, 0.6, 0.8), 0.2, 0.9, w_)]
    xs = [x0 + 16 + i * (w_ - 32) / (len(rel) - 1) for i in range(len(rel))]
    ypos = lambda v: y0 - (v - 0.2) / 0.7 * h
    zero = xs[rel.index(0)]
    out.append(f'<rect x="{zero - 12:.1f}" y="{y0 - h:.1f}" width="24" height="{h}" fill="#f0efec"/>')
    out.append(text(zero, y0 - h - 5, marker, 12, INK, "middle", "600"))
    out += [text(xx, y0 + 17, f"{r_:+d}" if r_ else "0", 11, MUTED, "middle") for xx, r_ in zip(xs, rel)]
    out.append(text(x0 + w_ / 2, y0 + 36, f"position relative to the {marker} token   ({n} events)", 11, MUTED, "middle"))
    lab, val = refline
    out += [line(zero + 14, ypos(val), xs[-1] + 12, ypos(val), AXIS), text(xs[-1] + 12, ypos(val) + (15 if val > 0.5 else -6), lab, 11, MUTED, "end")]
    out.append(polyline([(xx, ypos(v)) for xx, v in zip(xs, ex)], INK2))
    for vals, color, label, dx in ((rd, C1, "linear readout", -3.5), (im, C2, "implied by close-token probabilities", 3.5)):
        out += [dot(xx + dx, ypos(v), color, tip=f"{label} at {r_:+d}: {v:.2f} (exact {e:.2f})")
                for xx, v, r_, e in zip(xs, vals, rel, ex) if not np.isnan(v)]
    return "".join(out)


rel_o, rel_c = list(range(-4, 6)), list(range(-3, 4))
sub_o = "Probability of the type the parent favoured before the #"
body.append(event_panel(64, 300, 300, 170, rel_o, open_root, "#", "A bracket opened from the root", sub_o,
                        ("if the parent's belief carried on", open_root[0][3])))
body.append(event_panel(440, 300, 300, 170, rel_o, open_in, "#", "A bracket opened inside another bracket", sub_o,
                        ("if the parent's belief carried on", open_in[0][3])))
body.append(event_panel(816, 300, 280, 170, rel_c, close_ev, "close", "A close, back into the parent bracket",
                        "Probability of the type the parent favoured before the child", ("if the parent were forgotten", 1 / 3)))
body.append(legend(68, 366, [("exact", INK2, "line")]))
body.append(legend(160, 366, [("network: linear readout of the last block", C1, "dot")]))
body.append(legend(470, 366, [("network: implied by its own close-token probabilities (defined inside brackets only)", C2, "dot")]))
# invariance, as a dot plot on a log axis
iy = 440
body.append(text(24, iy - 14, "Nothing outside the bracket leaks in", 13, INK, weight="600"))
body.append(text(24, iy + 4, "Change tokens that exact prediction ignores, and measure how far the network's next-token prediction moves (KL, mnat/token, log scale).", 11, INK2))
c1, c2, ax0, aw_ = 24, 300, 560, 380
lg = lambda v: ax0 + (np.log10(v) + 2) / 5 * aw_
top = iy + 30
body += [text(c1, top, "Tokens changed", 11, MUTED), text(c2, top, "Prediction measured", 11, MUTED), text(ax0 + aw_ + 60, top, "Exact", 11, MUTED)]
for e in range(-2, 4):
    body += [line(lg(10.0 ** e), top + 8, lg(10.0 ** e), top + 8 + 6 * 24, GRID), text(lg(10.0 ** e), top + 8 + 6 * 24 + 14, f"{10.0 ** e:g}", 10, MUTED, "middle")]
for i, r_ in enumerate(invariance):
    for j, (where, val, col, exact_) in enumerate([(r_["where_still"], r_["net_still"], C1, "no change"),
                                                   (r_["where_moved"], r_["net_moved"], GRAY, f"{r_['exact_moved']:.0f}")]):
        yy = top + 24 + (2 * i + j) * 24
        if j == 0:
            body.append(text(c1, yy + 4, r_["name"], 12, INK))
        body += [text(c2, yy + 4, where, 12, INK2), line(lg(0.01), yy, lg(val), yy, GRID), dot(lg(val), yy, col, tip=f"{r_['name']}, {where}: {val:.2f} mnat"),
                 text(lg(val) + 9, yy + 4, f"{val:.1f}" if val < 10 else f"{val:.0f}", 11, INK2), text(ax0 + aw_ + 60, yy + 4, exact_, 12, INK2)]
body.append(legend(c1, top + 8 + 6 * 24 + 40, [("where exact prediction does not change", C1, "dot")]))
body.append(legend(c1 + 300, top + 8 + 6 * 24 + 40, [("where it does (for scale)", GRAY, "dot")]))
fig2 = figure(1140, 690, "".join(body), "2. The belief is refreshed at every open, and returns at a close",
              "Type posterior of the current bracket around an open or a close, averaged over events on held-out sequences.")

# ---- Figure 3 -------------------------------------------------------------------------------------
body = []
tx, ty, th, tw = 70, 310, 180, 300
body.append(text(tx - 40, 90, "Tags become readable in order", 13, INK, weight="600"))
body.append(text(tx - 40, 108, "Linear readout from the residual stream, held-out sequences", 11, INK2))
tag_cols = [C1, C2, C3]
chart, xs, ypos = linechart(tx, ty, tw, th, [(n, c, v) for (n, v), c in zip(tag_r2.items(), tag_cols)])
body.append(chart)
body.append(legend(tx + 4, ty + 44, [(f"{n}   (untrained network at best {tag_blank[n]:.2f})", c, "line") for n, c in zip(tag_r2, tag_cols)]))
# attention shares
ax, ay, aw = 470, 134, 460
body.append(text(ax - 40, 90, "Just after a close, the last block reads the bracket's own tokens", 13, INK, weight="600"))
body.append(text(ax - 40, 108, f"Share of attention by what the attended token is ({int(hard.sum())} positions whose own tokens sit behind the closed child)", 11, INK2))
cat_cols = [C1, C2, C3, C4, C5]
rows_ = [("even spread", even)] + [("block 1, mean of heads", share[0].mean(0))] \
    + [(f"block {L}, head {h}", share[L - 1, h]) for h in np.argsort(-share[L - 1, :, 0])]
for i, (label, vals) in enumerate(rows_):
    yy = ay + i * 30
    body.append(text(ax + 96, yy + 13, label, 11, INK2, "end"))
    xx = ax + 104
    for c, v in enumerate(vals / vals.sum()):
        wid = v * (aw - 104)
        if wid > 2.5:
            body.append(f'<rect x="{xx:.1f}" y="{yy}" width="{wid - 2:.1f}" height="18" rx="3" fill="{cat_cols[c]}"><title>{esc(label)}, {CATS[c]}: {v:.2f}</title></rect>')
            if wid > 30:
                body.append(text(xx + 5, yy + 13, f"{v:.2f}", 10, "#0b0b0b" if c == 3 else "#ffffff"))
        xx += wid
body.append(legend(ax + 104, ay + len(rows_) * 30 + 16, [(c, col, "box") for c, col in zip(CATS[:3], cat_cols[:3])]))
body.append(legend(ax + 304, ay + len(rows_) * 30 + 16, [(c, col, "box") for c, col in zip(CATS[3:], cat_cols[3:])]))
# example heatmaps
cell = 8.4
hx, hy = 60, 480


def heat(x0, y0, M, title_, sub, binary=False):
    out = [text(x0, y0 - 40, title_, 13, INK, weight="600"), text(x0, y0 - 24, sub, 11, INK2)]
    for i in range(W):
        for j in range(i + 1):
            v = M[i, j]
            if v <= (0 if binary else 0.02):
                continue
            col = "#256abf" if binary else SEQ[min(int(v ** 0.6 * (len(SEQ) - 1) + 0.999), len(SEQ) - 1)]
            out.append(f'<rect x="{x0 + j * cell:.1f}" y="{y0 + i * cell:.1f}" width="{cell - 1:.1f}" height="{cell - 1:.1f}" fill="{col}">'
                       f'<title>position {i} reads {j}: {v:.2f}</title></rect>')
    out.append(f'<path d="M{x0},{y0} v{W * cell} h{W * cell}" fill="none" stroke="{AXIS}"/>')
    string = proc.show(xin[ex_b, :W])
    for i, ch in enumerate(string):
        strong = ch in proc.syms[proc.OPEN0:]
        out.append(text(x0 + i * cell + cell / 2 - 0.5, y0 + W * cell + 12, ch, 9, INK if strong else MUTED, "middle", "700" if strong else "400"))
        out.append(text(x0 - 5, y0 + i * cell + cell - 1.5, ch, 9, INK if strong else MUTED, "end", "700" if strong else "400"))
    out.append(text(x0 + W * cell / 2, y0 + W * cell + 28, "token read at the attended position", 11, MUTED, "middle"))
    out.append(text(x0 - 22, y0 + W * cell / 2, "token read at the reading position", 11, MUTED, "middle", rotate=-90))
    return "".join(out)


body.append(heat(hx, hy, ex_own.astype(float), "What the algorithm gathers", "For each position, the content tokens of its own bracket", binary=True))
body.append(heat(hx + 540, hy, ex_attn, f"What block {L} head {reader} attends to", "Attention weights on the same held-out sequence"))
sx = hx + 540 + W * cell + 18
for i, c in enumerate(SEQ[1:]):
    body.append(f'<rect x="{sx}" y="{hy + 130 - i * 18}" width="12" height="17" fill="{c}"/>')
body += [text(sx + 16, hy + 146, "0", 10, MUTED), text(sx + 16, hy + 16, "1", 10, MUTED), text(sx - 2, hy - 2, "weight", 10, MUTED)]
fig3 = figure(1140, 470 + int(W * cell) + 50, "".join(body), "3. Positions are tagged by depth and by bracket, and attention uses the tags",
              "Depth is readable after block 1, the bracket's identity and the count of its tokens by block 3; the last block's heads then select by bracket.")

# ---------------------------------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------------------------------
sub = lambda a, b: f"{a}<sub>{b}</sub>"
example = proc.show(toks[ex_b, :64])
css = f"""body{{margin:0;background:{PAGE};color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color-scheme:light}}
.wrap{{width:1140px;margin:0 auto;padding:28px 0 60px}} h1{{font-size:24px;margin:0 0 6px}} p{{font-size:14px;line-height:1.5;color:{INK2};margin:6px 0}}
.card{{background:{SURF};border:1px solid rgba(11,11,11,0.10);border-radius:10px;margin:18px 0;overflow:hidden}} .pad{{padding:22px 24px}}
table.g{{border-collapse:collapse;font-size:14px;margin:8px 0}} table.g td{{padding:3px 18px 3px 0;vertical-align:baseline;font-variant-numeric:tabular-nums}}
td.r{{font-family:ui-monospace,Menlo,monospace;white-space:nowrap}} td.p{{color:{INK2}}} td.c{{color:{MUTED}}} code{{font-family:ui-monospace,Menlo,monospace;font-size:13px}}
h2{{font-size:17px;margin:0 0 8px}} .cap{{font-size:13px;color:{INK2};padding:0 24px 18px;line-height:1.5}} svg{{display:block}}"""
grammar = f"""<h2>The process, as a PCFG</h2>
<p>Tokens: content <code>A B C</code>, one open bracket <code>#</code>, three close brackets <code>) ] }}</code>. A bracket has a hidden type
<i>k</i> (one of three) and a hidden Mess3 state <i>s</i>. {sub('R','k,s')} is the root, which never closes; {sub('X','k,s')} is an open bracket.</p>
<table class="g">
<tr><td class="r">S → {sub('R','k,s')}</td><td class="p">1/9</td><td class="c">type and state chosen uniformly</td></tr>
<tr><td class="r">{sub('R','k,s')} → x {sub('R',"k,s′")}</td><td class="p">(1 − p<sub>o</sub>) · {sub('T','k')}<sup>x</sup>[s,s′]</td><td class="c">emit a content token; Mess3 of type k moves s → s′</td></tr>
<tr><td class="r">{sub('R','k,s')} → # {sub('X','j,u')} {sub('R','k,s')}</td><td class="p">p<sub>o</sub> / 9</td><td class="c">open a child of type j in state u; then resume in the same state</td></tr>
<tr><td class="r">{sub('X','k,s')} → x {sub('X',"k,s′")}</td><td class="p">(1 − p<sub>o</sub> − p<sub>c</sub>) · {sub('T','k')}<sup>x</sup>[s,s′]</td><td class="c"></td></tr>
<tr><td class="r">{sub('X','k,s')} → # {sub('X','j,u')} {sub('X','k,s')}</td><td class="p">p<sub>o</sub> / 9</td><td class="c"></td></tr>
<tr><td class="r">{sub('X','k,s')} → {sub('c','k')}</td><td class="p">p<sub>c</sub></td><td class="c">close with the token of the bracket's own type: {sub('c','1')} = ), {sub('c','2')} = ], {sub('c','3')} = }}</td></tr>
</table>
<p>x ∈ {{A, B, C}}. {sub('T','k')}<sup>x</sup>[s,s′] is the probability that the Mess3 of type k goes from state s to s′ and emits x.
Mess3 parameters (x, a): type <code>)</code> (0.05, 0.85), sticky; type <code>]</code> (0.15, 0.6), noisy; type <code>}}</code> (0.45, 0.85), anti-persistent.
p<sub>o</sub> = {proc.p_open}, p<sub>c</sub> = {proc.p_close}; no bracket opens at depth {proc.D}; sequences are the first {T} tokens.</p>
<p>The open token is the same for every type and only the close names it, so inside a bracket the type has to be inferred from the content.
Every push and pop is visible in the token itself (a visibly pushdown language), and a bracket's content is independent of everything outside it.</p>
<p>Example: <code>{esc(example)}</code></p>"""
net = (f"One network: {L} layers, d = {args['d']}, {H} heads, context {T}, trained {args['steps']:,} steps at batch {args['batch']} (about 12 minutes). "
       f"One seed. Its loss is {1e3 * algorithm.excess(ref['pred'], logp).mean():.1f} mnat/token above exact Bayes. All readouts are linear, "
       f"fitted on {NTR} sequences and shown on the other {N - NTR}.")
caps = [
    f"Question 0. After block {L - 1} the three beliefs sit in nearly separate planes (overlap {min(plane_overlap[f'after block {L - 1}']):.2f}–{max(plane_overlap[f'after block {L - 1}']):.2f}); "
    f"after block {L} they have partly merged ({min(plane_overlap[f'after block {L}']):.2f}–{max(plane_overlap[f'after block {L}']):.2f}). "
    f"A type's belief is read as well where that type is unlikely (R² {unlikely[-1]:.2f}) as where it is likely ({likely[-1]:.2f}). "
    "Caveats: the readouts are blurred versions of the exact fractals; and in this Mess3 family two of the three beliefs are almost linear functions "
    "of the others (R² 0.97), so the plane estimates are weakly identified.",
    f"Question 1. At a # both readouts fall to the prior (1/3), from the root and from inside a bracket alike. At a close the network's own predictions put "
    f"{close_ev[2][3]:.2f} on the parent's favoured type (exact {close_ev[0][3]:.2f}); the linear readout recovers less ({close_ev[1][3]:.2f}), so the type posterior is not fully linear in the residual there. "
    f"Changing tokens outside the current bracket moves the prediction by {max(r_['net_still'] for r_ in invariance):.1f} mnat at most, "
    f"against about {np.mean([r_['net_moved'] for r_ in invariance]):.0f} where the prediction should change.",
    f"Question 2. Depth is linearly readable after block 1 (R² {tag_r2['depth'][1]:.2f}); the distance back to the bracket's own # and the count of its tokens by block {L - 1} "
    f"({tag_r2['distance back to own #'][L - 1]:.2f}, {tag_r2['own tokens read so far'][L - 1]:.2f}). Just after a close, block {L} head {reader} puts "
    f"{share[L - 1, reader, 0]:.2f} of its attention on the bracket's own tokens, which lie behind the closed child. This shows the tags are present and that attention follows the bracket; "
    "that the heads read the tag itself has not been tested causally.",
]
figs = [fig1, fig2, fig3]
os.makedirs("figures", exist_ok=True)
page = lambda inner, width=1140: (f'<!doctype html><html data-theme="light"><head><meta charset="utf-8"><title>Bracketed Mess3: pilot figures</title>'
                                  f'<style>{css}</style></head><body><div class="wrap" style="width:{width}px">{inner}</div></body></html>')
cards = "".join(f'<div class="card">{f}<div class="cap">{esc(c)}</div></div>' for f, c in zip(figs, caps))
open("figures/report.html", "w").write(page(
    f'<h1>How a transformer tracks beliefs in a bracketed Mess3 process: pilot results</h1><p>{esc(net)}</p>'
    f'<div class="card"><div class="pad">{grammar}</div></div>{cards}'))
open("figures/fig0_process.html", "w").write(page(f'<div class="card" style="margin:0"><div class="pad">{grammar}</div></div>'))
for i, (f, c) in enumerate(zip(figs, caps), 1):
    open(f"figures/fig{i}.html", "w").write(page(f'<div class="card" style="margin:0">{f}<div class="cap">{esc(c)}</div></div>'))
tolist = lambda ev: dict(exact=ev[0].tolist(), readout=ev[1].tolist(), implied=[None if np.isnan(v) else float(v) for v in ev[2]], events=ev[3])
json.dump(dict(plane_overlap=plane_overlap, likely=likely, unlikely=unlikely, open_from_root=tolist(open_root), open_inside=tolist(open_in),
               close=tolist(close_ev), invariance=invariance, tag_r2=tag_r2, tag_blank=tag_blank, attention_share=share.tolist(),
               even=even.tolist(), categories=CATS, example=example, reader=reader), open("figures/data.json", "w"), indent=1)
print("wrote figures/report.html, fig0_process.html, fig1.html, fig2.html, fig3.html, data.json")
for name, ev in (("open from root", open_root), ("open inside", open_in), ("close", close_ev)):
    print(f"{name} ({ev[3]} events)\n  exact   {np.round(ev[0], 2)}\n  readout {np.round(ev[1], 2)}\n  implied {np.round(ev[2], 2)}")
print("example:", example)
