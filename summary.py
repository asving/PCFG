"""One line per finished run, from runs/*.validate.txt.

  python summary.py [pattern]

Loss columns are excess loss over Bayes in mnat/token. Readouts are R^2 after the last block
(type-posterior and belief), or the first block at which the readout passes 0.8 (tags and evidence).
"""
import glob, re, sys

pat = sys.argv[1] if len(sys.argv) > 1 else "*"
rows = []
for path in sorted(glob.glob(f"runs/{pat}.validate.txt")):
    txt = open(path).read()
    name = path[5:-13]
    g = lambda rx, default=None: (re.search(rx, txt, re.M) or [None, default])[1]
    net = g(r"^THE NETWORK\s+(.*)$")
    if not net:
        continue
    net = [float(v) for v in net.split()]
    depth = [float(v) for v in g(r"^the network by depth 0\.\.6:\s+(.*)$", "").split()]

    def probe(label):
        m = re.search(rf"^{label}.*?(?:early|after depth|after tag|middle|late|last|absent)\s+(.*)$", txt, re.M)
        vals = [float(v) for v in m[1].split()] if m else []
        return vals[:-1], (vals[-1] if vals else None)          # per stage, best untrained

    def first(vals, thr=0.8):
        return next((str(i) for i, v in enumerate(vals) if v >= thr), "-")
    dep, tag, idx, ev = probe("1 depth")[0], probe("2 tag")[0], probe("3 index")[0], probe("5 evidence")[0]
    typ, bel, par = probe("6 type")[0], probe("7 belief")[0], probe(r"  belief of the PARENT")
    inv = re.findall(r"^(?:enclosing|a closed).*?changed\s+[\d.]+\s+([\d.]+)\s+\d+\s+([\d.]+)\s+([\d.]+)", txt, re.M)
    patch = re.search(r"^R: the changed own tokens\s+(.*)\n^p: last own token.*?\s+(.*)\n^everything else before q\s+(.*)$", txt, re.M)
    alg = re.search(r"patched after its evidence step:\s+R ([\d.]+)\s+p ([\d.]+)", txt)
    rows.append((name, net, depth, first(dep), first(tag), first(idx), first(ev),
                 typ[-1] if typ else None, bel[-1] if bel else None, par[0][-1] if par[0] else None, par[1],
                 max(float(a) for a, _, _ in inv) if inv else None,
                 (patch[1].split()[-1], patch[2].split()[-1], patch[3].split()[-1]) if patch else None,
                 (alg[1], alg[2]) if alg else None))

f = lambda v: "   -" if v is None else f"{v:4.2f}"
print(f"{'run':34s}{'all':>6s}{'sib':>6s}{'close':>6s}{'short':>6s}{'long':>6s} | by depth 0..6{'':22s}| first block >= 0.8: dep tag idx evid | "
      f"type belief parent(untr) | leak | patch R/p/else (alg R/p)")
for name, net, depth, d1, t1, i1, e1, typ, bel, par, par0, leak, patch, alg in rows:
    print(f"{name:34s}{net[0]:6.1f}{net[2]:6.1f}{net[4]:6.1f}{net[5]:6.1f}{net[6]:6.1f} | " + " ".join(f"{v:4.1f}" for v in depth).ljust(35)
          + f"| {d1:>19s} {t1:>3s} {i1:>3s} {e1:>4s} | {f(typ)} {f(bel)}   {f(par)} ({f(par0)}) | {f(leak)} | "
          + (f"{patch[0]}/{patch[1]}/{patch[2]}" if patch else "-") + (f" ({alg[0]}/{alg[1]})" if alg else ""))
