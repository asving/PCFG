"""Train a transformer on a coupled bracket process with hidden types (vpa.py), and score it against the
observers that cut the down or the up recursion short.

  python vpa_train.py --config vpa_shift --layers 4 --device mps

Every evaluation prints the network's excess loss over exact Bayes (mnat/token) in the same columns as the
table of cuts printed at the start, then by nesting depth.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
import vpa
from model import GPT
import algorithm


def cells_of(proc, toks, ref):
    depth = ref["depth"]
    return {"all": np.ones_like(depth, dtype=bool), "depth >= 3": depth >= 3, "depth >= 5": depth >= 5,
            "just read a close": proc.is_close(algorithm.read_tokens(proc, toks))}


def cut_name(n):
    if n == "prior":
        return "no inference at all"
    if n == "ignore":
        return "neither recursion"
    return f"{n[1]} read{'s' if n[1] != 1 else ''} down, all up" if n[0] == "down" else f"all down, {n[1]} level{'s' if n[1] != 1 else ''} up"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vpa_shift")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_seqs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--depth_input", action="store_true")
    ap.add_argument("--tag_input", action="store_true")
    ap.add_argument("--depth_rotary", action="store_true")
    ap.add_argument("--pos_rotary", action="store_true")
    args = ap.parse_args()
    proc = vpa.make(args.config)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed + 1)
    ids = (proc.OPEN0, proc.CLOSE0)
    model = GPT(proc.VOCAB, args.T, args.d, args.layers, args.heads, depth_in=ids if args.depth_input else None,
                tag_in=ids if args.tag_input else None, depth_rot=ids if args.depth_rotary else None, pos_rot=args.pos_rotary).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    name = f"{args.config}_{args.tag}_d{args.d}_L{args.layers}_T{args.T}_s{args.seed}"

    etoks, _ = proc.sample(args.eval_seqs, args.T, np.random.default_rng(12345))
    ex = torch.from_numpy(algorithm.read_tokens(proc, etoks)).to(dev)
    preds, ref = proc.filter(etoks)
    exact, depth = preds["exact"], ref["depth"]
    cells = cells_of(proc, etoks, ref)
    cuts = ["prior", "ignore"] + [n for n in preds if isinstance(n, tuple)]
    floors = {cut_name(n): {c: float(algorithm.excess(exact, preds[n])[m].mean()) for c, m in cells.items()} for n in cuts}
    floors_depth = {cut_name(n): [float(algorithm.excess(exact, preds[n])[depth == d].mean()) for d in range(proc.D + 1)] for n in cuts}
    print(f"{name}: excess loss over exact Bayes, mnat/token")
    print(f"{'':30s}" + "".join(f"{c:>20s}" for c in cells))
    for n, f_ in floors.items():
        print(f"{n:30s}" + "".join(f"{1e3 * f_[c]:20.1f}" for c in cells))
    log = dict(args=vars(args), floors=floors, floors_depth=floors_depth, history=[])
    os.makedirs("runs", exist_ok=True)
    t0 = time.time()
    for s in range(args.steps + 1):
        if s % args.eval_every == 0 or s == args.steps:
            model.eval()
            with torch.no_grad():
                lp = torch.cat([F.log_softmax(model(ex[i:i + 512]), -1).cpu() for i in range(0, len(ex), 512)])
            model.train()
            kl = algorithm.excess(exact, lp.double().numpy())
            row = {c: float(kl[m].mean()) for c, m in cells.items()}
            row.update(step=s, time=time.time() - t0, by_depth=[float(kl[depth == d].mean()) for d in range(proc.D + 1)])
            log["history"].append(row)
            json.dump(log, open(f"runs/{name}.json", "w"))
            torch.save(model.state_dict(), f"runs/{name}.pt")
            print(f"network, step {s:6d}{'':11s}" + "".join(f"{1e3 * row[c]:20.1f}" for c in cells) + f"   {row['time']:.0f}s   by depth: "
                  + " ".join(f"{1e3 * v:.1f}" for v in row["by_depth"]), flush=True)
        if s < args.steps:
            toks, _ = proc.sample(args.batch, args.T, rng)
            y = torch.from_numpy(toks).to(dev)
            x = torch.from_numpy(algorithm.read_tokens(proc, toks)).to(dev)
            loss = F.cross_entropy(model(x).reshape(-1, proc.VOCAB), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()


if __name__ == "__main__":
    main()
