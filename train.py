"""Train a small causal transformer on a bracketed process (see process.FAMILIES).

At each eval it prints the network's excess loss over exact Bayes (KL from the Bayes law, mnat/token)
in the position classes of algorithm.cells, the same columns in which algorithm.py prices each
shortcut. validate.py then tests the trained network against the algorithm step by step.

  python train.py --steps 8000 --batch 64 --device mps      # the pilot
  python train.py --bench                                    # timing only
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from process import make
from model import GPT
import algorithm


def batch(proc, B, T, rng, device):
    toks, _ = proc.sample(B, T, rng)
    y = torch.from_numpy(toks)
    x = torch.cat([torch.full((B, 1), proc.BOS), y[:, :-1]], 1)
    return x.to(device), y.to(device), toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="mess3", help="a family in process.FAMILIES, optionally with +open")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--p_open", type=float, default=0.10)
    ap.add_argument("--p_close", type=float, default=0.12)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_seqs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=1e9)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--depth_input", action="store_true", help="add the nesting depth to the input embedding, linearly")
    ap.add_argument("--tag_input", action="store_true", help="add the bracket tag (position of the own open) to the input embedding")
    ap.add_argument("--depth_rotary", action="store_true", help="rotate queries and keys by depth, so scores depend on depth differences")
    ap.add_argument("--pos_rotary", action="store_true", help="rotary position embeddings in place of the learned position table")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    proc = make(args.process, p_open=args.p_open, p_close=args.p_close, max_depth=args.max_depth)
    rng = np.random.default_rng(args.seed + 1)
    ids = (proc.OPEN0, proc.CLOSE0)
    model = GPT(proc.VOCAB, args.T, args.d, args.layers, args.heads, depth_in=ids if args.depth_input else None,
                tag_in=ids if args.tag_input else None, depth_rot=ids if args.depth_rotary else None,
                pos_rot=args.pos_rotary).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    name = f"{args.tag}_d{args.d}_L{args.layers}_T{args.T}_s{args.seed}"
    if args.process != "mess3":
        name = f"{args.process}_{name}"

    def step():
        x, y, _ = batch(proc, args.batch, args.T, rng, dev)
        loss = F.cross_entropy(model(x).reshape(-1, proc.VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        return loss

    if args.bench:
        for _ in range(5):
            step()
        n, t0 = 0, time.time()
        while time.time() - t0 < 8:
            step().item()
            n += 1
        dt = (time.time() - t0) / n
        print(f"{args.device} d{args.d} L{args.layers} T{args.T} B{args.batch}: {1 / dt:.1f} steps/s, "
              f"{args.batch * args.T / dt / 1e6:.2f}M tokens/s", flush=True)
        return

    ex, _, etoks = batch(proc, args.eval_seqs, args.T, np.random.default_rng(12345), dev)
    print("what each shortcut would cost on the eval set:")
    ref, cs = algorithm.loss_table(proc, etoks)
    log = dict(args=vars(args), cells=list(cs), history=[])
    os.makedirs("runs", exist_ok=True)
    print("\nthe network, same columns:")
    t0 = time.time()
    for s in range(args.steps + 1):
        if s % args.eval_every == 0 or s == args.steps:
            model.eval()
            with torch.no_grad():
                lp = torch.cat([F.log_softmax(model(ex[i:i + 512]), -1).cpu() for i in range(0, len(ex), 512)])
            model.train()
            kl = algorithm.excess(ref["pred"], lp.double().numpy())
            row = {n: float(kl[m].mean()) for n, m in cs.items()}
            row["by_depth"] = [float(kl[ref["depth"] == d].mean()) for d in range(proc.D + 1)
                               if (ref["depth"] == d).any()]
            row.update(step=s, time=time.time() - t0)
            log["history"].append(row)
            json.dump(log, open(f"runs/{name}.json", "w"))
            torch.save(model.state_dict(), f"runs/{name}.pt")
            print(f"step {s:6d} {'':27s}" + "".join(f"{1e3 * row[n]:18.1f}" for n in cs)
                  + f"   {row['time']:.0f}s", flush=True)
            if time.time() - t0 > args.time_limit:
                break
        if s < args.steps:
            step()
    torch.save(model.state_dict(), f"runs/{name}.pt")


if __name__ == "__main__":
    main()
