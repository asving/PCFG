"""Coupled brackets: does a network's loss land on the floor that its depth allows?

A child's type equals its parent's with probability rho, otherwise it is uniform; the close token names
the type. The active bracket's belief then needs the parent's type posterior at the open, which needed
the grandparent's, and so on: one chained read per nesting level (see coupled.py). This trains a
transformer on that process and prints its excess loss over exact Bayes beside the excess loss of
observers that chain only 0, 1, 2 or 3 reads.

  python coupled_train.py --layers 4 --device mps

Columns: all positions; the first three positions of a bracket at depth >= 2 (where the chain matters
most); positions that have just read a close.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from process import BracketProcess, FAMILIES
from model import GPT
import algorithm

HOPS = (0, 1, 2, 3)


class CoupledProcess(BracketProcess):
    def __init__(self, rho, **kw):
        super().__init__(FAMILIES["mess3"](), **kw)
        self.PI = rho * np.eye(self.K) + (1 - rho) / self.K          # [parent type, child type]
        self.cumPI = self.PI.cumsum(-1)

    def sample(self, batch, length, rng):
        K, D, S, C = self.K, self.D, self.S, self.C
        typ = np.zeros((batch, D + 1), dtype=np.int64)
        eta = np.zeros((batch, D + 1, S))
        depth = np.zeros(batch, dtype=np.int64)
        rows = np.arange(batch)
        typ[:, 0] = rng.integers(0, K, batch)
        eta[:, 0] = self.eta0[typ[:, 0]]
        toks = np.zeros((batch, length), dtype=np.int64)
        for t in range(length):
            po, pc = self.struct_probs(depth)
            u = rng.random(batch)
            opn, cls = u < po, (u >= po) & (u < po + pc)
            emt = ~(opn | cls)
            k, e = typ[rows, depth], eta[rows, depth]
            probs = np.clip(np.einsum("bs,bxs->bx", e, self.emit[k]), 0, None)
            probs /= probs.sum(-1, keepdims=True)
            tok = (rng.random(batch)[:, None] > probs.cumsum(-1)).sum(-1).clip(0, C - 1)
            new = np.einsum("bs,bsr->br", e, self.T[k, tok])
            new /= (new * self.r[k]).sum(-1, keepdims=True)
            newk = (rng.random(batch)[:, None] > self.cumPI[k]).sum(-1).clip(0, K - 1)   # depends on the parent
            toks[:, t] = np.where(emt, tok, np.where(opn, self.OPEN0, self.CLOSE0 + k))
            eta[rows, depth] = np.where(emt[:, None], new, e)
            depth = depth + opn - cls
            typ[rows[opn], depth[opn]] = newk[opn]
            eta[rows[opn], depth[opn]] = self.eta0[newk[opn]]
        return toks

    def ladder(self, toks):
        """Next-token laws of observers that chain 0, 1, 2, 3 or unlimited reads of the parent.
        Returns pred [len(HOPS) + 1, B, L, VOCAB]; the last entry is exact."""
        B, L = toks.shape
        K, S, D, C = self.K, self.S, self.D, self.C
        Hn = len(HOPS) + 1
        v = np.broadcast_to(self.eta0, (B, D + 1, K, S)).copy()
        ll = np.zeros((B, D + 1, K))                     # own tokens and revealed children, per type
        prior = np.full((B, D + 1, Hn, K), 1.0 / K)
        depth = np.zeros(B, dtype=np.int64)
        rows = np.arange(B)
        pred = np.zeros((Hn, B, L, self.VOCAB))
        for t in range(L):
            top = v[rows, depth]
            lw = np.log(prior[rows, depth]) + ll[rows, depth][:, None, :]
            w = np.exp(lw - lw.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)                # [B, Hn, K]
            po, pc = self.struct_probs(depth)
            emit_k = np.einsum("bks,kxs->bkx", top, self.emit)
            pred[:, :, t, :C] = (1 - po - pc)[None, :, None] * np.einsum("bhk,bkx->hbx", w, emit_k)
            pred[:, :, t, self.OPEN0] = po[None]
            pred[:, :, t, self.CLOSE0:] = pc[None, :, None] * w.transpose(1, 0, 2)
            x = toks[:, t]
            emt, opn, cls = self.is_content(x), self.is_open(x), self.is_close(x)
            if emt.any():
                new = np.einsum("bks,kbsr->bkr", top[emt], self.T[:, x[emt]])
                mass = np.maximum((new * self.r).sum(-1), 1e-300)
                ll[rows[emt], depth[emt]] += np.log(mass)
                v[rows[emt], depth[emt]] = new / mass[..., None]
            if cls.any():                                # the close names the child's type: evidence for the parent
                ll[rows[cls], depth[cls] - 1] += np.log(self.PI[:, x[cls] - self.CLOSE0]).T
            if opn.any():
                wp = w[opn]
                child = np.empty((opn.sum(), Hn, K))
                child[:, 0] = 1.0 / K
                for i in range(1, Hn - 1):
                    child[:, i] = wp[:, i - 1] @ self.PI
                child[:, Hn - 1] = wp[:, Hn - 1] @ self.PI
                r, d = rows[opn], depth[opn] + 1
                prior[r, d], v[r, d], ll[r, d] = child, self.eta0, 0.0
            depth = depth + opn - cls
        return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.9)
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
    ap.add_argument("--check", action="store_true", help="compare the batched ladder with coupled.py and exit")
    args = ap.parse_args()
    proc = CoupledProcess(args.rho)
    if args.check:
        from coupled import Coupled, INF
        toks = proc.sample(40, 96, np.random.default_rng(5))
        slow = Coupled(FAMILIES["mess3"](), proc.PI, True)
        fast = proc.ladder(toks)
        obs = [(h, INF) for h in HOPS] + [(INF, INF)]
        worst = max(np.abs(fast[i, b] - slow.filter(toks[b], obs)[o]).max() for b in range(len(toks)) for i, o in enumerate(obs))
        base = proc.filter(toks, "bayes")[0]
        print(f"batched ladder vs coupled.py on {toks.size} positions: max |difference| {worst:.1e}; "
              f"exact differs from the uncoupled filter by up to {np.abs(fast[-1] - base).max():.2f}")
        return

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed + 1)
    model = GPT(proc.VOCAB, args.T, args.d, args.layers, args.heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    name = f"coupled{args.rho}_night_d{args.d}_L{args.layers}_T{args.T}_s{args.seed}"

    etoks = proc.sample(args.eval_seqs, args.T, np.random.default_rng(12345))
    ex = torch.cat([torch.full((args.eval_seqs, 1), proc.BOS), torch.from_numpy(etoks)[:, :-1]], 1).to(dev)
    lad = proc.ladder(etoks)
    exact = lad[-1]
    st = algorithm.run(proc, etoks)                       # structure only: depth and own-token count
    cells = {"all": np.ones_like(etoks, dtype=bool), "new bracket, depth >= 2": (st["depth"] >= 2) & (st["index"] <= 2),
             "just read a close": proc.is_close(st["read"])}
    rows = [("ignores the coupling", proc.filter(etoks, "bayes")[0])] \
        + [(f"{h} read{'s' if h != 1 else ''} of the parent", lad[i]) for i, h in enumerate(HOPS)]
    floors = {n: {c: float(algorithm.excess(exact, p)[m].mean()) for c, m in cells.items()} for n, p in rows}
    print(f"{name}: excess loss over exact Bayes, mnat/token")
    print(f"{'':28s}" + "".join(f"{c:>26s}" for c in cells))
    for n, f_ in floors.items():
        print(f"{n:28s}" + "".join(f"{1e3 * f_[c]:26.1f}" for c in cells))
    log = dict(args=vars(args), floors=floors, history=[])
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
            row.update(step=s, time=time.time() - t0)
            log["history"].append(row)
            json.dump(log, open(f"runs/{name}.json", "w"))
            torch.save(model.state_dict(), f"runs/{name}.pt")
            print(f"network, step {s:6d}{'':9s}" + "".join(f"{1e3 * row[c]:26.1f}" for c in cells) + f"   {row['time']:.0f}s", flush=True)
        if s < args.steps:
            y = torch.from_numpy(proc.sample(args.batch, args.T, rng))
            x = torch.cat([torch.full((args.batch, 1), proc.BOS), y[:, :-1]], 1).to(dev)
            loss = F.cross_entropy(model(x).reshape(-1, proc.VOCAB), y.to(dev).reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()


if __name__ == "__main__":
    main()
