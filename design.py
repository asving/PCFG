"""Which processes would test the algorithm hardest? A design study with no training.

For each candidate process this prints what every shortcut would cost (excess loss over Bayes,
mnat/token). A process is informative about a step of the algorithm when skipping that step is
expensive, because then a trained network's loss says whether the step is being done.

  python design.py [n_seq]

Columns:
  bayes         entropy rate of the process, nats/token
  infer         what all inference is worth: excess loss of an observer that never updates
  H(type) n=8   type-posterior entropy after 8 tokens of one uninterrupted bracket (ln K at the start)
  win 4, win 2  state belief from only the last 4 / 2 own tokens            (step 4: how long a memory)
  pos clock     recency counted in positions; overall, and 1-3 tokens after a long child  (step 3)
  depth tag     no bracket tag; overall, and in a new bracket with an earlier one at its depth (step 2)
  no reset      the thread also includes the enclosing brackets' tokens                  (step 2)
  no type       no type inference                                                        (steps 5-6)
  forget state / forget all    at the position that has just read a close    (what a close must restore)
"""
import sys
import numpy as np
from process import make, BracketProcess, FAMILIES
import algorithm

n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 512
CANDIDATES = ["mess3", "mess3_sharp", "mess3_slow", "clock3", "tomq", "fern",
              "mess3+open", "clock3+open", "tomq+open", "fern+open", "mess3_single"]
COLS = [("win 4", "state from last 4 own tokens", "all"),
        ("win 2", "state from last 2 own tokens", "all"),
        ("pos clock", "recency counted in positions", "all"),
        ("..long child", "recency counted in positions", "after long child"),
        ("depth tag", "depth tag only, no bracket tag", "all"),
        ("..sibling", "depth tag only, no bracket tag", "new, has sibling"),
        ("no reset", "no reset at an open", "all"),
        ("no type", "no type inference", "all"),
        ("forget state", "keeps type, forgets state at a close", "at a close"),
        ("forget all", "forgets the parent at a close", "at a close")]

print(f"{n_seq} sequences of 128 tokens per process; open rate 0.10, close rate 0.12, depth capped at 8\n")
print(f"{'process':14s}{'bayes':>7s}{'infer':>7s}{'H(type) n=8':>13s}" + "".join(f"{c[0]:>14s}" for c in COLS))
for name in CANDIDATES:
    proc = make(name)
    toks, _ = proc.sample(n_seq, 128, np.random.default_rng(0))
    losses, ref, cs = algorithm.shortcut_losses(proc, toks)
    bayes = -np.log(np.take_along_axis(ref["pred"], toks[..., None], -1)).mean()
    infer = algorithm.excess(ref["pred"], proc.filter(toks, "prior")[0]).mean()
    flat = BracketProcess(FAMILIES[name.replace("+open", "")](), p_open=0.0, p_close=0.0)
    w = algorithm.run(flat, flat.sample(n_seq, 9, np.random.default_rng(1))[0])["w"][:, 8]
    ent = -(w * np.log(w + 1e-12)).sum(-1).mean()
    print(f"{name:14s}{bayes:7.3f}{1e3 * infer:7.0f}{ent:8.2f} / {np.log(proc.K):.2f}"
          + "".join(f"{1e3 * losses[row][cs[cell]].mean():14.1f}" for _, row, cell in COLS), flush=True)
