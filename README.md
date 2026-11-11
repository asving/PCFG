# Predicting a bracketed process without a stack

A proposal for how a transformer does exact next-token prediction on a process that needs a stack, and an experiment that tests the proposal step by step.

Status: the algorithm is exact (checked against a stack-based Bayes filter). One small network, trained for 12 minutes, has been tested against it. Everything under "Pilot" is that one seed.

## The process

Tokens are `A B C`, an open bracket `#`, and three close brackets `) ] }`.

- Each `#` opens a bracket of hidden type, one of three, chosen uniformly.
- The `A/B/C` tokens directly inside a bracket are emitted by a Mess3 process whose parameters depend on the bracket's type. The three Mess3s share the vocabulary.
- A bracket closes with the close token of its type, so the type is revealed only at the end.
- Brackets nest. When a child closes, its parent's Mess3 continues from where it stopped.

Example: `#AA#BBB#C}B]A)`. The outer bracket emits `AA…A`, the middle one `BBB…B`, the inner one `C`.

To predict the next token one needs, for the bracket that is currently open, a belief over (type, Mess3 state): 9 numbers. The full Bayesian state is one such belief per open bracket, so it grows with nesting depth. A recurrent filter keeps them on a stack. A transformer has no stack and no recurrence.

## The algorithm

Position `i` has read tokens `0..i-1` and predicts token `i`. Call the tokens directly inside one bracket its **thread**. The algorithm uses three operations, each of which one attention head can perform:

| operation | what it does | as attention |
|---|---|---|
| count | running sum over earlier positions | even attention |
| most recent | latest earlier position that matches a key | content match, tie broken by recency |
| gather | sum over earlier positions that share a tag | content match |

Seven steps. Each leaves one quantity at every position.

| step | quantity at position `i` | how |
|---|---|---|
| 1 depth | opens minus closes among the tokens read | count |
| 2 tag | position of the `#` that opened `i`'s bracket | most recent open whose depth equals `depth[i]` |
| 3 index | number of own-thread tokens read | count over positions with the same tag |
| 4 state | for each type `k`, the Mess3-`k` state belief | gather the thread's tokens, recency counted in thread tokens |
| 5 evidence | at each `A/B/C` token: `log P(token | thread so far, type k)` | local, from step 4 at the previous thread position |
| 6 type | log-posterior of the type | gather the thread's evidence, no decay |
| 7 belief | type posterior × state belief, then the next-token law | local |

Three properties follow.

- **One subspace, any depth.** The belief lives in the same 9 numbers at every position. A `#` does not move it to a new subspace; it changes the tag, and therefore which tokens feed it.
- **Nothing happens at a close.** The depth drops, step 2 returns the parent's tag, and steps 3–7 gather the parent's tokens again. No belief is stored and fetched.
- **Nothing is carried.** Inside a child, no position holds the parent's belief. The parent's tokens stay where they are.

Worked example (`python algorithm.py` prints it). Columns are positions; each has read the token shown.

```
read             BOS    #    A    A    #    B    B    B    #    C    }    B    ]    A
position           0    1    2    3    4    5    6    7    8    9   10   11   12   13
1 depth            0    1    1    1    2    2    2    2    3    3    2    2    1    1
2 tag (own #)      0    1    1    1    4    4    4    4    8    8    4    4    1    1
3 index            0    0    1    2    0    1    2    3    0    1    3    4    2    3
```

Position 10 has just read `}`. Its depth is back to 2, the most recent `#` at depth 2 is position 4, so its tag is 4 and it gathers `BBB`.

`algorithm.py` implements these steps with the three operations only and reproduces the stack filter to 1e-14 on 131,072 positions.

## What each position should hold

| the position has just read | depth | tag | index | belief subspace holds |
|---|---|---|---|---|
| `A/B/C` | unchanged | unchanged | +1 | own bracket's belief, updated |
| `#` | +1 | this position | 0 | the prior |
| a close | −1 | the parent's `#` | the parent's count | the parent's belief as of before the child |

## Shortcuts, and where each would show

A network could do something cheaper. `algorithm.py` implements four shortcuts and prices them, in excess loss over Bayes (mnat/token), by position class:

| | all | new bracket with an earlier one at its depth | just read a close | 1–3 tokens after a short child | 1–3 tokens after a long child |
|---|---|---|---|---|---|
| state from the last 4 own tokens only | 0.2 | 0.0 | 0.2 | 0.2 | 0.2 |
| state from the last 2 own tokens only | 3.8 | 0.0 | 3.7 | 4.5 | 4.5 |
| recency counted in positions, not thread tokens | 2.1 | 0.0 | 2.0 | 5.7 | 13.2 |
| depth tag only, no bracket tag | 47.0 | 143.5 | 42.6 | 37.3 | 27.7 |
| keeps the type, forgets the state at a close | 7.6 | 0.0 | 57.8 | 14.8 | 14.5 |
| forgets the parent at a close | 18.9 | 0.0 | 73.3 | 51.2 | 51.8 |

So the state belief needs only the last few thread tokens, the type needs the whole thread, and the bracket tag (not depth alone) is the expensive thing to get wrong.

## The experiment

Train a small causal transformer on the process (`train.py`), then run `validate.py`. It makes four measurements, each with a prediction from the algorithm.

1. **Loss.** Excess loss over Bayes in the position classes above. Prediction: near zero in every column, flat in depth.
2. **Probes.** Linear readout of each step's quantity from the residual after each block, with an untrained network as control. Prediction: the quantities become readable in the order of the steps, and the parent's belief is never readable at a child's positions.
3. **Attention.** For each head, the share of attention by what the attended position is to the current one: own thread, own `#`, inside closed children, enclosing brackets, earlier brackets. Prediction: even-spread heads early (the count), then heads that read only the own thread, including just after a close, when the thread sits behind the child. A second test asks whether recency is counted in thread tokens or in positions.
4. **Patching.** After a close at `q`, change the parent's earlier tokens `R` (all but the last one before the child, at `p`). Overwrite the residual after one block at one set of positions with its value in the changed run, and measure what fraction of the change in `q`'s prediction results. Prediction: nothing travels through the child, or reaches `q` before the last block; `p` carries only its own evidence term, a minority share that the algorithm itself predicts.

## Pilot

One seed: 4 layers, d = 128, 4 heads, context 128, batch 64, 8,000 steps, 12 minutes on a laptop GPU. Evaluated on 768 held-out sequences.

**1. Loss** (mnat/token): 2.6 over all positions; 0.3 in new brackets with an earlier one at the same depth (depth-only tagging would give 143.5); 6.1 just after a close (forgetting the parent would give 73.3); 4.0 and 5.7 after short and long children. By depth 0 to 6: 2.1, 2.3, 2.6, 2.9, 3.0, 3.2, 3.8, a mild rise.

**2. Probes** (R² on held-out sequences):

| quantity | emb | blk 1 | blk 2 | blk 3 | blk 4 | best untrained |
|---|---|---|---|---|---|---|
| 1 depth | 0.13 | 0.95 | 0.99 | 0.99 | 0.98 | 0.23 |
| 2 tag (distance back to own `#`) | 0.16 | 0.20 | 0.56 | 0.83 | 0.81 | 0.16 |
| 3 index | 0.11 | 0.39 | 0.68 | 0.86 | 0.86 | 0.15 |
| 4 state belief per type | 0.78 | 0.80 | 0.87 | 0.91 | 0.93 | 0.79 |
| 5 evidence of the token just read | 0.00 | 0.00 | 0.72 | 0.88 | 0.89 | 0.00 |
| 6 type log-posterior | 0.00 | 0.03 | 0.25 | 0.35 | 0.82 | 0.00 |
| 7 belief of the own bracket | 0.31 | 0.33 | 0.47 | 0.55 | 0.84 | 0.32 |
| belief of the parent bracket | 0.00 | 0.04 | 0.04 | 0.12 | 0.11 | 0.02 |

The order matches the steps: depth after block 1, tag and index over blocks 2–3, evidence from block 2, type and belief at block 4. The state row is weak evidence, because the last token alone predicts most of the state (0.78 from the embedding).

**3. Attention.** All four block-1 heads match even spread to within 0.02. Just after a close with at least three own tokens behind the child (2,971 positions; even spread would put 0.12 on own tokens, 0.24 inside closed children, 0.37 on enclosing brackets):

| head | own tokens | own `#` | inside closed children | enclosing brackets |
|---|---|---|---|---|
| block 4 head 1 | 0.90 | 0.05 | 0.03 | 0.01 |
| block 4 head 3 | 0.73 | 0.16 | 0.03 | 0.01 |
| block 4 head 2 | 0.64 | 0.27 | 0.03 | 0.01 |
| block 3 head 0 | 0.48 | 0.02 | 0.08 | 0.18 |

Recency test on the three block-4 readers (share of attention on the third most recent own token, relative to the second): head 1 gives 0.47 with no child in between, 0.45 with a short child, 0.46 with a long one, as counting in thread tokens predicts. Heads 2 and 3 shift, in opposite directions (0.54, 0.54, 0.68 and 0.52, 0.40, 0.35).

**4. Patching** (723 events; the change moves the Bayes law at `q` by 166 mnat; the network's change has cosine 0.95 with the Bayes change):

| patched after | embedding | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| `R`, the changed tokens | 1.00 | 1.00 | 0.81 | 0.76 |
| `p`, last own token before the child | 0.00 | 0.00 | 0.19 | 0.22 |
| everything else before `q` | 0.00 | 0.00 | 0.00 | 0.00 |
| `q` itself | 0.00 | 0.00 | 0.00 | 0.02 |

The same patch applied to the algorithm after its evidence step gives `R` 0.74 and `p` 0.31. So block 4 reads the parent's information from the parent's own token positions, nothing passes through the child, and the share at `p` is about what its evidence term accounts for.

**Not explained.**
- Block 2 head 2 puts 0.38 of its attention on the previous position in the thread overall, but just after a close it reads inside the closed child (0.70) and not the thread's previous position (0.02).
- Block 4 head 0 puts 0.22–0.27 on brackets that closed before the current one opened.
- The parent's belief is faintly readable at child positions (0.11, control 0.02), although patching shows nothing passing through the child.
- A probe fitted only at depth ≤ 1 fails at depth ≥ 3, while one probe fitted on all depths works at each. Not followed up.

## Not yet done

- More seeds. Everything above is one.
- Width below 72, the size that would hold a full depth-8 stack of beliefs. The pilot's d = 128 does not test whether the per-token dimension is bounded.
- Fewer and more layers. The algorithm suggests about four attention steps.
- A hand-built transformer that implements the seven steps.
- Depths beyond those seen in training.
- Coupled brackets, below.

## Files

| file | what |
|---|---|
| `process.py` | sampler, stack-based exact filter (the ground truth), observers that forget at a close |
| `algorithm.py` | the seven steps with attention-style operations; the shortcuts; position classes; run it for the self-check, the worked example and the shortcut table |
| `model.py` | the transformer, and a hand-run forward pass that returns residuals and attention and takes patches |
| `train.py` | training, with excess loss by position class at each eval |
| `validate.py` | the four measurements |
| `check_coupled.py` | side result, below |
| `runs/` | training logs; checkpoints are not committed |

```
python algorithm.py
python train.py --steps 8000 --batch 64 --device mps
python validate.py runs/pilot_d128_L4_T128_s0
```

Needs numpy and torch. The Mess3 parameters are (0.05, 0.85), (0.15, 0.6), (0.45, 0.85); the open and close rates are 0.10 and 0.12; depth is capped at 8.

## Beyond independent brackets

Here a bracket's contents are independent of everything outside it, given the bracket structure. That is what makes the belief of one bracket depend only on its own thread. `check_coupled.py` checks a variant in which a child's type depends on its parent's and a single close token never reveals it. The belief of the active bracket is still a sufficient, local statistic there (agreement with brute-force enumeration to 3.5e-14), but computing it needs one read of the parent per nesting level, which a fixed-depth network cannot do for arbitrary depth. That case is not part of this experiment.
