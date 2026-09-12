# Hierarchical action tokenizer (unified action space)

Discrete tokenizer for VLA action chunks: a `T x 119` chunk of the unified `bimanual_rotation6d` space plus its
`{0,1}` padding mask becomes `N` tokens from a vocabulary of `V` codes. `T`, `N` and `V` are independent settings;
the model sees padding and body-part structure explicitly instead of guessing from zeros, and it never sees the
robot state.

## How it works

1. **Pointwise embedding.** Every `(timestep, dimension)` becomes a vector: real values through a shared scalar
   projector plus a per-dimension embedding, padded entries through one learned `[PAD]` vector; a group embedding
   (17 semantic groups: arms, hands, head, torso, base, legs) and a time embedding are added on top.
2. **Per-step queries.** `K = 17 + free` queries attend to the 119 positions of their timestep. Group queries are
   restricted by an attention mask to their own group, so a missing group produces a "group absent" token; free
   queries see everything. A within-step self-attention block afterwards lets the queries of a timestep mix.
3. **Dynamics.** Self-attention over all `T*K` tokens with a time code: the tokenizer models the trajectory, not
   isolated points.
4. **Latents.** `N` learned queries compress the sequence through Perceiver blocks; the quantizer (VQ-EMA by
   default, FSQ / LFQ / Gumbel available) turns them into `N` tokens.
5. **Decoder.** Mirrors the encoder, is conditioned on the same mask, and emits exactly zeros where the mask says
   padding; per-dimension heads read their group token plus a pooled summary of the free tokens.

Loss: masked MSE over real entries only (optional per-group weights) plus the quantizer's auxiliary term.

**A fixed grid needs a warmup; a learned codebook does not.** Quantizing with FSQ from step 0 collapses the code
to a single value and the encoder to a constant, so `train.quantizer_warmup_steps` (10,000 by default) trains the
plain autoencoder first, and the latents pass through the quantizer's `bound` so both phases share one value range.
VQ-EMA, the default arm, needs none of this: its codebook follows whatever scale the encoder settles on, and at an
identical budget `quantizer_warmup_steps=0` used 103 of 110 bits against 95 with a warmup. The default stays at
10,000 because it is the safe value for every arm; pass 0 for the learned codebook.

**Watch the per-position numbers, not the pooled ones.** `eval/perplexity` counts all token positions together and
happily reports 2031 of 2048 codes while three of ten positions carry one bit each. `eval/min_position_perplexity`
and `eval/effective_bits` are the honest measures of what the bottleneck carries.

## Data

Chunks come from the `action_chunks` package (mixture R0_v2.1, 95 embodiments, manifest `r0_v2.1-6885099`):
`WeightedChunkDataset` with the VLA mixture weights for training, and a frozen eval set of 44,642 held-out chunks
(at most 500 per embodiment) shared with the FAST / BEAST / ActionCodec / OAT arms, so every number is comparable.

## Running

```bash
python train.py run_name=hier-vq-n10-v2048                     # defaults: N=10, V=2048 (VQ-EMA), 300k steps
python train.py model=hier_fsq model.num_tokens=16 train.steps=100000
accelerate launch --multi_gpu --num_processes 8 train.py run_name=hier-vq-n10-v2048
```

On the cluster use `launchers/`: `smoke.sh` (IB node, tmux, 200 steps on 2 GPUs), `submit.sh` (bot `8gpu` queue),
`comet_upload.sh` (ship offline Comet archives). `launchers/common.sh` puts `action_chunks`, `pylib` and
`pylib_comet` on `PYTHONPATH` and points `COMET_CONFIG` at the key file outside the repository.

## Experiment tracking

Comet ML, workspace `dont4rootme`, project `hier-action-tokenizer`. `comet.mode=auto` logs online when the API is
reachable and writes an offline archive otherwise; every metric is also mirrored into `train_log.jsonl` and
`eval_log.jsonl` inside the run directory. The API key is never stored in the repository: it comes from
`COMET_API_KEY` or `COMET_CONFIG` (`~/.comet.config` locally, `/mnt/.../afedorov/.comet.config` on the cluster).

## Tests

```bash
PYTHONPATH=.:../action_chunks pytest -q
```

Design: `docs/superpowers/specs/2026-09-11-hierarchical-action-tokenizer-design.md`.
Implementation plan: `docs/superpowers/plans/2026-09-12-hier-tokenizer-implementation.md`. Run journal: `docs/runs.md`.
