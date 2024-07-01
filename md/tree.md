# grammar/tree-based datasets

## ud ewt:

everything neatly fits under 128 tokens, 64 would be a reasonable upper bound with truncation

train:
12544 sentences, avg traj length 5.66, weight 0.1, detokenize wit TreebankWordTokenizer, tokenize with bert-base-uncased

dev:
2001 sentences, same setup

we run out of memory after sampling just 19 trajectories, so try batch size of 16?

fixed memory leak, we can now do batch size 64 but not consistently (some batches might have longer trajectories)

batch size 32 with 2 accumulation steps seems stable. majority of the loss comes from token (unsurprisingly) -> gives some more credit to the pretraining idea because that would mostly benefit the token head

on GPU takes between 1-1.5 seconds for each particle filter step. by far the largest bottleneck.

## ud gum:

train: ~9000
eval: ~1000

use context length 64 again

1.0.*: ud_ewt
2.0.*: ud_gum, sample POS noising
2.1.*: ud_gum, deterministic noising

evolver is around 3000 (non-pad) tokens for batch size 32, so a single example is on average ~100 tokens
AR denoising is around 15 tokens per example, on average
100 / 15 ~= 7 => roughly corresponds to trajectory multiple

ar_bsz * X = evo_bsz * traj * Y => Y / X = ar_bsz / (evo_bsz * traj)

improved particle filter does batch size 64 in ~25 seconds. basically a clean 3x speedup.