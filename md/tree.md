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

improved particle filter does batch size 64 in ~25 seconds. basically a clean 3x speedup

ud-2.0 and 2.2: basic with different training steps
sup-ud-2.0: basic with supervision
ud-2.3: more exploration
ud-2.4: clip grads
ar-ud-2.0: decoder-only baseline
ud-3.*: halve the sequence noising, sort of like diffusion forcing

## ud-3.*

eval results look bad... inspecting sup-ud-3.0 shows pretty good adherence to the pattern, except for SUB(..., EOS) appearing in random places

not really actually. majority of the loss comes from incorrect token prediction. we get the correct operation generally. 

we observe a gradual shift in probability mass from CPY to INS as we get closer to the "break even" then a sudden phase shift

think about an approach where we take index/token loss regardless of what the operation is.
-> operation breaks ties, but the model still knows what token it should insert and where to look in the previous sequence
-> addresses hypothesis that the token loss doesn't get as low because less signal in the training data?
    (added per-occurrence plotting to verify this hypothesis, because otherwise our values are artifically deflated)

try to compute token loss for every index (even when CPY operation is supervised)
-> for some reason makes loss worse