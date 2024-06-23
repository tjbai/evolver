# grammar/tree-based datasets

## ud ewt:

everything neatly fits under 128 tokens, 64 would be a reasonable upper bound with truncation
try batch size 256 at first for evolver and 512 for baseline?

train:
12544 sentences, avg traj length 5.66, detokenize with
TreebankWordTokenizer, tokenize with bert-base-uncased

dev:
2001 sentences, same setup

evolver:
