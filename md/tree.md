# grammar/tree-based datasets

## ud ewt:

everything neatly fits under 128 tokens, 64 would be
a reasonable upper bound with truncation

train:
12544 sentences, avg traj length 5.66, weight 0.1, detokenize with
TreebankWordTokenizer, tokenize with bert-base-uncased

dev:
2001 sentences, same setup

1.0.0:
```
{
    "d_model": 512,
    "nhead": 8,
    "max_len": 64,
    "encoder_layers": 6,
    "decoder_layers": 6,

    "lr": 3e-4,
    "epochs": 10,
    "bsz": 128,
    "grad_accum_steps": 1,
    "checkpoint_at": 5,
    "eval_at": 1,
    "eval_limit": 1000,

    "num_particles": 5,
    "threshold": 2,
    "temperature": 1,
    "eval_samples": 1
}
```

we run out of memory after sampling just 19 trajectories, so try batch size of 16?