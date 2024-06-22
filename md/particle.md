Ensemble of M particles, initialize to INS(BOS), equal weight 1/M.

At each step: For each particle, get the probabilities of all 1 + C + S options.
Compute this in a batch (M, max_len, D) for D = {3, V, 5}.

# Extract probs

1. Always extract INS(x_i), one log prob.

2. Match CPY(index) against all relevant indices, S log probs.

3. Match against SUB(index, token) for ALL indices.

When we combine everything together, we can make a matrix of log probs:
rows: Vocab + None, cols: Indices + None => linearize to sample efficiently.

After sampling, use the sampled edits to seed the source of the next set of trajectories. (!!!)

# Algorithm 1: Particle Filter Sampling

```
Inputs: x\_{t-1}, x_t, threshold, M
Outputs: e_t

Procedure(EStep(x_t-1, x_t)):

    \ve\psup{1}, ..., \ve\psup{M} = {[INS(BOS)], ..., [INS(BOS)]}
    \vw = {1/M, ..., 1/M}

    for x_it in x_t:
        for e_{1:i-1} in edit_seqs:
            p = Evolver(x_{t-1}, edit_seq)
            edit_dist = restrict(edit_dist, x_it)
            pi = normalize(edit_dist)
            e_i ~ pi(e_i | e_{1:i-1}, x_it)
            edit_seq.append(edit_i)
            \vw_i /= pi(e_i)

        \vw /= \sum_{i=1}^M \vw_i
        N_eff = 1 / (\sum_{i=1}^M \vw_i^2)

        if N_eff < threshold:
            \vwe = sample(edit_seqs, weight, M)
            weights = {1/M, ..., 1/M}

    return sample(edit_seqs, weight, 1)
```

# Algorithm 2: MCEM

```
for i in N:
S = []
for trajectory in trajectories:
    samples = []
    for x_t-1, x_t in trajectory:
        sample = EStep(x_t-1, x-t)
        samples.append(sample)
    S.append(samples)
MStep(Evolver, S)
```