# how to evaluate?

the dumb idea: just look at performance on downstream tasks

likelihood: we have an approximate ELBO formulation, where we just approximate
p(x, z) with a single importance-sampled edit trajectory (possibly more),
basically an MN problem.

# intro

We want to formulate a lower bound on observed data p(x) in terms of our latent
variables z. In this case, z is the trajectory of intermediate edits (and
implicitly the intermediate sequences as well).

ln p(x) >= E[ln p(x, z) - ln q(z | x)] ~ average (q(z | x) \* ln (p / q))

q(z | x) is basically our noising process => to get probability we either need
to extract log-probs from LLM or use simpler noising? For ex. if noising is a
dependency parse and there's some random selection of nodes at each step to
prune.

p(x, z) = p(z) p(x | z), provided we get to a uniform prior, we only need to
compute p(x | z). Here lies the difficulty because if z includes the EXACT
edits, then it's hard to define what q(z | x) is, unless our noising process is
edit-based as well? Otherwise, there's a marginalization problem again over all
the plausible edits?

If p(x, z) is just the full trajectory, can we approximate this? Idea: sample a
ton of _edit_ trajectories to explain p(x, z) and average over p(edit seq) p(x,
z)? This is an _unbiased_ estimator of p(x, z), right?

# rationale

p(trajectory) = sum(p(trajectory | edits) p(edits)) = sum(p(trajectory | edits)
p(edits) \* q(edits) / q(edits)) = E_q[p(trajectory, edits) / q(edits)]

Thus, sampling q ~ edits gives an unbiased estimator for p(trajectory) via
p(trajectory, edits) / q(edits) = p(trajectory | edits) p(edits) / q(edits).

Just like in the particle filter, p is unnormalized and q is normalized.

# Algorithm 3:

```
Procedure(ApproximateTrajectory(x0, ..., xN)):

    logq = 0
    logp = 0

    src = static(x_0)
    for each x_t, x_{t+1} in trajectory:
        for each prefix edit seq in x_t:

            # complete forward pass, true distribution
            edits = Evolver(x_t, prefix)

            # proposal distribution
            normalized_edits = normalize(edits)

            # sample and accumulate log numerator and denominator
            sampled = multinomial(normalized_edits)
            logq += normalized_edits(sampled)
            logp += edits(sampled)

    return logp - logq
```

# notes from evaluating topic models

ultimately, we want p(output) = p(output | revisions, edits) p(revisions, edits)
for all revisions and edits

in the ELBO scheme, we get q(z | x) for free as our noising process and then
approximate p(x, z) using some number of (importance?) samples

we want to approximate p(x, z) = p(x, z | edits) p(edits) summed over all edits

eisner: you can approximate p(x, z) using a single importance sample?

generate edits from some proposal distribution, then p(x, z | edits) then
reweight?
