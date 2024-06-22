# C4 warmup

Noise with dependency parse pruning. Each leaf has a 50% chance to be removed at
each step. Downsample by 2 and keep trajectories between 3 and 8.

# why does ELBO only go down?

note to Eisner:

From what I’ve seen so far though, I think the model and training method will just require quite a bit of experimentation/hyperparam tuning. Number of parameters, learning rate, particle filter ensemble size/resampling threshold, number of ELBO samples, etc.

I've been trying to simplify the setting as much as possible to tease out possible implementation errors but haven't found anything to explain this problem yet.

1. At first, I trained the model on the same "doubling" pattern (`t -> t t -> t t t t -> t t t t t t t t`) for t=a,b,c. I also construct a simple validation set where t=x,y,z and we want to approximate the likelihood of observing the terminal sequence in each trajectory, e.g. `x x x x x x x x`. I use a simple forward noising function that just drops a token from the sequence with probability p, which makes the posterior q(z | x) easy to compute. So, to approximate the ELBO we sample some noising trajectories z_1, z_2, ... for each observation x, then we approximate p(x, z_i) with an importance reweighted sample from the particle filter subprocedure. The ELBO only went down, but I thought this was just an overfitting problem, especially because the model is strictly trained on the same pattern but might be evaluated on something like `z z z z -> z z z -> z`.

2. To simplify. I restricted the eval to only sample the same latent trajectory that the model sees in training with probability 1. So, the ELBO which is normally an average over all `p(x, z) / q(z | x)` is just equal to `p(x, z)` with the same `z` that it sees during training, so increasing the ELBO should be the same as maximizing the training objective.

To test this, I restricted the eval to _only_ sample the same latent trajectory the model sees in training with probability 1. Thus, the ELBO which is normally an average over all `p(x, z) / q(z | x)` is just equal to `p(x, z)` which is the training objective. Still, the ELBO doesn't consistently increase. I thought there might be too much noise-to-signal so I simplified it even further by altering the particle filter to argmax rather than sample from the proposal.

The reasoning being that at each training step, the model will backpropagate over some latent edit trajectory z. In the following eval step, we particle filter again to get some z' and approximate the ELBO, but that z' _should_ be the same as the z we just backpropagated over, so the ELBO should strictly increase. Basically, I wanted to force the model to converge to a single latent trajectory as fast as possible and see if the likelihood of that trajectory increases.

I was still surprised to see the ELBO would increase 

RAT tensor(0.0922, grad_fn=<DivBackward0>)
RAT tensor(0.7079, grad_fn=<DivBackward0>)

RAT tensor(0.0844, grad_fn=<DivBackward0>)
RAT tensor(0.8948, grad_fn=<DivBackward0>)
RAT tensor(0.9092, grad_fn=<DivBackward0>)
RAT tensor(0.9091, grad_fn=<DivBackward0>)

RAT tensor(0.0917, grad_fn=<DivBackward0>)
RAT tensor(0.9172, grad_fn=<DivBackward0>)
RAT tensor(0.9289, grad_fn=<DivBackward0>)
RAT tensor(0.9293, grad_fn=<DivBackward0>)
RAT tensor(0.9251, grad_fn=<DivBackward0>)
RAT tensor(0.9211, grad_fn=<DivBackward0>)
RAT tensor(0.9207, grad_fn=<DivBackward0>)
RAT tensor(0.9230, grad_fn=<DivBackward0>)

INFO:train:
INS(101) CPY(1) CPY(1) EOS PAD PAD PAD PAD PAD PAD
INS(101) CPY(1) CPY(2) CPY(2) CPY(2) EOS PAD PAD PAD PAD
INS(101) CPY(1) CPY(2) CPY(2) CPY(2) CPY(2) CPY(2) CPY(2) CPY(2) EOS
INFO:train:eval loss: -1.6871118545532227

RAT tensor(0.0840, grad_fn=<DivBackward0>)
RAT tensor(0.5247, grad_fn=<DivBackward0>)

RAT tensor(0.0764, grad_fn=<DivBackward0>)
RAT tensor(0.7394, grad_fn=<DivBackward0>)
RAT tensor(0.7563, grad_fn=<DivBackward0>)
RAT tensor(0.7440, grad_fn=<DivBackward0>)

RAT tensor(0.0794, grad_fn=<DivBackward0>)
RAT tensor(0.7814, grad_fn=<DivBackward0>)
RAT tensor(0.7957, grad_fn=<DivBackward0>)
RAT tensor(0.7853, grad_fn=<DivBackward0>)
RAT tensor(0.7662, grad_fn=<DivBackward0>)
RAT tensor(0.7527, grad_fn=<DivBackward0>)
RAT tensor(0.7523, grad_fn=<DivBackward0>)
RAT tensor(0.7601, grad_fn=<DivBackward0>)

INFO:train:
INS(101) CPY(1) CPY(1) EOS PAD PAD PAD PAD PAD PAD
INS(101) CPY(1) CPY(2) CPY(2) CPY(2) EOS PAD PAD PAD PAD
INS(101) CPY(1) CPY(2) CPY(2) CPY(2) CPY(2) CPY(2) CPY(2) CPY(2) EOS
INFO:train:eval loss: -1.6077560186386108