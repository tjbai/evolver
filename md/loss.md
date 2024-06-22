# how to report/normalize loss correctly

we want to report per-token loss (op_loss + tok_loss + idx_loss)

at the sub-trajectory level, we already normalize this.

we then sum over...
- the entire trajectory
- the entire batch
- the entire epoch

we can eliminate summing over the entire epoch by reporting per-batch loss

we can eliminate summing over the entire batch by reporting 