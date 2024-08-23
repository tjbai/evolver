# new new

[ ] evaluate (debug?) different ar-d training methods (ar-d vs. den)
[ ] formalize integrating substitutions
[ ] set up inference harness to consider other meta-objectives
[ ] figure out how to do controllable generation/guidance?

we don't have to beat on LL to do well!!!!

# deprecated p2

[ ] supervised training off the ground
[ ] baseline AR off the ground
[ ] tackle Evolver optimizations
[ ] try pretrained weights in Evolver
[ ] hyperparam sweep on best dataset
[ ] collect all results (baseline AR, denoising AR, supervised Evo, particle filter Evo)

# deprecated

## paper
[ ] __MCEM writeup__
[ ] research better evaluation methods

## training
[ ] __inference__ (modify particle filter)
[ ] conditional generation
[ ] validation
[ ] concatenative training

## experiments
[ ] match toy distributions
[ ] __simple sentences with tree-pruning noise__
[ ] WMT datasets

## completeness / output quality
[ ] decoding/MCEM with temperature

## efficiency
[ ] disable grad everywhere during eval
[ ] 1D/2D beam search
[ ] target-side embedding cache (possibly optional)

## misc/quality of life
[ ] __validate on-device (GPU) training__
[ ] __complete code review__
[ ] init inference from partial sequence/trajectory
[ ] move particle filter function to `run.py`