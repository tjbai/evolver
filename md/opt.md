# optimization checklist

## priority (major bottleneck)
- [ ] batched particle filter
- [ ] gradient checkpointing
- [ ] length-based sampler

## low hanging fruit
- [ ] async dataloader
- [ ] look for fused kernel opportunities
- [ ] refactor TransformerEncoder for torch.compile compatibility
- [ ] remove all CPU/GPU synchronizations and copies
- [ ] lower/mixed precision training
- [ ] preallocate memory for heterogeneous trajectories

## misc
- [ ] sequence and/or trajectory packing (possibly very hard)