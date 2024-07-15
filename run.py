import os
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data import (
    get_edit_tgts,
    get_input_ids,
    pad_traj_input_ids,
    pad_traj_edit_tgts
)

from dep import noise

from constants import (
    BOS_TOKEN_ID, PAD_TOKEN_ID, EOS_TOKEN_ID,
    PAD_ID, INS_ID, CPY_ID, SUB_ID, EOS_ID,
    INS_BOS, VOCAB_SIZE
)

logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG if ('DEBUG' in os.environ) else logging.INFO)

def sample_trajectory(
    evolver, traj_input_ids,
    num_particles, threshold=0, temperature=1, resample_at=1
):
    B, T, N = traj_input_ids.shape
    device = traj_input_ids.device
    
    traj_src, traj_pad_mask = evolver.get_src(traj_input_ids)
    src = traj_src[:, 0]
    
    traj_log_prob = torch.zeros(B, device=device)
    traj_op_tgts = torch.zeros(B, T-1, N, 5, device=device)
    traj_tok_tgts = torch.zeros(B, T-1, N, VOCAB_SIZE, device=device)
    traj_idx_tgts = torch.zeros(B, T-1, N, N, device=device)

    for i in range(T-1):
        edit_tgts, src, log_prob = particle_filter(
            evolver, traj_input_ids[:, i], traj_input_ids[:, i+1],
            src, traj_pad_mask[:, i, :], traj_pad_mask[:, i+1, :],
            num_particles, threshold, temperature, resample_at
        )
        
        traj_log_prob += log_prob
        traj_op_tgts[:, i] = edit_tgts[0]
        traj_tok_tgts[:, i] = edit_tgts[1]
        traj_idx_tgts[:, i] = edit_tgts[2]
        
    return (traj_op_tgts, traj_tok_tgts, traj_idx_tgts), traj_log_prob

def get_align_ids(input_ids, output_ids):
    B, N = input_ids.shape
    device = input_ids.device
    
    op_ids = torch.full((B, N, N*2+1), PAD_ID, dtype=torch.long, device=device)
    tok_ids = torch.full((B, N, N*2+1), PAD_TOKEN_ID, dtype=torch.long, device=device)
    idx_ids = torch.full((B, N, N*2+1), 0, dtype=torch.long, device=device)
    
    op_ids[:, :, 0] = INS_ID
    op_ids[:, :, 1:N+1] = CPY_ID
    op_ids[:, :, N+1:] = SUB_ID
    
    tok_ids[:, :, 0] = output_ids
    tok_ids[:, :, N+1:] = output_ids.unsqueeze(-1).expand(-1, -1, N)
    
    idx_range = torch.arange(N, device=device)
    idx_ids[:, :, 1:N+1] = idx_range.expand(B, N, -1)
    idx_ids[:, :, N+1:] = idx_range.expand(B, N, -1)
    
    return op_ids, tok_ids, idx_ids

def maybe_resample(weights, threshold, M):
    B = weights.shape[0]
    device = weights.device
    
    ess = 1 / torch.sum(torch.exp(weights)**2, dim=-1)
    resample_mask = ess <= threshold
    
    indices = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    samples = torch.multinomial(torch.exp(weights), M, replacement=True)
    samples = torch.where(resample_mask.unsqueeze(1), samples, indices)
    
    return samples, resample_mask

@torch.no_grad()
def particle_filter(
    evolver,
    input_ids,    # BxN
    output_ids,   # BxN
    src,          # BxNxD 
    src_pad_mask, # BxN
    tgt_pad_mask, # BxN
    M, threshold,
    temperature,
    resample_at
):
    B, N = input_ids.shape
    device = input_ids.device
    
    batch_ids = input_ids.unsqueeze(1).repeat(1, M, 1)       # BxMxN
    src = src.unsqueeze(1).repeat(1, M, 1, 1)                # BxMxNxD
    src_pad_mask = src_pad_mask.unsqueeze(1).repeat(1, M, 1) # BxMxN
    tgt_pad_mask = tgt_pad_mask.unsqueeze(1).repeat(1, M, 1) # BxMxN
   
    weights = torch.zeros(B, M, device=device)                                   # BxM
    ens_ops = torch.zeros(B, M, N, 5, dtype=torch.long, device=device)           # BxMxNx5
    ens_toks = torch.zeros(B, M, N, VOCAB_SIZE, dtype=torch.long, device=device) # BxMxNxV
    ens_idxs = torch.zeros(B, M, N, N, dtype=torch.long, device=device)          # BxMxNxN
    
    # initialize BOS
    ens_ops[:, :, 0, INS_ID] = 1
    ens_toks[:, :, 0, BOS_TOKEN_ID] = 1
    ens_idxs[:, :, 0, 0] = 1
    
    op_ids, tok_ids, idx_ids = get_align_ids(input_ids, output_ids) # BxNx(2N+1) each
   
    memory = cache = None 
    for i in range(1, N):
        forced = output_ids[:, i]
     
        # BxMxIx_ -> BMxIx_
        ens = tuple(map(
            lambda x: x.view(B*M, N, -1)[:, :i, :],
            (ens_ops, ens_toks, ens_idxs)
        ))
        
        edit_probs, _, memory, cache = evolver.forward(
            batch_ids.view(B*M, N),
            src.view(B*M, N, -1), ens,
            src_pad_mask.view(B*M, N),
            tgt_pad_mask.view(B*M, N),
            memory, cache
        )
        
        # BMxIx_ -> BxMxIx_
        op_probs, tok_probs, idx_probs = tuple(map(
            lambda x: x.view(B, M, i, -1),
            edit_probs
        ))
      
        # basically just black magic
        posterior = torch.full((B, M, N*2+1), -1e9, device=device)
        posterior[:, :, 0] = op_probs[:, :, -1, INS_ID] + tok_probs[torch.arange(B, device=device), :, -1, forced]
        cpy_mask = input_ids.eq(forced.unsqueeze(1)).unsqueeze(1).expand(B, M, N)
        posterior[:, :, 1:N+1][cpy_mask] = (op_probs[:, :, -1, CPY_ID].unsqueeze(-1) + idx_probs[:, :, -1, :])[cpy_mask]
        posterior[:, :, N+1:] = op_probs[:, :, -1, SUB_ID].unsqueeze(-1) + tok_probs[torch.arange(B), :, -1, forced].unsqueeze(-1) + idx_probs[:, :, -1, :]
       
        # normalize proposal and sample
        logits = posterior / temperature
        proposal = F.log_softmax(logits, dim=-1)
        samples = torch.multinomial(torch.exp(proposal.view(B*M, -1)), 1)
        posterior_probs = torch.gather(posterior.view(B*M, -1), dim=-1, index=samples)
        proposal_probs = torch.gather(proposal.view(B*M, -1), dim=-1, index=samples)
        
        # update particles and weights
        samples = samples.view(B, M)
        posterior_probs = posterior_probs.view(B, M)
        proposal_probs = proposal_probs.view(B, M)
        sample_weight = posterior_probs - proposal_probs
        
        # C[i, j] = one_hot(B[i, A[i, j]]])
        fn = lambda A, B, D: F.one_hot(B[torch.arange(A.size(0), device=device).unsqueeze(1).expand_as(A), A], num_classes=D)
       
        # TODO -- implement a fast(er) update 
        update = ~(forced.eq(PAD_TOKEN_ID) | forced.eq(EOS_TOKEN_ID))
        ens_ops[update, :, i, :] = fn(samples, op_ids[:, i, :], 5)[update]
        ens_toks[update, :, i, :] = fn(samples, tok_ids[:, i, :], VOCAB_SIZE)[update]
        ens_idxs[update, :, i, :] = fn(samples, idx_ids[:, i, :], N)[update]
        weights[update] += sample_weight[update]
       
        # if torch.sum(update) > 0:
        #     print('INS PROB', op_probs[:, :, -1, INS_ID], 'CPY PROB', op_probs[:, :, -1, CPY_ID])
        #     print('PROB OF THIS TOKEN', tok_probs[torch.arange(B), :, -1, forced])
        #     print('WEIGHT DIFF', posterior_probs[update], proposal_probs[update], sample_weight[update])
        
        eos = forced.eq(EOS_TOKEN_ID)
        ens_ops[eos, :, i, EOS_ID] = 1
        ens_toks[eos, :, i, PAD_TOKEN_ID] = 1
        ens_idxs[eos, :, i, 0] = 1
        weights[eos] += op_probs[eos, :, -1, EOS_ID]
        
        pad = forced.eq(PAD_TOKEN_ID)
        ens_ops[pad, :, i, PAD_ID] = 1
        ens_toks[pad, :, i, PAD_TOKEN_ID] = 1
        ens_idxs[pad, :, i, 0] = 1
        
        if M == 1: continue
        
        # normalize weights
        normalize = ~forced.eq(PAD_TOKEN_ID)
        weights[normalize] -= torch.logsumexp(weights[normalize], dim=-1, keepdim=True)
       
        # maybe resample
        if i % resample_at == 0:
            samples, resample_mask = maybe_resample(weights, threshold, M)
            ens_ops = ens_ops[torch.arange(B).unsqueeze(1), samples]
            ens_toks = ens_toks[torch.arange(B).unsqueeze(1), samples]
            ens_idxs = ens_idxs[torch.arange(B).unsqueeze(1), samples]
            weights[resample_mask] = 0
  
    # sampled edit targets for each ensemble
    samples = torch.multinomial(torch.exp(weights), 1).squeeze() if M > 1 else torch.zeros(B, dtype=torch.long)
    edit_tgts = tuple(map(lambda x: x[torch.arange(B, device=device), samples], (ens_ops, ens_toks, ens_idxs)))
    
    # next step source-side embeddings
    memory = memory.view(B, M, N, -1)[:, 0, :, :]
    src = evolver.compute_tgt(input_ids, memory, edit_tgts)
    
    return edit_tgts, src, weights[torch.arange(B, device=device), samples]

@torch.no_grad()
def fast_sample(
    evolver,
    input_ids,
    src, src_pad_mask,
    M, threshold,
    resample_at
):
    B, N = input_ids.shape
    device = input_ids.device
    
    batch_ids = input_ids.unsqueeze(1).expand(-1, M, -1).reshape(B*M, -1)
    src = src.unsqueeze(1).expand(-1, M, -1, -1).reshape(B*M, N, -1)
    src_pad_mask = src_pad_mask.unsqueeze(1).expand(-1, M, -1).reshape(B*M, -1)
    
    log_probs = torch.zeros(B*M, device=device)
    alive = torch.ones(B*M, dtype=torch.bool)
    
    ens_ops = torch.zeros(B*M, N, 5, dtype=torch.long, device=device)
    ens_toks = torch.zeros(B*M, N, VOCAB_SIZE, dtype=torch.long, device=device)
    ens_idxs = torch.zeros(B*M, N, N, dtype=torch.long, device=device)
    
    # initialize BOS
    ens_ops[:, 0, INS_ID] = 1
    ens_toks[:, 0, BOS_TOKEN_ID] = 1
    ens_idxs[:, 0, 0] = 1
    
    # initialize PAD everywhere else 
    ens_ops[:, 1:, PAD_ID] = 1
   
    memory = cache = None
    for i in range(1, N):
        
        # handle pad
        ens_ops[~alive, i, PAD_ID] = 1
        ens_toks[~alive, i, PAD_TOKEN_ID] = 1
        ens_idxs[~alive, i, 0] = 1
        
        if not torch.any(alive): break
        
        edit_probs, _, memory, cache = evolver.forward(
            batch_ids, src,
            (ens_ops[:, :i],
            ens_toks[:, :i],
            ens_idxs[:, :i]),
            src_pad_mask,
            None, memory, cache
        )
    
        op_probs, tok_probs, idx_probs = tuple(map(
            lambda x: x[:, -1],
            edit_probs
        ))
        
        ops = torch.multinomial(torch.exp(op_probs), num_samples=1).squeeze()
        toks = torch.multinomial(torch.exp(tok_probs), num_samples=1).squeeze()
        idxs = torch.multinomial(torch.exp(idx_probs), num_samples=1).squeeze()
    
        # handle eos 
        ens_ops[alive & ops.eq(EOS_ID), i, EOS_ID] = 1
        ens_toks[alive & ops.eq(EOS_ID), i, PAD_TOKEN_ID] = 1
        ens_idxs[alive & ops.eq(EOS_ID), i, 0] = 1
        
        # update the living
        alive &= ~ops.eq(EOS_ID)
        ens_ops[torch.arange(B*M, device=device)[alive], i, ops[alive]] = 1
        ens_toks[torch.arange(B*M, device=device)[alive], i, toks[alive]] = 1
        ens_idxs[torch.arange(B*M, device=device)[alive], i, idxs[alive]] = 1
    
        # update log probs 
        log_probs += op_probs[torch.arange(B*M), ops]
        log_probs[ops.eq(INS_ID) | ops.eq(SUB_ID)] += tok_probs[torch.arange(B*M), toks][ops.eq(INS_ID) | ops.eq(SUB_ID)]
        log_probs[ops.eq(CPY_ID) | ops.eq(SUB_ID)] += tok_probs[torch.arange(B*M), toks][ops.eq(CPY_ID) | ops.eq(SUB_ID)]
        
        if i % resample_at == 0:
            weights = log_probs.view(B, M) - torch.logsumexp(log_probs.view(B, M), dim=-1, keepdim=True)
            samples, _ = maybe_resample(weights, threshold, M)
            
            ens_ops, ens_toks, ens_idxs = tuple(map(
                lambda x: x.view(B, M, N, -1)[torch.arange(B, device=device).unsqueeze(1), samples].view(B*M, N, -1),
                (ens_ops, ens_toks, ens_idxs)
            ))
            
            log_probs = log_probs.view(B, M)[torch.arange(B, device=device).unsqueeze(1), samples].view(B*M)
        
    return (
       (ens_ops.view(B, M, N, -1),
        ens_toks.view(B, M, N, -1),
        ens_idxs.view(B, M, N, -1)),
        log_probs.view(B, M)
    )

def apply_edits(input_ids, edits):
    edits = tuple(map(lambda x: torch.argmax(x, dim=-1), edits)) 
    
    B = input_ids.shape[0]
    ops, toks, indices = edits
    res = torch.zeros_like(input_ids, dtype=torch.long)

    ins_mask = ops.eq(INS_ID) | ops.eq(SUB_ID)
    res[ins_mask] = toks[ins_mask]
    
    cpy_mask = ops.eq(CPY_ID)
    permuted_inputs = input_ids[torch.arange(B).view(-1, 1), indices]
    res[cpy_mask] = permuted_inputs[cpy_mask]
    
    eos_mask = ops.eq(EOS_ID)
    res[eos_mask] = EOS_TOKEN_ID
    
    return res
   
### to deprecate

def _decode_stochastic(edit_probs):
    return tuple(map(lambda x: torch.multinomial(torch.exp(x), num_samples=1).squeeze(), edit_probs)) 
    
def _decode_greedy(edit_probs):
    cands = []
    op_probs, tok_probs, index_probs = edit_probs
            
    ins_prob = op_probs[:, INS_ID]
    tok_prob, best_toks = torch.max(tok_probs, dim=-1)
    cands.append(ins_prob + tok_prob)
    
    cpy_prob = op_probs[:, CPY_ID]
    index_prob, best_indices = torch.max(index_probs, dim=-1)
    cands.append(cpy_prob + index_prob)
    
    # TODO -- SUB
    sub_prob = op_probs[:, SUB_ID]
    cands.append(torch.empty_like(cands[-1]).fill_(-1e9))
    
    cands.append(op_probs[:, EOS_ID])
    
    return (
        torch.argmax(torch.stack(cands, dim=1), dim=-1),
        best_toks,
        best_indices,
    )   
 
def _decode(
    evolver,
    src, pad_mask, edit_tgts,
    strategy='greedy' # who needs OOP
):
    B, max_len, _ = src.shape
    alive = torch.ones(B, dtype=torch.bool)
    op_tgts, tok_tgts, index_tgts = edit_tgts
    
    if strategy == 'greedy': decoder = decode_greedy
    elif strategy == 'stochastic': decoder = decode_stochastic
    else: raise NotImplementedError()
    
    for i in range(1, max_len):
        
        edit_logits, tgt = evolver.forward(src, edit_tgts)
        edit_probs = map(lambda x: x[:,i,:], evolver.get_probs(edit_logits))
        ops, toks, indices = decoder(edit_probs)
       
        # masks for decoded ops
        ins_mask = alive & (ops == INS_ID)
        cpy_mask = alive & (ops == CPY_ID)
        sub_mask = alive & (ops == SUB_ID)
        eos_mask = alive & (ops == EOS_ID) 
       
        # default is PAD, so turn off vectors that will be updated
        op_tgts[alive, i, :] = 0
        tok_tgts[None, i, :] = 0
        index_tgts[None, i, :] = 0

        raise NotImplementedError()
        
    return edit_tgts, tgt

@torch.no_grad()
def _particle_filter(
    evolver, input_ids, output_ids,
    src, src_pad_mask, tgt_pad_mask,
    M, threshold, temperature=1.0,
    device='cuda'
):
    # repeat along batch (ensemble)
    src = src.repeat(M, 1, 1)
    src_pad_mask = src_pad_mask.repeat(M, 1)
    tgt_pad_mask = tgt_pad_mask.repeat(M, 1)
    
    # initialize particles and weights 
    weights = torch.zeros(M).to(device)
    _ens = get_edit_tgts([INS_BOS], evolver.max_len)
    ens = tuple(map(lambda x: x.unsqueeze(0).repeat(M, 1, 1).to(device), _ens))
    ens_ops, ens_toks, ens_idxs = ens
   
    # precompute edit ids
    op_ids, tok_ids, idx_ids = get_align_ids(input_ids, output_ids)
    
    cache = None
    memory = None
    for i, forced in enumerate(output_ids[1:], start=1):
        # forward pass 
        edit_probs, tgt, memory, cache = evolver.forward(
            input_ids, src,
            tuple(map(lambda x: x[:, :i, :], ens)),
            src_pad_mask, tgt_pad_mask,
            memory, cache
        )
        
        # get logits 
        op_probs, tok_probs, idx_probs = edit_probs
        
        # handle EOS
        if forced == EOS_TOKEN_ID:
            eos = op_probs[:, -1, EOS_ID]
            ens_ops[:, i, :] = F.one_hot(torch.tensor(EOS_ID), 5).repeat(M, 1)
            ens_toks[:, i, :] = F.one_hot(torch.tensor(PAD_TOKEN_ID), VOCAB_SIZE).repeat(M, 1)
            ens_idxs[:, i, :] = F.one_hot(torch.tensor(0), evolver.max_len).repeat(M, 1)
            weights += eos
            if M > 1: weights -= torch.logsumexp(weights, dim=0)
            break
    
        # compute proposal weights
        ins = op_probs[:, -1, INS_ID].unsqueeze(1) + tok_probs[:, -1, forced].unsqueeze(1)
        cpy = op_probs[:, -1, CPY_ID].unsqueeze(1) + idx_probs[:, -1, input_ids.eq(forced)]
        sub = op_probs[:, -1, SUB_ID].unsqueeze(1) + tok_probs[:, -1, forced].unsqueeze(1) + idx_probs[:, -1, :]
        
        # normalize proposal and sample
        logits = torch.cat([ins, cpy, sub], dim=1)
        posterior = logits
        proposal = F.log_softmax(logits / temperature, dim=1)
        next = torch.multinomial(torch.exp(proposal), 1).squeeze(1)
        
        # update particles
        ens_ops[:, i, :] = F.one_hot(op_ids[i].to(device)[next], 5)
        ens_toks[:, i, :] = F.one_hot(tok_ids[i].to(device)[next], VOCAB_SIZE)
        ens_idxs[:, i, :] = F.one_hot(idx_ids[i].to(device)[next], evolver.max_len)
        
        # update weights
        posterior_probs = torch.gather(posterior, dim=1, index=next.unsqueeze(1)).squeeze()
        proposal_probs = torch.gather(proposal, dim=1, index=next.unsqueeze(1)).squeeze()
        
        weights += posterior_probs - proposal_probs
        if M > 1: weights -= torch.logsumexp(weights, dim=0)
        
        # maybe resample
        ess = 1 / torch.sum(torch.exp(weights)**2)
        if ess <= threshold and M > 1:
            samples = torch.multinomial(torch.exp(weights), M, replacement=True)
            ens_ops = ens_ops[samples]
            ens_toks = ens_toks[samples]
            ens_idxs = ens_idxs[samples]
            ens = (ens_ops, ens_toks, ens_idxs)
            weights = torch.zeros(M).to(device)

    sample = torch.multinomial(torch.exp(weights), 1) if M > 1 else torch.zeros(1, dtype=torch.long)
    return tuple(map(lambda x: x[sample], ens)), tgt[sample], weights[sample]

@torch.no_grad()
def _sample_batch(
    evolver, batch_ids,
    num_particles, threshold, temperature,
    device
):
    traj_input_ids = []
    traj_edit_tgts = tuple([] for _ in range(3))
    T = max(input_ids.shape[0] for input_ids in batch_ids)
    
    for input_ids in tqdm(batch_ids):
        input_ids = input_ids.to(device)
       
        cur_tgts, _ = sample_trajectory(
            evolver, input_ids,
            num_particles, threshold, temperature,
            device
        )

        input_ids = pad_traj_input_ids(input_ids, T)
        cur_tgts = pad_traj_edit_tgts(cur_tgts, T-1)
        
        traj_input_ids.append(input_ids)
        for i in range(3): traj_edit_tgts[i].append(cur_tgts[i])
    
    traj_input_ids = torch.stack(traj_input_ids).to(device)
    traj_edit_tgts = tuple(map(lambda x: torch.stack(x).to(device), traj_edit_tgts))
    
    return traj_input_ids, traj_edit_tgts

@torch.no_grad()
def _sample_trajectory(
    evolver, traj_input_ids,
    num_particles, threshold, temperature,
    device='cuda'
):
    T = len(traj_input_ids)
    
    traj_src, traj_pad_mask = evolver.get_src(traj_input_ids)
    src = traj_src[0]
  
    log_traj_prob = 0
    traj_edit_tgts = tuple([] for _ in range(3))
    
    for i in range(T-1):
        edit_tgts, src, log_prob = particle_filter(
            evolver, traj_input_ids[i], traj_input_ids[i+1],
            src, traj_pad_mask[i], traj_pad_mask[i+1],
            num_particles, threshold, temperature,
            device
        )

        # ugly, but we short-circuit the particle filter after EOS so we need this input to be padded again
        pad = torch.zeros(src.shape[0], evolver.max_len-src.shape[1], evolver.d_model, device=device)
        src = torch.cat([src, pad], dim=1)
        for i in range(3): traj_edit_tgts[i].append(edit_tgts[i])
        log_traj_prob += log_prob
    
    return tuple(map(lambda x: torch.stack(x).squeeze(), traj_edit_tgts)), log_traj_prob

def _baseline_elbo(model, tokenizer, observed, num_samples=5, device='cuda'):
    model.eval()
    samples, log_posteriors = zip(*[noise(observed) for _ in range(num_samples)])
    T = max(len(s) for s in samples)
    
    traj_input_ids = torch.stack(
        [pad_traj_input_ids(
            get_input_ids(s, model.max_len, tokenizer), T
        ).to(device) for s in samples]
    )
   
    log_likelihoods = baseline_likelihood(model, traj_input_ids) 
    tot = log_likelihoods - log_posteriors
    return torch.sum(tot) / num_samples
