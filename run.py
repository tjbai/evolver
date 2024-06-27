import os
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from data import (
    get_align_ids,
    get_edit_tgts,
    get_input_ids,
    pad_traj_input_ids,
    pad_traj_edit_tgts
)

from dep import noise

from constants import (
    PAD_TOKEN_ID, EOS_TOKEN_ID,
    INS_ID, CPY_ID, SUB_ID, EOS_ID, INS_BOS,
    VOCAB_SIZE
)

logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG if ('DEBUG' in os.environ) else logging.INFO)

@torch.no_grad()
def sample_batch(
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
def sample_trajectory(
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
        src = torch.cat([src, torch.zeros(src.shape[0], evolver.max_len-src.shape[1], evolver.d_model).to(device)], dim=1)
        for i in range(3): traj_edit_tgts[i].append(edit_tgts[i])
        log_traj_prob += log_prob
    
    return tuple(map(lambda x: torch.stack(x).squeeze(), traj_edit_tgts)), log_traj_prob

@torch.no_grad()
def particle_filter(
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
        edit_logits, tgt, memory, cache = evolver.forward(
            input_ids, src,
            tuple(map(lambda x: x[:, :i, :], ens)),
            src_pad_mask, tgt_pad_mask,
            memory, cache
        )
        
        # get logits 
        op_probs, tok_probs, idx_probs = evolver.get_probs(edit_logits, src_pad_mask)
        
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

def baseline_elbo(model, tokenizer, observed, num_samples=5, device='cuda'):
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

def baseline_likelihood(model, traj_input_ids):
    B, T, _ = traj_input_ids.shape
    traj_src, traj_pad_mask = model.get_src(traj_input_ids)
    log_probs = torch.zeros(B)
    
    for i in range(T-1):
        src = traj_src[:, i, :]
        tgt = traj_src[:, i+1, :]
        src_pad_mask = traj_pad_mask[:, i, :]
        tgt_pad_mask = traj_pad_mask[:, i+1, :]
      
        cache = None
        memory = None
        for j in range(1, model.max_len):
            forced = traj_input_ids[:, i+1, j]
            
            tok_logits, cache, memory = model.forward(
                src, tgt[:, :j, :],
                src_pad_mask, tgt_pad_mask,
                memory, cache
            )
            
            tok_probs = F.log_softmax(tok_logits, dim=-1)
            probs = torch.gather(tok_probs, dim=2, index=forced.view(1, 1, B)).squeeze()
            probs[forced.eq(PAD_TOKEN_ID)] = 0
            log_probs += probs
        
    return log_probs

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
   
def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    pass

if __name__ == '__main__':
    main()

### deprecated utilities

def decode_stochastic(edit_probs):
    return tuple(map(lambda x: torch.multinomial(torch.exp(x), num_samples=1).squeeze(), edit_probs)) 
    
def decode_greedy(edit_probs):
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
 
def decode(
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
