import json
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm
from simalign import SentenceAligner
from transformers import BertTokenizer

from dep import noise_forward

from constants import (
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, VOCAB_SIZE,
    INS_ID, CPY_ID, SUB_ID, EOS_ID, PAD_ID,
    OP_VERB,
)

def get_align_ids(input_ids, output_ids):
    N = input_ids.shape[0]
    op_ids, tok_ids, idx_ids = ([], [], [])

    for forced in output_ids:
        C = torch.sum(input_ids.eq(forced))
        
        op_ids.append(torch.cat([
            torch.tensor([INS_ID]),
            torch.tensor([CPY_ID]).repeat(C),
            torch.tensor([SUB_ID]).repeat(N)
        ]))
        
        tok_ids.append(torch.cat([
            torch.tensor([forced]),
            torch.tensor([PAD_TOKEN_ID]).repeat(C),
            torch.tensor([forced]).repeat(N)
        ]))
        
        idx_ids.append(torch.cat([
            torch.tensor([0]),
            torch.arange(N)[input_ids.eq(forced).to('cpu')],
            torch.arange(N)
        ]))
        
    return op_ids, tok_ids, idx_ids

def get_edit_tgts(edit_seq, max_len):
    
    edit_seq = [
        (op, PAD_TOKEN_ID if tok is None else tok, 0 if idx is None else idx)
        for (op, tok, idx) in edit_seq
    ]
    op_tgts, tok_tgts, idx_tgts = map(lambda x: list(x), (zip(*edit_seq)))
  
    # pad and/or truncate
    if (n := len(op_tgts)) < max_len:
        op_tgts.extend([PAD_ID for _ in range(max_len-n)])
        tok_tgts.extend([PAD_TOKEN_ID for _ in range(max_len-n)])
        idx_tgts.extend([0 for _ in range(max_len-n)])
        
    return (
        F.one_hot(torch.tensor(op_tgts[:max_len]), 5),
        F.one_hot(torch.tensor(tok_tgts[:max_len]), VOCAB_SIZE),
        F.one_hot(torch.tensor(idx_tgts[:max_len]), max_len)
    )
    
def pad_traj_input_ids(traj_input_ids, T):
    t, max_len = traj_input_ids.shape
    
    # we have to set EOS_TOKEN_ID here so that the encoder pad_mask isn't ALL true
    # as a result, we avoid any nan values while preserving the integrity of the decoder (?)
    # the rationale is that during inference we never see an EOS token starting the sequence
    pad_seq = torch.tensor([PAD_TOKEN_ID]).repeat(max_len).to(traj_input_ids.device)
    pad_seq[0] = EOS_TOKEN_ID
    
    return torch.cat([traj_input_ids, pad_seq.repeat(T-t, 1)])

def pad_traj_edit_tgts(traj_edit_tgts, T):
    
    # covers edge case where trajectory is length 2
    if len(traj_edit_tgts[0].shape) == 2:
        traj_edit_tgts = tuple(lambda x: x.unsqueeze(0), traj_edit_tgts)
        
    t, max_len, _ = traj_edit_tgts[0].shape
    device = traj_edit_tgts[0].device
    
    return (
        torch.cat([traj_edit_tgts[0], F.one_hot(torch.tensor(PAD_ID), 5).repeat(T-t, max_len, 1).to(device)]),
        torch.cat([traj_edit_tgts[1], F.one_hot(torch.tensor(PAD_TOKEN_ID), VOCAB_SIZE).repeat(T-t, max_len, 1).to(device)]),
        torch.cat([traj_edit_tgts[2], F.one_hot(torch.tensor(0), max_len).repeat(T-t, max_len, 1).to(device)]),
    )

def get_input_ids(trajectory, max_len, tokenizer):
    return tokenizer(
        trajectory,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )['input_ids']
    
class TrainLoader:
    
    @classmethod
    def from_disk(cls, path, **kwargs):
        with open(path, 'r') as f:
            traj_list = [json.loads(line) for line in f.readlines() if line]
            
        return cls(traj_list, **kwargs)
    
    def __init__(self, traj_list, bsz, max_len, tokenizer):
        self.bsz = bsz 
        self.num_batches = (len(traj_list) + bsz - 1) // bsz
        self.traj_input_ids = [
            get_input_ids(t, max_len, tokenizer)
            for t in tqdm(traj_list, desc='Tokenizing inputs')
        ]
       
    def __iter__(self):
        random.shuffle(self.traj_input_ids)
        self.cur = 0
        return self
    
    def __next__(self):
        if self.cur >= self.num_batches: raise StopIteration()
        start = self.cur * self.bsz
        end = min((self.cur + 1) * self.bsz, len(self.traj_input_ids))
        
        # doesn't automatically pad, but we normally use bsz=1 anwyays
        # it's more natural for batching to be done through grad accumulation too...
        batch = torch.stack(self.traj_input_ids[start:end]).to(self.device)
        self.cur += 1
        return batch
    
    def to(self, device):
        self.device = device
        return self
    
class EvalLoader:
    
    @classmethod
    def from_disk(cls, path, **kwargs):
        with open(path, 'r') as f:
            observed_list = [line.strip() for line in f.readlines() if line]
            
        return cls(observed_list, **kwargs)
    
    def __init__(self, observed_list, num_samples, max_len, tokenizer):
        self.traj_input_ids = []
        self.log_probs = []
        
        traj_list = []
        for observed in tqdm(observed_list, desc='Noising observations'):
            for _ in range(num_samples):
                traj, log_prob = noise_forward(observed)
                traj_list.append(traj)
                self.log_probs.append(log_prob)
        
        self.traj_input_ids = [get_input_ids(traj, max_len, tokenizer) for traj in traj_list]
        self.num_samples = num_samples
    
    def __iter__(self):
        self.cur = 0
        return self
    
    def __next__(self):
        if self.cur >= len(self.traj_input_ids): raise StopIteration()
        traj_input_ids = self.traj_input_ids[self.cur].to(self.device)
        log_prob = self.log_probs[self.cur]
        self.cur += 1
        return traj_input_ids, log_prob
    
    def to(self, device):
        self.device = device
        return self
    
### logging utilities

def to_verbose_str(op, tok, idx, prev_toks, tokenizer):
    op = OP_VERB[op]
    if op == 'PAD' or op == 'EOS': edit = op
    elif tok and idx: edit = f'SUB({tok}, {prev_toks[idx]})'
    elif tok: edit = f'INS({tokenizer.decode(tok)})'
    elif idx: edit = f'CPY({prev_toks[idx]})'
    return edit

def to_str(op, tok, idx, prev_toks=None, tokenizer=None):
    op = OP_VERB[op]
    if op == 'PAD' or op == 'EOS': return op
    
    if idx: idx_str = idx if prev_toks is None else prev_toks[idx-1]
    if tok: tok_str = tok if tokenizer is None else ''.join(tokenizer.decode(tok).split())
    
    if tok and idx: return f'SUB({tok_str}, {idx_str})'
    elif tok: return f'INS({tok_str})'
    elif idx: return f'CPY({idx_str})'

def elaborate(traj_edit_tgts, batch_first=True):
    if len(traj_edit_tgts[0].shape) == 3: traj_edit_tgts = tuple(map(lambda x: x.unsqueeze(1), traj_edit_tgts))
    if not batch_first: traj_edit_tgts = tuple(map(lambda x: x.transpose(0, 1), traj_edit_tgts))
    B, T, max_len, _ = traj_edit_tgts[0].shape
    kernel = lambda b, t, i: to_str(*map(lambda x: torch.argmax(x[b, t, i]).item(), traj_edit_tgts))
    return [[' '.join([kernel(b, t, i) for i  in range(max_len)]) for t in range(T)] for b in range(B)]
    
### alignment utilities

def generate_alignment(s1, s2, aligner):
    _alignment = aligner.get_word_aligns(s1, s2)['mwmf']
    _alignment.sort(key=lambda x: x[1])
    seen = set()
    for src, tgt in _alignment:
        if tgt in seen: continue
        seen.add(tgt)
        yield src, tgt

def generate_edits(s1, s2, tokenizer, aligner):
    s1_ids = tokenizer.encode(s1.lower())[1:-1]
    s2_ids = tokenizer.encode(s2.lower())[1:-1]
  
    # always start with BOS
    yield INS_ID, BOS_TOKEN_ID, None
    
    last = -1 
    for src, tgt in generate_alignment(s1, s2, aligner):
        
        # insert everything since last seen
        if last is not None and (tgt-last > 1):
            for missing in range(last+1, tgt):
                yield INS_ID, s2_ids[missing], None
        
        # if aligned and equal, CPY 
        if s1_ids[src] == s2_ids[tgt]: yield CPY_ID, None, src+1
        
        # if aligned but not equal, SUB
        if s1_ids[src] != s2_ids[tgt]: yield SUB_ID, s2_ids[tgt], src+1
        
        last = tgt
      
    # insert remaining values 
    if last is not None and (len(s2_ids)-last > 1):
        for missing in range(last+1, len(s2_ids)):
            yield INS_ID, s2_ids[missing], None
         
    # always end with EOS 
    yield EOS_ID, None, None
    
def get_traj_edit_tgts(trajectory, max_len, tokenizer, aligner):
    traj_edit_tgts = ([], [], [])
    traj_op_tgts, traj_tok_tgts, traj_idx_tgts = traj_edit_tgts

    t1 = iter(trajectory)
    t2 = iter(trajectory)
    next(t2)
    
    for s1, s2 in zip(t1, t2):
        edit_seq = generate_edits(s1, s2, tokenizer, aligner)
        op_tgts, tok_tgts, idx_tgts = get_edit_tgts(edit_seq, max_len)
        traj_op_tgts.append(op_tgts)
        traj_tok_tgts.append(tok_tgts)
        traj_idx_tgts.append(idx_tgts)
        
    return tuple(map(torch.stack, traj_edit_tgts))
    
class EvolverDataset(Dataset):
    
    @classmethod
    def from_pickle(_, name):
        with open(f'cache/{name}-dataset.pkl', 'rb') as f:
            return pickle.load(f)
    
    def __init__(
        self,
        traj_list, max_len,
        force_targets=False, name=None,
    ):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        aligner = None # SentenceAligner(model='bert', token_type='bpe', matching_methods='m')
        
        self.traj_input_ids = [
            get_input_ids(t, max_len, tokenizer)
            for t in tqdm(traj_list, desc='Tokenizing inputs')
        ]
        
        self.traj_edit_tgts = [
            (get_traj_edit_tgts(t, max_len, tokenizer, aligner) if force_targets else None)
            for t in tqdm(traj_list, desc='Computing edits')
        ]
       
        if not name: return
        with open(f'cache/{name}-dataset.pkl', 'wb') as f:
            pickle.dump(self, f)
            
    def empty_traj_edit_tgts(self):
        self.traj_edit_tgts = []
        
    def add_traj_tgt(self, traj_tgt):
        self.traj_edit_tgts.append(traj_tgt) 
            
    def __getitem__(self, i):
        return (self.traj_input_ids[i], self.traj_edit_tgts[i])
    
    def __len__(self):
        return len(self.traj_input_ids)
