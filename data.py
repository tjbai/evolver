import os
import time
import math
import json
import pickle
import random
import logging

import torch
import torch.nn.functional as F

from tqdm import tqdm
from simalign import SentenceAligner
from transformers import BertTokenizer
from torch.utils.data import Dataset, Sampler, DataLoader

from const import (
    PAD_TOKEN_ID, BOS_TOKEN_ID, VOCAB_SIZE,
    INS_ID, CPY_ID, SUB_ID, EOS_ID, PAD_ID,
    OP_VERB,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REMOTE_PREFIX = os.environ.get('REMOTE_PREFIX', '/scratch4/jeisner1')
    
def pad_traj_input_ids(traj_input_ids, T):
    t, N = traj_input_ids.shape
    
    # we have to set EOS_TOKEN_ID here so that the encoder pad_mask isn't all true
    # this won't affect the decoder because we left shift everything
    pad_seq = torch.full((T-t, N), PAD_TOKEN_ID, device=traj_input_ids.device)
    pad_seq[:, 0] = BOS_TOKEN_ID
    
    return torch.cat([traj_input_ids, pad_seq])

def pad_traj_edit_tgts(traj_edit_tgts, T):
    if len(traj_edit_tgts[0].shape) == 2:
        traj_edit_tgts = tuple(map(lambda x: x.unsqueeze(0), traj_edit_tgts))

    t, N, _ = traj_edit_tgts[0].shape
    device = traj_edit_tgts[0].device
    
    return (
        torch.cat([traj_edit_tgts[0], F.one_hot(torch.full((T-t-1, N), PAD_ID, device=device), 5)], dim=0),
        torch.cat([traj_edit_tgts[1], F.one_hot(torch.full((T-t-1, N), PAD_TOKEN_ID, device=device), VOCAB_SIZE)], dim=0),
        torch.cat([traj_edit_tgts[2], F.one_hot(torch.full((T-t-1, N), 0, device=device), N)], dim=0),
    )

def get_input_ids(trajectory, max_len, tokenizer):
    if isinstance(trajectory[0], str):
        return tokenizer(
            trajectory,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids']
    
    traj_input_ids = []
    for seq in trajectory:
        n = len(seq)
        traj_input_ids.append(torch.cat([torch.tensor(seq), torch.full((max_len-n,), PAD_TOKEN_ID)]))
    return torch.stack(traj_input_ids)
    
def sum_tokens(traj_input_ids):
    return sum(torch.sum(traj[-1] != PAD_TOKEN_ID) for traj in traj_input_ids)
    
def collate_unsupervised(x):
    traj_input_ids, log_probs = zip(*x)
    toks = sum_tokens(traj_input_ids)
    T = max(traj.shape[0] for traj in traj_input_ids)
    traj_input_ids = torch.stack([pad_traj_input_ids(traj, T) for traj in traj_input_ids])
    log_probs = torch.tensor(log_probs)
    return traj_input_ids, log_probs, None, toks

def collate_supervised(x):
    traj_input_ids, traj_edit_tgts = zip(*x)
    toks = sum_tokens(traj_input_ids)
    T = max(traj.shape[0] for traj in traj_input_ids)
    traj_input_ids = torch.stack([pad_traj_input_ids(traj, T) for traj in traj_input_ids])
    traj_edit_tgts = [pad_traj_edit_tgts(tgts, T) for tgts in traj_edit_tgts]
    return traj_input_ids, None, tuple(map(lambda x: torch.stack(x), zip(*traj_edit_tgts))), toks
    
class TrajectoryDataset(Dataset):
    
    @classmethod
    def from_disk(cls, path, **kwargs):
        traj_list = []
        log_probs = []
        
        with open(path, 'r') as f:
            for line in f.readlines():
                if not line: continue
                traj, log_prob = json.loads(line)
                traj_list.append(traj)
                log_probs.append(log_prob)

        return cls(traj_list, log_probs, **kwargs)

    def __init__(self, traj_list, log_probs, max_len, tokenizer, limit=None):
        self.traj_input_ids = [get_input_ids(t, max_len, tokenizer) for t in tqdm(traj_list, desc='tokenizing trajectories')]
        self.log_probs = log_probs
        self.limit = len(self.traj_input_ids) if limit is None else limit
        self.max_len = max_len
       
    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        return self.traj_input_ids[idx], self.log_probs[idx]
    
class SupervisedTrajectoryDataset(TrajectoryDataset):

    def __init__(
        self, traj_list, log_probs, max_len, tokenizer,
        limit=None, cache_prefix=None, all_tokens=False
    ):
        super().__init__(traj_list, log_probs, max_len, tokenizer, limit)
        self.all_tokens = all_tokens
        
        aligner = SentenceAligner(
            model='bert-base-uncased',
            token_type='bpe',
            matching_methods='m',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.cache_prefix = cache_prefix
        self.cache_path = f'{REMOTE_PREFIX}/cache/' if torch.cuda.is_available() else 'cache'
        os.makedirs(self.cache_path, exist_ok=True)
        
        for i, traj in tqdm(
            enumerate(traj_list),
            desc='Computing alignments'
        ):
            path = f'{self.cache_path}/{self.cache_prefix}_{i}.zst'
            if os.path.exists(path): continue
            
            data = get_traj_edit_tgts(traj, max_len, tokenizer, aligner)
            with open(path, 'wb') as f: pickle.dump(data, f)
        
    def __getitem__(self, idx):
        with open(f'{self.cache_path}/{self.cache_prefix}_{idx}.zst', 'rb') as f:
            op_tgts, tok_tgts, idx_tgts = pickle.load(f)
            
            tok_tgts = F.one_hot(tok_tgts, VOCAB_SIZE) if not self.all_tokens else \
                       F.one_hot(self.traj_input_ids[idx], VOCAB_SIZE)[1:]
            
            return self.traj_input_ids[idx], (
                F.one_hot(op_tgts, 5),
                tok_tgts,
                F.one_hot(idx_tgts, self.max_len)
            )
            
def supervised_loader(path, max_len, tokenizer, batch_size, cache_prefix, all_tokens):
    dataset = SupervisedTrajectoryDataset.from_disk(path=path, max_len=max_len, tokenizer=tokenizer, cache_prefix=cache_prefix, all_tokens=all_tokens) 
    loader = DataLoader(dataset, batch_size=batch_size, sampler=StratifiedInfiniteSampler(dataset, batch_size), collate_fn=collate_supervised, pin_memory=True)
    return loader

def unsupervised_loader(path, max_len, tokenizer, batch_size, **_):
    dataset = TrajectoryDataset.from_disk(path=path, max_len=max_len, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=StratifiedInfiniteSampler(dataset, batch_size), collate_fn=collate_unsupervised)
    return loader
    
class SequenceDataset(Dataset):
    
    @classmethod
    def from_trajectories(cls, path, denoising=True, **kwargs):
        inputs = []
        outputs = []
        
        with open(path, 'r') as f:
            for line in f:
                traj, _ = json.loads(line)
                
                if denoising:
                    for i, input in enumerate(traj[:-1]):
                        inputs.append(input)
                        outputs.append(traj[i+1])
                        
                else:
                    inputs.append('')
                    outputs.append(traj[-1])
                    
        return cls(inputs, outputs, **kwargs)
    
    def __init__(self, inputs, outputs, max_len, tokenizer, limit=None):
        assert len(inputs) == len(outputs), 'length mismatch'
       
        s = time.time()
        logger.info('tokenizing seq2seq pairs')
        self.input_ids = get_input_ids(inputs, max_len, tokenizer)
        self.output_ids = get_input_ids(outputs, max_len, tokenizer)
        logger.info(f'done in {time.time() - s:.2f} seconds!')
    
        self.limit = len(self.input_ids) if limit is None else limit
        
    def __len__(self):
        return self.limit
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]
    
class StratifiedInfiniteSampler(Sampler):
    
    def __init__(self, source, batch_size):
        self.source = source
        self.batch_size = batch_size
        self.num_samples = len(self.source)
        self.bucket_size = math.ceil(self.num_samples / self.batch_size)
        
    def __iter__(self):
        while True:
            batch = []
            
            for i in range(self.batch_size):
                start = i * self.bucket_size
                end = min((i+1) * self.bucket_size, self.num_samples)
                if start >= end: break # edge case i'm too tired to fix
                batch.append(random.randint(start, end - 1))
                
            # random.shuffle(batch)
            yield from batch
    
    def __len__(self):
        return float('inf')
    
### logging utilities

def to_str(op, tok, idx, verbose=True):
    op = OP_VERB[op]
    if op == 'PAD' or op == 'EOS': return op
    if verbose: return f'{op}({tok}, {idx})'
    if tok and idx: return f'SUB({tok}, {idx})'
    elif tok: return f'INS({tok})'
    elif idx: return f'CPY({idx})'
    return 'UNK'    

def elaborate(traj_edit_tgts, batch_first=True):
    if len(traj_edit_tgts[0].shape) == 3: traj_edit_tgts = tuple(map(lambda x: x.unsqueeze(1), traj_edit_tgts))
    if not batch_first: traj_edit_tgts = tuple(map(lambda x: x.transpose(0, 1), traj_edit_tgts))
    B, T, max_len, _ = traj_edit_tgts[0].shape
    kernel = lambda b, t, i: to_str(*map(lambda x: torch.argmax(x[b, t, i]).item(), traj_edit_tgts))
    return [[' '.join([kernel(b, t, i) for i  in range(max_len)]) for t in range(T)] for b in range(B)]
    
### alignment utilities

def generate_alignment(s1, s2, aligner):
    if s1 == '': return
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
    yield INS_ID, BOS_TOKEN_ID, 0
    
    last = -1 
    for src, tgt in generate_alignment(s1, s2, aligner):
        
        # insert everything since last seen
        if last is not None and (tgt-last > 1):
            for missing in range(last+1, tgt):
                yield INS_ID, s2_ids[missing], 0
        
        # if aligned and equal, CPY 
        if s1_ids[src] == s2_ids[tgt]: yield CPY_ID, PAD_TOKEN_ID, src+1
        
        # if aligned but not equal, SUB
        if s1_ids[src] != s2_ids[tgt]: yield SUB_ID, s2_ids[tgt], src+1
        
        last = tgt
      
    # insert remaining values 
    if last is not None and (len(s2_ids)-last > 1):
        for missing in range(last+1, len(s2_ids)):
            yield INS_ID, s2_ids[missing], 0
         
    # always end with EOS 
    yield EOS_ID, PAD_TOKEN_ID, 0
    
def get_edit_tgts(edit_seq, max_len):
    op_tgts, tok_tgts, idx_tgts = map(lambda x: list(x), (zip(*edit_seq)))
  
    # pad and/or truncate
    if (n := len(op_tgts)) < max_len:
        op_tgts.extend([PAD_ID for _ in range(max_len-n)])
        tok_tgts.extend([PAD_TOKEN_ID for _ in range(max_len-n)])
        idx_tgts.extend([0 for _ in range(max_len-n)])
    
    return op_tgts[:max_len], tok_tgts[:max_len], idx_tgts[:max_len]
    
def get_traj_edit_tgts(trajectory, max_len, tokenizer, aligner):
    t1 = iter(trajectory)
    t2 = iter(trajectory)
    next(t2)
    
    traj_edit_tgts = ([], [], [])
    for s1, s2 in zip(t1, t2):
        edit_seq = generate_edits(s1, s2, tokenizer, aligner)
        edit_tgts = get_edit_tgts(edit_seq, max_len)
        for i in range(3): traj_edit_tgts[i].append(edit_tgts[i])
        
    return tuple(map(torch.tensor, traj_edit_tgts))

### to deprecate
    
class _EvolverDataset(Dataset):
    
    @classmethod
    def from_pickle(_, name):
        with open(f'cache/{name}-dataset.pkl', 'rb') as f:
            return pickle.load(f)
    
    def __init__(
        self,
        traj_list, max_len,
        force_targets=False, name=None,
    ):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        aligner = None
        
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

def _get_align_ids(input_ids, output_ids):
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
