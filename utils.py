import os
import json
import math
import pickle
from datetime import datetime
from const import *

import torch
from transformers import BertTokenizer
from simalign import SentenceAligner
from tqdm import tqdm

BT = BertTokenizer.from_pretrained('bert-base-uncased')
ALIGN = SentenceAligner(
    model='bert-base-uncased',
    token_type='bpe',
    matching_methods='m',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def parse_model_id(config_path):
    _, name = os.path.split(config_path)
    id = '.'.join(name.split('.')[:-1]) or name
    return id

def get_name(config_path):
    prefix = parse_model_id(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'{prefix}_{timestamp}'
    return name

def replace(t, a, b):
    return torch.where(t == a, b, t)

def log1mexp(p):
    return torch.log1p(-torch.exp(p))

def check_nan(t, name):
    if torch.isnan(t).any():
        print(f'nan detected in {name}')
        raise Exception()
    
def xent(logprobs, tgts, ignore=-1, ignore_mask=None):
    loss = torch.sum(logprobs * tgts, dim=-1)
    keep_mask = torch.argmax(tgts, dim=-1) != ignore
    if ignore_mask is not None: keep_mask &= ~ignore_mask
    loss = loss * keep_mask
    tot = torch.sum(loss)
    n = torch.sum(keep_mask)
    return -tot, n

def compute_tgt(model, input_ids, edit_tgts, mem):
    if len(input_ids.shape) == 1: input_ids = input_ids.unsqueeze(0).expand(B, -1)
    op_ids, tok_ids, idx_ids = tuple(map(lambda x: torch.argmax(x, dim=-1), edit_tgts))
    B, N = op_ids.shape
    
    tgt = torch.zeros(B, N, model.d_model, device=model.device)
    memory = model.positional_embedding(memory, d=-1)
    
    permuted_memory = memory[torch.arange(B, device=model.device).unsqueeze(1), idx_ids]
    permuted_input_ids = input_ids[torch.arange(B, device=model.device).unsqueeze(1), idx_ids]
    
    ins_mask = op_ids.eq(INS_ID)
    if torch.any(ins_mask):
        tgt[ins_mask] = model.token_embedding(tok_ids[ins_mask]) * math.sqrt(model.d_model)
        
    cpy_mask = op_ids.eq(CPY_ID)
    if torch.any(cpy_mask):
        cpy_mask = cpy_mask.unsqueeze(-1).expand_as(tgt)
        tgt[cpy_mask] = permuted_memory[cpy_mask]
    
    sub_mask = op_ids.eq(SUB_ID)
    if torch.any(sub_mask):
        old_embeds = model.token_embedding(permuted_input_ids[sub_mask]) * math.sqrt(model.d_model)
        new_embeds = model.token_embedding(tok_ids[sub_mask]) * math.sqrt(model.d_model)
        tgt[sub_mask] = permuted_memory[sub_mask] - old_embeds + new_embeds
    
    eos_mask = op_ids.eq(EOS_ID)
    if torch.any(eos_mask):
        tgt[eos_mask] = model.token_embedding.weight[model.eos_token_id] * math.sqrt(model.d_model)
    
    tgt = model.positional_embedding(tgt, d=1)
    return tgt
