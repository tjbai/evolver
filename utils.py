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
