import os
import json
import pickle
from datetime import datetime
from const import *

import torch
from data import get_input_ids
from transformers import BertTokenizer
from tqdm import tqdm

BT = BertTokenizer.from_pretrained('bert-base-uncased')

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

def generate_prefix_alignment(input_path, output_dir, cache_prefix, max_len=512):
    with open(input_path, 'r') as f:
        for i, line in tqdm(enumerate(f.readlines())):
            if not line.strip(): continue
            traj, _ = json.loads(line)
            ids = get_input_ids(traj, max_len=max_len, tokenizer=BT)
            
            traj_op_tgts = []
            traj_tok_tgts = []
            traj_idx_tgts = []
            
            for j, seq in enumerate(ids[1:], start=1):
                N = len(ids[j-1]) - 2
                M = len(seq) - 2 - N
                
                op_tgts = [INS_ID] + [CPY_ID for _ in range(N)] + [INS_ID for _ in range(M)] + [EOS_ID]
                tok_tgts = [BOS_TOKEN_ID] + [PAD_TOKEN_ID for _ in range(N)] + [tok for tok in seq[1+N:1+N+M]] + [PAD_TOKEN_ID]
                idx_tgts = [0]  + [i for i in range(1, N+1)] + [0 for _ in range(M)] + [0]
            
                op_tgts += [PAD_ID for _ in range(max_len-len(seq))]
                tok_tgts += [PAD_TOKEN_ID for _ in range(max_len-len(seq))]
                idx_tgts += [0 for _ in range(max_len-len(seq))]
                
                traj_op_tgts.append(op_tgts)
                traj_tok_tgts.append(tok_tgts)
                traj_idx_tgts.append(idx_tgts)
                
            traj_op_tgts = torch.tensor(traj_op_tgts)
            traj_tok_tgts = torch.tensor(traj_tok_tgts)
            traj_idx_tgts = torch.tensor(traj_idx_tgts)
            
            with open(f'{output_dir}/{cache_prefix}_{i}.zst', 'wb') as f:
                pickle.dump((traj_op_tgts, traj_tok_tgts, traj_idx_tgts), f)
