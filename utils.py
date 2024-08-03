import os
from datetime import datetime

import torch

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
