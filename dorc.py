import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import load_config
from embed import SinusoidalEmbedding
from trans import (
    TransformerEncoderLayer,
    TransformerEncoder,
    CausalTransformerDecoderLayer,
    CausalTransformerDecoder,
    MultiheadPointer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_wandb(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f'step {step}: {data}')

def DaggerDataset(Dataset):
    
    def __init__(self, compact_to):
        self.samples = []
        self.compact_to = compact_to
        
    def union(self, new_samples):
        self.samples.extend(new_samples)
        # TODO -- figure out compaction strategy
        
class Evolver(nn.Module):
    pass
        
class Teacher(nn.Module):
    
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        max_len,
        d_model=512,
        dim_feedforward=2048,
        nhead=8,
        dropout=0,
        layer_norm_eps=1e-5,
        encoder_layers=6,
        decoder_layers=6):
        
        super().__init__()
        
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        t_params = {
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps}
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len+1)
       
        encoder_layer = TransformerEncoderLayer(**t_params)
        decoder_layer = CausalTransformerDecoderLayer(**t_params)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = CausalTransformerDecoder(decoder_layer, decoder_layers)
        
        self.tok_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, batch, cache=None):
        src_ids = batch['src_ids']
        input_ids = batch['input_ids']
        tgt_ids = batch['tgt_ids']
        
        src = torch.cat([self.embed(src_ids), self.embed(input_ids)], dim=1)
        pad_mask = torch.cat([src_ids, input_ids], dim=1).eq(self.pad_token_id)
        tgt = self.embed(tgt_ids)
        
        mem = self.encoder(src, src_key_padding_mask=pad_mask)
        h, _, cache = self.decoder(tgt, mem, memory_key_padding_mask=pad_mask, cache=cache)
        tok_probs = F.log_softmax(self.tok_head(h), dim=-1)
        
        return tok_probs, cache
    
    @torch.no_grad()
    def generate(self, batch, temp=1.0, **_):
        src_ids = batch['src_ids']
        
        B = src_ids.shape[0]
        device = src_ids.device
        self.decoder.set_causal()
        
        tgt_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        cache = None
        for _ in range(self.max_len):
            logits, cache = self.forward({'src_ids': src_ids.to(device), 'tgt_ids': tgt_ids.to(device)}, cache=cache)
            next_tok_logits = logits[:, -1, :] / temp
            next_tok = torch.multinomial(F.softmax(next_tok_logits, dim=-1), num_samples=1)
            tgt_ids = torch.cat([tgt_ids, next_tok], dim=-1)
            finished |= (next_tok.squeeze(-1) == self.eos_token_id)
            if finished.all(): break
        
        self.decoder.set_parallel()
        return tgt_ids
        
    def rollout(self, T):
        traj = []
        
        pass

def save_checkpoint(model, optimizer, step, config):
    save_path = os.path.join(config['checkpoint_dir'], f'{model.name}_{step}.pt')
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)

def train(config):
    device = torch.device(config['device'])
    
    teacher = None
    student = None
    optim = None
    
    dataset = DaggerDataset(compact_to=config['compact_to'])
    dataset.aggregate(sum((teacher.rollout(config['rollout_steps']) for _ in range(config['init_num_rollouts'])), []))
    
    global_step = 0
    for dagger_step in range(config['dagger_steps']):

        loader = DataLoader(dataset, batch_size=config['student_batch_size'], collate_fn=dataset.collate_fn, num_workers=config['num_workers'])
        
        for step, batch in tqdm(enumerate(loader)):
            batch = {k: v.to(device) for k, v in batch}
            
            op_loss, tok_loss, idx_loss = student.step(batch)
            (op_loss + tok_loss + idx_loss).backward()
            
            if (step + 1) % config['grad_accum_steps'] == 0:
                optim.step()
                optim.zero_grad()

            if (step + 1) % config['log_every'] == 0:
                log_to_wandb({
                    'train/op_loss': op_loss,
                    'train/tok_loss': tok_loss,
                    'train/idx_loss': idx_loss,},
                    step=global_step)

            if (step + 1) % config['save_every'] == 0:
                save_checkpoint(student, optim, global_step, config)
                
            global_step += 1
        
        logger.info(f'finished dagger step {dagger_step}')
        
        # step 1: rollout samples from the student
        student_rollout = sum((student.rollout(config['rollout_steps']) for _ in range(config['num_rollouts'])), [])
        student_dataset = DataLoader(dataset, batch_size=config['teacher_batch_size'], num_workers=config['num_workers'])
        
        # step 2: run the teacher model on visited states
        new_dataset = []
        for batch in student_dataset:
            new_states = batch['state']
            new_actions = teacher.generate(batch)
            new_dataset.extend(zip(new_states, new_actions))

        # step 3: do the dagger thing
        dataset.aggregate(new_dataset)

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--config')
    return parser.parse_args()

def main():
    args = parse_args() 
    config = load_config(args.config)

if __name__ == '__main__':
    main()
