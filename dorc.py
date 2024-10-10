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
    
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
    def generate(self, batch):
        input_ids = batch['input_ids']
        
    def rollout(self, T):
        pass
   
def rollout(model, N, T):
    # returns dict of {state, action}
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