import os
import random
import argparse
import logging
from itertools import islice
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_config
from mt import TrajectoryDataset, Evolver, Teacher, SpacyTokenizer, init_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_wandb(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f'step {step}: {data}')
    
class DaggerDataset(TrajectoryDataset):

    def __init__(self, split, max_len, buffer_size, tokenizer, compact_to, starting_size):
        # TODO -- sample from parent until we can fill up starting_size samples 
        
        super().__init__(split, max_len, buffer_size, tokenizer)
        self.compact_to = compact_to
        self.additional_samples = []

    def union(self, new_samples):
        self.additional_samples.extend(new_samples)
        if self.compact_to and len(self.additional_samples) > self.compact_to:
            self.additional_samples = random.sample(self.additional_samples, self.compact_to)

    def __getitem__(self, idx):
        if idx < len(self.buffer): return super().__getitem__(idx)
        else: return self.additional_samples[idx - len(self.buffer)]

    def collate_fn(self, batch):
        return super().collate_fn(batch)

def save_checkpoint(model, optimizer, step, config):
    save_path = os.path.join(config['checkpoint_dir'], f'{model.name}_{step}.pt')
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
   
# should operate on a batch
# don't bother stacking along batch dim, just map along each pair of examples

def get_edits():
    
    # PROBLEM -- mismatch between bert and spacy tokenizers
    # can do a little dp here to align things
    
    def aux(input_ids, tgt_ids):
        pass
    
    pass

def train(config):
    device = torch.device(config['device'])
    
    tokenizer = SpacyTokenizer()
    student = init_model(config, tokenizer)
    teacher = init_model(config['teacher_config'], tokenizer)
    optim = AdamW(student.parameters(), lr=config['lr'])
    
    dataset = DaggerDataset(
        split='train',
        max_len=config['max_len'],
        tokenizer=tokenizer,
        compact_to=config['compact_to'],
        starting_size=config['starting_size'])
    
    global_step = 0
    for _ in range(config['dagger_steps']):

        loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=dataset.collate_fn, num_workers=config['num_workers'])
        
        for step, batch in tqdm(enumerate(loader), desc='training over current dataset'):
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
        
        # step 1: generate rollouts from student
        visited = []
        for batch in tqdm(enumerate(islice(loader, config['rollout_steps'])), desc='rolling out student'):
            for t in student.rollout(batch, T=config['rollout_depth'])[1:-1]:
                visited.extend(
                    {'src_ids': batch['src_ids'][i], 'input_ids': t[i]}
                    for i in range(config['batch_size']))
                
        visited_dataset = DataLoader(
            visited,
            batch_size=config['teacher_config']['batch_size'],
            num_workers=config['num_workers'],
            collate_fn=dataset.collate_fn_reduced)
    
        # step 2: retrieve labels from teacher 
        new_dataset = []
        for batch in tqdm(visited_dataset, desc='generating teacher labels'):
            tgt_ids = teacher._generate(batch)
            
            # TODO -- do this!!!
            edit_ids = get_edits(batch['input_ids'], tgt_ids)
            
            new_dataset.extend(
                {'src_ids': batch['src_ids'][i], 'input_ids': batch['input_ids'][i], 'tgt_ids': tgt_ids[i], 'edit_ids': edit_ids[i],}
                for i in range(config['teacher_config']['batch_size']))
        
        # step 3: do the dagger thing
        dataset.aggregate(new_dataset)

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--config')
    parser.add_argument('--teacher-config')
    return parser.parse_args()

def main():
    args = parse_args() 
    config = load_config(args.config)
    config['teacher_config'] = load_config(args.teacher_config)
    config['device'] = args.device
    config['local'] = args.local
    config['name'] = f"dag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not config['local']: wandb.init(project='mt-evolver', name=config['name'], config=config, resume='allow')
    train(config)

if __name__ == '__main__':
    main()
