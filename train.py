import os
import time
import json
import logging
import argparse
from datetime import datetime

import wandb
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from constants import PAD_TOKEN_ID
from model import Evolver, Transformer
from run import sample_trajectory
from data import (
    elaborate,
    collate_supervised,
    collate_unsupervised,
    TrajectoryDataset,
    SupervisedTrajectoryDataset,
    Seq2SeqDataset,
    StratifiedInfiniteSampler
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
   
def log_edits(traj_edit_tgts):
    logger.info(
        '\n' + 
        '\n'.join(
            ' '.join([e.ljust(8) for e in edit_tgts])
            for edit_tgts in elaborate(traj_edit_tgts)
        ))
    
def get_memory():
    return torch.cuda.memory_allocated() / 1024**2
    
def log(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: print(f"Step {step}: {data}")

def train_evolver(
    evolver, optim, lr_scheduler, train_loader, eval_loader,
    train_steps, eval_steps, grad_accum_steps, checkpoint_at, eval_at,
    num_particles=None, threshold=None, temperature=1.0, resample_at=1,
    device='cuda', name='test', start_step=0
):
    for step, (traj_input_ids, _, traj_edit_tgts) in tqdm(
        enumerate(train_loader, start=start_step),
        total=train_steps
    ):
        if step >= train_steps: break
        
        traj_input_ids = traj_input_ids.to(device)
        
        # E-step
        evolver.eval() 
        if traj_edit_tgts is not None:
            traj_edit_tgts = tuple(map(
                lambda x: x.to(device),
                traj_edit_tgts
            ))
        else:
            s = time.time()
            traj_edit_tgts, _ = sample_trajectory(
                evolver, traj_input_ids,
                num_particles, threshold, temperature, resample_at
            )
            log({'train/e_time': time.time()-s}, step=step)
        
        # M-step
        s = time.time()
        evolver.train()
        traj_loss, op_loss, tok_loss, idx_loss = \
            evolver.traj_loss(traj_input_ids, traj_edit_tgts)
            
        traj_loss.backward()
        if step % grad_accum_steps == 0:
            optim.step()
            optim.zero_grad()
            
        if lr_scheduler:
            lr_scheduler.step()
        
        log({
            'train/total_loss': op_loss + tok_loss + idx_loss,
            'train/op_loss': op_loss,
            'train/tok_loss': tok_loss,
            'train/idx_loss': idx_loss,
            'train/m_time': time.time()-s
        }, step=step)
        
        if (step + 1) % checkpoint_at == 0:
            save_path = f'/scratch4/jeisner1/checkpoints' if device == 'cuda' else 'checkpoints'
            torch.save({
                'step': step,
                'model': evolver.state_dict(),
                'optim': optim.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'wandb_run_id': wandb.run.id
            }, f'{save_path}/{name}-{step+1}.pt')
            
        if (step + 1) % eval_at == 0:
            s = time.time()
            eval_loss = evaluate_evolver(evolver, eval_loader, eval_steps, device)
            log({'eval/loss': eval_loss, 'eval/time': time.time()-s}, step=step)

@torch.no_grad()
def evaluate_evolver(evolver, eval_loader, eval_steps, device):
    evolver.eval()
    tot_loss = 0
    tot_n = 0
    
    for step, (traj_input_ids, log_posterior, _) in enumerate(eval_loader):
        if step >= eval_steps: break
        
        traj_input_ids = traj_input_ids.to(device)
        log_posterior = log_posterior.to(device)
        
        _, log_likelihood = sample_trajectory(
            evolver, traj_input_ids,
            num_particles=1, threshold=0, temperature=0.5, resample_at=1e9
        )
        
        tot_loss += torch.sum(log_likelihood - log_posterior) 
        tot_n += torch.sum(traj_input_ids[:, -1] != PAD_TOKEN_ID)
       
    return tot_loss / tot_n

def train_ar(
    model, optim, lr_scheduler, train_loader, eval_loader,
    train_steps, grad_accum_steps, checkpoint_at, eval_at,
    device, name, start_step
):
    for step, (input_ids, output_ids) in enumerate(
        train_loader,
        start=start_step
    ):
        if step == train_steps: break

        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)
       
        model.train() 
        tot_loss, n = model.loss(input_ids, output_ids)
        loss = tot_loss / n 
        loss.backward()
        
        if (step + 1) % grad_accum_steps == 0:
            optim.step()
            optim.zero_grad()
           
        if lr_scheduler: 
            lr_scheduler.step()
       
        log({'train/loss': loss.item()}, step=step)
        
        if (step + 1) % checkpoint_at == 0:
            save_path = f'/scratch4/jeisner1/checkpoints' if device == 'cuda' else 'checkpoints'
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'wandb_run_id': wandb.run.id
            }, f'{save_path}/{name}-{step+1}.pt')
            
        if (step + 1) % eval_at == 0:
            model.eval()
            tot_loss = 0
            tot_n = 0
            
            with torch.no_grad():
                for input_ids, output_ids in eval_loader:
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                    loss, n = model.loss(input_ids, output_ids)
                    tot_loss += loss
                    tot_n += n
                    
            eval_loss = tot_loss / tot_n
            log({'eval/loss': eval_loss.item()}, step=step)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--from-checkpoint', default=None)
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()

def parse_model_id(s):
    _, name = os.path.split(s)
    id = '.'.join(name.split('.')[:-1]) or name
    return id

def init_run(prefix, name, device, local, config):
    model = \
        (Transformer if prefix.startswith('ar') else Evolver)(
            d_model=config['d_model'],
            nhead=config['nhead'],
            max_len=config['max_len'],
            encoder_layers=config['encoder_layers'],
            decoder_layers=config['decoder_layers'],
            device=device
        ).to(device)
    
    optim = AdamW(model.parameters(), lr=config['lr'])
    
    lr_scheduler = OneCycleLR(
        optim,
        max_lr=config['lr'],
        total_steps=config['train_steps'],
        pct_start=config['warmup_percent'],
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=10,
        final_div_factor=1
    )
   
    wandb_run_id = None 
    start_step = 0
    if 'from_checkpoint' in config:
        logger.info(f'Loading from {config["from_checkpoint"]}')
        checkpoint = torch.load(config['from_checkpoint'])
        model.load_state_dict(checkpoint['model'])

        if 'resume' in config:
            optim.load_state_dict(checkpoint['optim'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_step = checkpoint['step'] + 1
            wandb_run_id = checkpoint['wandb_run_id']

    if wandb_run_id is None:
        wandb_run_id = wandb.util.generate_id()

    if not local:
        wandb.init(
            id=wandb_run_id,
            project='evolver',
            name=name,
            config=config,
            resume='allow',
            notes=config.get('notes', 'N/A')
        )
    
    return model, optim, lr_scheduler, start_step

def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level))
    
    with open(args.config, 'r') as f: config = json.load(f)
    prefix = parse_model_id(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'{prefix}_{timestamp}'
    
    model, optim, lr_scheduler, start_step = \
        init_run(prefix, name, args.device, args.local, config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
       
    # autoregressive seq2seq 
    if prefix.startswith('ar-d'):
        train_dataset = Seq2SeqDataset.from_trajectories(
            path=config['train'],
            denoising=True,
            max_len=config['max_len'],
            tokenizer=tokenizer
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=StratifiedInfiniteSampler(train_dataset, config['batch_size'])
        )
        
        eval_dataset = Seq2SeqDataset.from_trajectories(
            path=config['eval'],
            denoising=True,
            max_len=config['max_len'],
            tokenizer=tokenizer,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )

        train_ar(
            model, optim, lr_scheduler,
            train_loader, eval_loader,
            train_steps=config['train_steps'],
            grad_accum_steps=config['grad_accum_steps'],
            checkpoint_at=config['checkpoint_at'],
            eval_at=config['eval_at'],
            device=args.device,
            name=name
        ) 
       
    ### baseline autoregressive 
    elif prefix.startswith('ar'):
        pass
   
    ### supervised evolver
    elif prefix.startswith('sup'):
        train_dataset = SupervisedTrajectoryDataset.from_disk(
            path=config['train'],
            max_len=config['max_len'],
            tokenizer=tokenizer
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=StratifiedInfiniteSampler(train_dataset, config['batch_size']),
            collate_fn=collate_supervised
        )
        
        eval_dataset = TrajectoryDataset.from_disk(
            path=config['eval'],
            max_len=config['max_len'],
            tokenizer=tokenizer,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config['batch_size'],
            sampler=StratifiedInfiniteSampler(eval_dataset, config['batch_size']),
            collate_fn=collate_unsupervised
        )
        
        train_evolver(
            model, optim, lr_scheduler,
            train_loader, eval_loader,
            train_steps=config['train_steps'],
            eval_steps=config['eval_steps'],
            grad_accum_steps=config['grad_accum_steps'],
            checkpoint_at=config['checkpoint_at'],
            eval_at=config['eval_at'],
            device=args.device,
            name=name,
            start_step=start_step
        )
    
    ### unsupervised evolver 
    else:
        train_dataset = TrajectoryDataset.from_disk(
            path=config['train'],
            max_len=config['max_len'],
            tokenizer=tokenizer
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=StratifiedInfiniteSampler(train_dataset, config['batch_size']),
            collate_fn=collate_unsupervised
        )
        
        eval_dataset = TrajectoryDataset.from_disk(
            path=config['eval'],
            max_len=config['max_len'],
            tokenizer=tokenizer,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config['batch_size'],
            sampler=StratifiedInfiniteSampler(eval_dataset, config['batch_size']),
            collate_fn=collate_unsupervised
        )
        
        train_evolver(
            model, optim, lr_scheduler,
            train_loader, eval_loader,
            train_steps=config['train_steps'],
            eval_steps=config['eval_steps'],
            grad_accum_steps=config['grad_accum_steps'],
            checkpoint_at=config['checkpoint_at'],
            eval_at=config['eval_at'],
            num_particles=config['num_particles'],
            threshold=config['threshold'],
            temperature=config['temperature'],
            resample_at=config['resample_at'],
            device=args.device,
            name=name,
            start_step=start_step
        )

if __name__ == '__main__':
    main()
