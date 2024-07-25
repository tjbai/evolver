import os
import time
import json
import logging
import argparse
from datetime import datetime

import wandb
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer
from tqdm import tqdm

from model import Evolver, NoShareEvolver, Transformer
from run import sample_trajectory
from data import (
    elaborate,
    TrajectoryDataset,
    SequenceDataset,
    StratifiedInfiniteSampler,
    supervised_loader,
    unsupervised_loader
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)
    print(f'RUNNING ON: {gpu_properties.name}')

REMOTE_PREFIX = os.environ.get('REMOTE_PREFIX', '/scratch4/jeisner1')
   
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

def grad_norm(mod):
    tot = 0
    for p in mod.parameters():
        if p.grad is not None:
            norm = p.grad.data.norm(2)
            tot += norm.item() ** 2
    return tot ** 0.5

def record_grad_norms(evolver, step):
    log({
        'train/op_grad_norm': grad_norm(evolver.op_head),
        'train/tok_grad_norm': grad_norm(evolver.tok_head),
        'train/idx_grad_norm': grad_norm(evolver.idx_head)
    }, step=step) 

def train_evolver(
    evolver, optim, lr_scheduler, train_loader, eval_loader,
    train_steps, eval_steps, grad_accum_steps, clip_gradients,
    checkpoint_at, eval_at,
    num_particles=None, threshold=None, temperature=1.0, resample_at=1,
    device='cuda', name='test', start_step=0
):
    for step, (traj_input_ids, _, traj_edit_tgts, _) in tqdm(
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
        traj_loss, op_loss, tok_loss, idx_loss = evolver.traj_loss(traj_input_ids, traj_edit_tgts, step=step)
            
        traj_loss.backward()
        record_grad_norms(evolver, step)
        if clip_gradients:
            clip_grad_norm_(evolver.parameters(), 1)
        
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
            save_path = f'{REMOTE_PREFIX}/checkpoints' if device == 'cuda' else 'checkpoints'
            torch.save({
                'step': step,
                'model': evolver.state_dict(),
                'optim': optim.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'wandb_run_id': None if wandb.run is None else wandb.run.id
            }, f'{save_path}/{name}-{step+1}.pt')
            
        if (step + 1) % eval_at == 0:
            s = time.time()
            eval_loss = evolver_elbo(evolver, eval_loader, eval_steps, device)
            log({'eval/loss': eval_loss, 'eval/time': time.time()-s}, step=step)

@torch.no_grad()
def evolver_elbo(evolver, eval_loader, eval_steps, device):
    evolver.eval()
    tot_loss = 0
    tot_n = 0
    
    for step, (traj_input_ids, log_posterior, _, n) in enumerate(eval_loader):
        if step >= eval_steps: break
        
        traj_input_ids = traj_input_ids.to(device)
        log_posterior = log_posterior.to(device)
        
        _, log_likelihood = sample_trajectory(evolver, traj_input_ids, num_particles=1, temperature=0.5)
        tot_loss += torch.sum(log_likelihood - log_posterior)
        tot_n += n
       
    return tot_loss / tot_n

def train_ar(
    model, optim, lr_scheduler,
    train_loader, eval_loader,
    train_steps, eval_steps, grad_accum_steps,
    checkpoint_at, eval_at,
    device, name,
    start_step, input_is_tgt
):
    for step, (input_ids, output_ids) in tqdm(
        enumerate(train_loader, start=start_step),
        total=train_steps
    ):
        if step == train_steps: break

        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)
       
        model.train() 
        if input_is_tgt: tot_loss, n = model.loss(output_ids)
        else: tot_loss, n = model.loss(input_ids, output_ids)
        loss = tot_loss / n 
        loss.backward()
        
        if (step + 1) % grad_accum_steps == 0:
            optim.step()
            optim.zero_grad()
           
        if lr_scheduler: 
            lr_scheduler.step()
       
        log({'train/total_loss': loss.item()}, step=step)
        
        if (step + 1) % checkpoint_at == 0:
            save_path = f'{REMOTE_PREFIX}/checkpoints' if device == 'cuda' else 'checkpoints'
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'wandb_run_id': None if wandb.run is None else wandb.run.id
            }, f'{save_path}/{name}-{step+1}.pt')
            
        if (step + 1) % eval_at == 0:
            if isinstance(eval_loader.dataset, TrajectoryDataset):
                eval_loss = ar_elbo(model, eval_loader, eval_steps, device)
            elif isinstance(eval_loader.dataset, SequenceDataset):
                eval_loss = ar_likelihood(model, eval_loader, eval_steps, input_is_tgt, device)
            log({'eval/loss': eval_loss.item()}, step=step)

def traj_likelihood(model, traj_input_ids):
    B, T, _ = traj_input_ids.shape
    tot_loss = tot_n = 0
    
    for i in range(T-1):
        input_ids = traj_input_ids[:, i]
        output_ids = traj_input_ids[:, i+1]
        
        loss, _ = model.loss(input_ids, output_ids)
        tot_loss += loss
        
    return -tot_loss
            
@torch.no_grad()
def ar_elbo(model, eval_loader: TrajectoryDataset, eval_steps, device):
    model.eval()
    tot_loss = 0
    tot_n = 0
    
    for step, (traj_input_ids, log_posterior, _, n) in enumerate(eval_loader):
        if step >= eval_steps: break
        
        traj_input_ids = traj_input_ids.to(device)
        log_posterior = log_posterior.to(device)
        
        log_likelihood = traj_likelihood(model, traj_input_ids)
        tot_loss += log_likelihood - torch.sum(log_posterior)
        tot_n += n
       
    return tot_loss / tot_n

@torch.no_grad()
def ar_likelihood(model, eval_loader: SequenceDataset, eval_steps, input_is_tgt, device):
    model.eval()
    tot_loss = 0
    tot_n = 0
    
    for step, (input_ids, output_ids) in enumerate(eval_loader):
        if step >= eval_steps: break
    
        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)
        
        if input_is_tgt: loss, n = model.loss(output_ids)
        else: loss, n = model.loss(input_ids, output_ids)
        
        tot_loss += loss
        tot_n += n
            
    eval_loss = tot_loss / tot_n
    return -eval_loss
            
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
    num_encoders = config.get('num_encoders', 1)
    num_decoders = config.get('num_decoders', 1)

    Model = \
    Transformer if prefix.startswith('ar') \
    else NoShareEvolver if (num_encoders > 1 or num_decoders > 1) \
    else Evolver

    model = Model(
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        max_len=config['max_len'],
        op_scale=config.get('op_scale', 1),
        tok_scale=config.get('tok_scale', 1),
        idx_scale=config.get('idx_scale', 1),
        positional_embeddings=config.get('positional_embeddings', 'sinu'),
        static_embeddings=config.get('static_embeddings', False),
        depth_embeddings=config.get('depth_embeddings', False),
        num_encoders=num_encoders,
        num_decoders=num_decoders,
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
        logger.info(f'loading from {config["from_checkpoint"]}')
        checkpoint = torch.load(config['from_checkpoint'])
        model.load_state_dict(checkpoint['model'])

        if 'resume' in config:
            optim.load_state_dict(checkpoint['optim'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_step = checkpoint['step'] + 1
            wandb_run_id = checkpoint['wandb_run_id']
            logger.info(f'resuming from {start_step}')

    if wandb_run_id is None:
        wandb_run_id = wandb.util.generate_id()
        logger.info('starting new run')

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
    
    model, optim, lr_scheduler, start_step = init_run(prefix, name, args.device, args.local, config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
       
    # autoregressive seq2seq 
    if prefix.startswith('ar-d'):
        train_dataset = SequenceDataset.from_trajectories(
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
        
        eval_loader = unsupervised_loader(
            path=config['eval'],
            max_len=config['max_len'],
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
        )

        train_ar(
            model, optim, lr_scheduler,
            train_loader, eval_loader,
            train_steps=config['train_steps'],
            eval_steps=config['eval_steps'],
            grad_accum_steps=config['grad_accum_steps'],
            checkpoint_at=config['checkpoint_at'],
            eval_at=config['eval_at'],
            input_is_tgt=False,
            device=args.device,
            name=name,
            start_step=start_step
        ) 
       
    ### baseline autoregressive 
    elif prefix.startswith('ar'):
        train_dataset = SequenceDataset.from_trajectories(
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
        
        eval_dataset = SequenceDataset.from_trajectories(
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
            eval_steps=config['eval_steps'],
            grad_accum_steps=config['grad_accum_steps'],
            checkpoint_at=config['checkpoint_at'],
            eval_at=config['eval_at'],
            input_is_tgt=True,
            device=args.device,
            name=name,
            start_step=start_step
        )  
   
    ### supervised evolver
    elif prefix.startswith('sup'):
        train_loader = supervised_loader(
            path=config['train'],
            max_len=config['max_len'],
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
            cache_prefix=prefix.split('.')[0],
            all_tokens=config.get('all_tokens', False)
        )
        
        eval_loader = unsupervised_loader(
            path=config['eval'],
            max_len=config['max_len'],
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
        )
        
        train_evolver(
            model, optim, lr_scheduler,
            train_loader, eval_loader,
            train_steps=config['train_steps'],
            eval_steps=config['eval_steps'],
            grad_accum_steps=config['grad_accum_steps'],
            clip_gradients=config['clip_gradients'],
            checkpoint_at=config['checkpoint_at'],
            eval_at=config['eval_at'],
            device=args.device,
            name=name,
            start_step=start_step
        )
    
    ### unsupervised evolver 
    else:
        train_loader = unsupervised_loader(
            path=config['train'],
            max_len=config['max_len'],
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
        )
        
        eval_loader = unsupervised_loader(
            path=config['eval'],
            max_len=config['max_len'],
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
        )
        
        train_evolver(
            model, optim, lr_scheduler,
            train_loader, eval_loader,
            train_steps=config['train_steps'],
            eval_steps=config['eval_steps'],
            grad_accum_steps=config['grad_accum_steps'],
            clip_gradients=config['clip_gradients'],
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
