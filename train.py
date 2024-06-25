import time
import json
import logging
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from model import Evolver, Transformer
from run import sample_trajectory, sample_batch
from constants import PAD_TOKEN_ID
from data import elaborate, TrajectoryDataset, StratifiedInfiniteSampler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def plot_edit_loss(op_losses, tok_losses, idx_losses, prefix='test'):
    fig, ax = plt.subplots()
    ax.plot(op_losses, color='red', label='op loss')
    ax.plot(tok_losses, color='blue', label='tok loss')
    ax.plot(idx_losses, color='green', label='idx loss')
    
    # the sum of the xents aren't really useful
    # ax.plot([sum(x) for x in zip(op_losses, tok_losses, idx_losses)], color='black', label='total')
    
    ax.set_xlabel('training step')
    ax.set_ylabel('per-token xent')
    ax.legend({'op loss': 'red', 'tok loss': 'blue', 'idx loss': 'green'})
    
    plt.tight_layout()
    plt.savefig(f'figures/{prefix}-loss.png')
    plt.close(fig)
    
def plot_eval_loss(eval_losses, prefix='test'):
    fig, ax = plt.subplots()
    ax.plot(eval_losses)
    ax.set_xlabel('epoch')
    ax.set_ylabel('per-token approximate ELBO')
    
    plt.tight_layout()
    plt.savefig(f'figures/{prefix}-eval-loss.png')
    plt.close(fig)
    
def plot_loss(losses, prefix):
    plt.plot(losses)
    plt.xlabel('training step')
    plt.ylabel('per-token xent')
    plt.tight_layout()
    plt.savefig(f'figures/{prefix}-baseline-loss.png')
   
def log_edits(traj_edit_tgts):
    logger.debug(
        '\n' + 
        '\n'.join(
            ' '.join([e.ljust(8) for e in edit_tgts])
            for edit_tgts in elaborate(traj_edit_tgts)
        )
    )
    
def log_memory():
    if not torch.cuda.is_available(): return
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    current_memory = torch.cuda.memory_allocated() / 1024**2
    logger.info(f'Peak memory: {peak_memory:.2f}MB')
    logger.info(f'Current memory: {current_memory:.2f}MB')
    
def train_evolver(
    evolver, optim, lr_scheduler, train_loader, eval_loader,
    train_steps, grad_accum_steps, checkpoint_at, eval_at,
    num_particles, threshold, temperature,
    device, prefix
):
    op_losses = []
    tok_losses = []
    idx_losses = []
    eval_losses = []
   
    for step, (batch_ids, _) in enumerate(train_loader):
        if step >= train_steps: break
        logger.info(f'step: {step + 1}')
       
        s = time.time() 
        log_memory()
        
        # E-step
        evolver.eval() 
        traj_input_ids, traj_edit_tgts = \
            sample_batch(evolver, batch_ids, num_particles, threshold, temperature, device)
        
        # M-step
        evolver.train()
        traj_loss, op_loss, tok_loss, idx_loss = \
            evolver.traj_loss(traj_input_ids, traj_edit_tgts)
        
        if step % grad_accum_steps == 0:
            optim.zero_grad()
            traj_loss.backward()
            optim.step()
        
        logger.info(f'loss: {op_loss + tok_loss + idx_loss}')
        op_losses.append(op_loss.cpu().item())
        tok_losses.append(tok_loss.cpu().item())
        idx_losses.append(idx_loss.cpu().item())
        logger.info(f'op: {op_losses[-1]}, tok: {tok_losses[-1]}, idx: {idx_losses[-1]}')
        
        plot_edit_loss(op_losses, tok_losses, idx_losses, prefix=prefix)
    
        if (step + 1) % checkpoint_at == 0:
            save_path = f'/scratch4/jeisner1/{prefix}' if device == 'cuda' else 'checkpoints'
            torch.save(evolver.state_dict(), f'{save_path}/{prefix}-model-{step+1}')
            torch.save(optim.state_dict(), f'{save_path}/{prefix}-optim-{step+1}')
            
        if (step + 1) % eval_at == 0:
            s = time.time()
            avg_eval_loss = evaluate_evolver(evolver, eval_loader, device)
            logger.info(f'completed eval in {(time.time() - s):.3f} seconds')
            eval_losses.extend(avg_eval_loss for _ in range(eval_at))
            logger.info(f'eval loss: {eval_losses[-1]}')
            plot_eval_loss(eval_losses, prefix=prefix)
            
def evaluate_evolver(evolver, eval_loader, device):
    evolver.eval()
    cur_eval_losses = []
    
    for _traj_input_ids, _log_posterior in eval_loader:
        traj_input_ids = _traj_input_ids.squeeze().to(device)
        log_posterior = _log_posterior.squeeze()
        
        traj, log_likelihood = sample_trajectory(
            evolver, traj_input_ids,
            num_particles=1, threshold=0, temperature=1,
            device=device
        )
        
        log_edits(traj)
        num_toks = torch.sum(traj_input_ids[-1] != PAD_TOKEN_ID)
        cur_eval_losses.append((log_likelihood - log_posterior) / num_toks)
        
        print(cur_eval_losses[-1])
            
    return torch.mean(torch.stack(cur_eval_losses)).cpu().item()

def train_ar(
    model, optim, train_loader,
    epochs, checkpoint_at, eval_at,
    prefix
):
    losses = [] 
    
    pass

def train_ar(
    model, optim, train_loader,
    epochs, checkpoint_at, eval_at,
    prefix, **_
):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        for traj_input_ids in train_loader:
            traj_input_ids = torch.stack(traj_input_ids)
            loss = model.step(optim, traj_input_ids)
            
            losses.append(loss.cpu().item())
            logger.info(f'loss: {loss}')
            
            plot_loss(losses, prefix)
            
        if (epoch + 1) % checkpoint_at == 0:
            logger.info('checkpointing...')
            torch.save(model.state_dict(), f'checkpoints/{prefix}-model-baseline-{epoch+1}')
            torch.save(optim.state_dict(), f'checkpoints/{prefix}-optim-baseline-{epoch+1}')
            
        if (epoch + 1) % eval_at == 0:
            logger.info('eval...')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--eval')
    parser.add_argument('--config')
    parser.add_argument('--prefix')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()
    
def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level))
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    evolver = Evolver(
        d_model=config['d_model'],
        nhead=config['nhead'],
        max_len=config['max_len'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        device=args.device
    ).to(args.device)
    
    optim = AdamW(evolver.parameters(), lr=config['lr'])
    
    train_dataset = TrajectoryDataset.from_disk(
        path=args.train,
        max_len=config['max_len'],
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=StratifiedInfiniteSampler(train_dataset, 128),
        collate_fn=lambda x: x # don't try to stack mismatched trajectories
    )
    
    eval_dataset = TrajectoryDataset.from_disk(
        path=args.eval,
        max_len=config['max_len'],
        tokenizer=tokenizer,
        limit=config['eval_limit']
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=True
    )
    
    train_evolver(
        evolver, optim, None,
        train_loader, eval_loader,
        train_steps=config['epochs'],
        grad_accum_steps=config['grad_accum_steps'],
        checkpoint_at=config['checkpoint_at'],
        eval_at=config['eval_at'],
        num_particles=config['num_particles'],
        threshold=config['threshold'],
        temperature=config['temperature'],
        device=args.device,
        prefix=args.prefix
    )

if __name__ == '__main__':
    main()
            
### deprecated

def train_forced(
    evolver, optim, train_loader,
    epochs, checkpoint_at, eval_at, prefix
):
    evolver.train()

    op_losses = []
    tok_losses = []
    idx_losses = []
    
    for epoch in range(epochs):
        logger.info(f'starting epoch: {epoch + 1}') 
        
        for batch in train_loader:
            op_loss, tok_loss, idx_loss = evolver.step(optim, *batch)

            logger.info(f'loss: {-(op_loss + tok_loss + idx_loss)}')
            op_losses.append(-op_loss.cpu().item())
            tok_losses.append(-tok_loss.cpu().item())
            idx_losses.append(-idx_loss.cpu().item())
            logger.info(f'op: {op_losses[-1]}, tok: {tok_losses[-1]}, idx: {idx_losses[-1]}')
            
            plt.edit_loss(op_losses, tok_losses, idx_losses, prefix=prefix)
        
        if (epoch + 1) % checkpoint_at == 0:
            logger.info('checkpointing...')
            torch.save(evolver.state_dict(), f'checkpoints/{prefix}-model-{epoch+1}')
            torch.save(optim.state_dict(), f'checkpoints/{prefix}-optim-{epoch+1}')
            
        if (epoch + 1) % eval_at == 0:
            pass
