import sys
if '..' not in sys.path: sys.path.append('..')

import wandb

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertTokenizer

from model import Evolver
from train import train_evolver
from data import (
    TrajectoryDataset,
    SupervisedTrajectoryDataset,
    StratifiedInfiniteSampler,
    collate_supervised,
    collate_unsupervised
)

CACHE_PREFIX = 'sup-imdb-3'

def main():
    wandb.init()
    config = wandb.config
    enc, dec = map(int, config.layer_allocation.split('-'))
    
    model = Evolver(
        d_model=config.d_model,
        nhead=config.nhead,
        max_len=config.max_len,
        dropout=config.dropout,
        dim_feedforward=config.dim_feedforward,
        encoder_layers=enc,
        decoder_layers=dec,
        op_scale=config.op_scale,
        tok_scale=config.tok_scale,
        idx_scale=config.idx_scale,
        device=config.device
    ).to(config.device)
    
    optim = AdamW(model.parameters(), lr=config.lr)

    lr_scheduler = OneCycleLR(
        optim,
        max_lr=config.lr,
        total_steps=config.train_steps,
        pct_start=config.warmup_percent,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=10,
        final_div_factor=1
    )
   
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = SupervisedTrajectoryDataset.from_disk(
        path=config.train,
        max_len=config.max_len,
        tokenizer=tokenizer,
        cache_prefix=CACHE_PREFIX,
        all_tokens=config.all_tokens
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=StratifiedInfiniteSampler(train_dataset, config.batch_size),
        collate_fn=collate_supervised,
        num_workers=0,
        pin_memory=True
    )
    
    eval_dataset = TrajectoryDataset.from_disk(
        path=config.eval,
        max_len=config.max_len,
        tokenizer=tokenizer,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        sampler=StratifiedInfiniteSampler(eval_dataset, config.batch_size),
        collate_fn=collate_unsupervised
    )
    
    train_evolver(
        model, optim, lr_scheduler,
        train_loader, eval_loader,
        train_steps=config.train_steps,
        eval_steps=config.eval_steps,
        grad_accum_steps=config.grad_accum_steps,
        clip_gradients=config.clip_gradients,
        checkpoint_at=config.checkpoint_at,
        eval_at=config.eval_at,
        device=config.device,
        name=wandb.run.name,
        start_step=0
    )
    
if __name__ == '__main__':
    main()
