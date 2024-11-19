import os
import json
import yaml
import logging
import argparse
from datetime import datetime

import wandb
import torch
import torch.distributed as dist
from torch.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm

from .data import WMT, EvolverWMT, MarianTokenizer
from .models import Transformer, Evolver, Teacher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

def log_to_wandb(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f'step {step}: {data}')

def save_checkpoint(model, optimizer, step, config):
    save_path = os.path.join(config['checkpoint_dir'], f'{model.name}_{step}.pt')
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
    
def train_step(model, batch, device, step=None):
    if isinstance(model, Transformer):
        return model.step({k: v.to(device) for k, v in batch.items()})
    else:
        loss = model.step({
            'src_ids': batch['src_ids'].to(device),
            'input_ids': batch['input_ids'].to(device),
            'tgt_ids': batch['tgt_ids'].to(device),
            'edit_ids': tuple(map(lambda x: x.to(device), batch['edit_ids']))
        })

        if step is not None and isinstance(model, Evolver):
            log_to_wandb({
                'train/op_loss': loss[0],
                'train/tok_loss': loss[1],
                'train/idx_loss': loss[2]}, step=step)

        return sum(loss) if isinstance(model, Evolver) else loss

@torch.no_grad()
def evaluate(model, eval_loader, device, num_eval_steps):
    model.eval()
    
    tot_loss = 0
    num_samples = 0

    for i, batch in enumerate(tqdm(eval_loader, desc="eval...", total=num_eval_steps)):
        if i >= num_eval_steps: break
        
        if isinstance(model, Transformer):
            loss = train_step(model, batch, device)
            tot_loss += loss.item()
            generated = model.generate(batch['src_ids'].to(device))
            
        else:
            loss = train_step(model, batch, device)
            tot_loss += loss * batch['src_ids'].shape[0]
            num_samples += batch['src_ids'].shape[0]

            # TODO -- figure out the distributed sampling
   
    # NOTE -- not fully sure this works but that's what the sanity check's for! 
    dist.all_reduce(tot_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
    return tot_loss.item() / num_samples.item(), 0
    
def init_model(config):
    params = {
        'd_model': config['d_model'],
        'dim_feedforward': config['dim_feedforward'],
        'nhead': config['nhead'],
        'dropout': config['dropout'],
        'layer_norm_eps': config['layer_norm_eps'],
        'decoder_layers': config['decoder_layers'],
        'encoder_layers': config['encoder_layers'],
        'max_len': config['max_len'],
        'name': config['name'],
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'vocab_size': tokenizer.vocab_size+1, # add one for special bos
    }
    
    if config['model_type'] == 'decoder_only': return Transformer(**params)
    elif config['model_type'] == 'evolver': return Evolver(**params)
    else: return Teacher(**params)
    
def load_checkpoint(model, optimizer, config):
    if config.get('from_checkpoint') is not None:
        checkpoint = torch.load(config['from_checkpoint'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step'] + 1
        logger.info(f'resuming from step {start_step}')
        return start_step
    return 0

def train(config):
    device = torch.device(config['device'])
    logger.info(f'using device: {device} here')
    
    dataset = WMT if config['model_type'] == 'decoder_only' else EvolverWMT
    train_dataset = dataset(split='train', max_len=config['max_len'], truncate=config.get('truncate'))
    eval_dataset = dataset(split='validation', max_len=config['max_len'], truncate=config.get('truncate'))
    logger.info('loaded datasets!')
    
    kwargs = {'batch_size': config['batch_size'], 'collate_fn': train_dataset.collate_fn, 'num_workers': config['num_workers']}
    sampler = DistributedSampler(dataset, num_replicas=config['world_size'], rank=config['rank'])
    train_loader = DataLoader(train_dataset, sampler=sampler, **kwargs)

    # TODO -- maybe just do single node for now...
    eval_loader = DataLoader(eval_dataset, **kwargs)
   
    model = init_model(config).to(device)
    model = DDP(model, device_ids=[config['rank']])
    optim = AdamW(model.parameters(), lr=config['lr'])
    step = load_checkpoint(model, optim, config)
    logger.info('loaded model!')
    
    if not config['skip']:
        logger.info('starting eval sanity check...')
        evaluate(model, eval_loader, device, 1)
        logger.info('sanity check passed')
    
    model.train()
    for _ in range(config['train_epochs']):
        for batch in tqdm(train_loader, disable=config['local']):
            if step >= config['train_steps']: break
            
            loss = train_step(model, batch, device, step=step)
            loss.backward()

            if (step + 1) % config['grad_accum_steps'] == 0:
                optim.step()
                optim.zero_grad()

            if (step + 1) % config['log_every'] == 0:
                log_to_wandb({'train/loss': loss.item()}, step=step)

            if (step + 1) % config['eval_every'] == 0:
                eval_loss, bleu_score = evaluate(model, eval_loader, device, config['num_eval_steps'])
                log_to_wandb({'eval/loss': eval_loss, 'eval/bleu': bleu_score}, step=step)
                model.train()

            if (step + 1) % config['save_every'] == 0:
                save_checkpoint(model, optim, step, config)
            
            step += 1

    eval_loss, bleu_score = evaluate(model, eval_loader, device, config['num_eval_steps'])
    log_to_wandb({'eval/loss': eval_loss, 'eval/bleu': bleu_score}, step=step)
    save_checkpoint(model, optim, config['train_steps'], config)

def load_config(config_path):
    with open(config_path, 'r') as f:
        if config_path[-3:] == 'yml':
            return yaml.safe_load(f)
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--skip', action='store_true')
    return parser.parse_args()

def setup_ddp():
    dist.init_process_group(backends='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    logger.info(f'training on {rank} out of {world_size}')
    logger.info(f'master addr: {os.environ.get("MASTER_ADDR")}, master port: {os.environ.get("MASTER_PORT")}')
    return rank, world_size

def main():
    args = parse_args()
    config = load_config(args.config)

    config['device'] = args.device
    config['local'] = args.local
    config['skip'] = args.skip
    config['name'] = (
        f"mt_{config['model_type']}_,
        {config['d_model']}d_,
        {config.get('encoder_layers', 0)}enc_{config['decoder_layers']}dec_,
        {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    rank, world_size = setup_ddp()
    config['rank'] = rank
    config['world_size'] = world_size
    torch.cuda.set_device(rank)
    
    if not config['local']: wandb.init(project='mt-evolver', name=config['name'], config=config, resume='allow')

    train(config)
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
