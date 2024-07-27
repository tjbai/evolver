import os
import json
import time
import math
import pickle
import logging
import argparse
from datetime import datetime

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import Transformer as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import SinusoidalEmbedding

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO -- these default come from gum
TOK_V, REL_V, POS_V = 19030, 69, 17
INS_ID, CPY_ID, PRO_ID, EOS_ID = 0, 1, 2, 3

REMOTE_PREFIX = os.environ.get('REMOTE_PREFIX', '/scratch4/jeisner1')

def log(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: print(f"step {step}: {data}")
    
class ParseDataset(Dataset):
    
    @classmethod
    def from_pkl(cls, path, *args, **kwargs):
        with open(path, 'rb') as f:
            samples = pickle.load(f)
            return cls(samples, *args, **kwargs)
   
    # each sample is a (root, tgt)  tuple
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        return self.samples[i]
    
def collate_single(sample):
    assert len(sample) == 1
    
    [(root, traj)] = sample

    # the java is rubbing off on me
    tgts = torch.tensor(traj) \
                .transpose(0, 1) \
                .unsqueeze(2)
                
    return root, tgts

def single_loader(path, shuffle=True):
    dataset = ParseDataset.from_pkl(path) 
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_single, shuffle=shuffle)
    return loader

class DependencyEvolver(nn.Module):
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        encoder_layers=6,
        decoder_layers=6,
        N=64,
        tok_v=TOK_V,
        rel_v=REL_V,
        pos_v=POS_V,
        name='test',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.N = N
        self.d_model = d_model
        self.name = name
        self.device = device
        self.rel_offset = tok_v
        self.pos_offset = tok_v + rel_v
        self.vocab_size = tok_v + rel_v + pos_v + 1 # add one for UNK
        self.causal_mask = T.generate_square_subsequent_mask(N, dtype=torch.bool)
        
        codec_params = {
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'dropout': dropout,
            'batch_first': True
        }
       
        encoder_layer = nn.TransformerEncoderLayer(**codec_params)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(**codec_params)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model, max_len=N)
        
        self.op_head = nn.Linear(d_model, 4)
        self.cpy_head = nn.Linear(d_model, 512)
        self.par_head = nn.Linear(d_model, 512)
        self.v_head = nn.Linear(d_model, tok_v+rel_v+pos_v)
       
        self.done = nn.Parameter(torch.zeros(d_model))
        self.plh = nn.Parameter(torch.zeros(d_model))
        self.root_bos = nn.Parameter(torch.zeros(d_model))
        self.root_eos = nn.Parameter(torch.zeros(d_model))
        self.init_params()
    
    def init_params(self):
        for param in [self.done, self.plh, self.root_bos, self.root_eos]:
            nn.init.trunc_normal_(param, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        
    def root(self, tok, rel, pos):
        root = (self.embedding.weight[tok] + self.embedding.weight[rel] + self.embedding.weight[pos]).unsqueeze(0)
        embeds = torch.cat([self.root_bos.unsqueeze(0), root, self.root_eos.unsqueeze(0)], dim=0)
        embeds = torch.cat([embeds, torch.full((self.N-3, self.d_model), 0)])
        return embeds.unsqueeze(0)
    
    def _replace(self, t, a, b):
        return torch.where(t == a, b, t)
    
    def tgt_op(self, mem, tgt_op, tgt_cpy):
        B, N, _ = mem.shape
        
        permuted_mem = self.positional_embedding(mem, d=-1)[
            torch.arange(B, device=mem.device).unsqueeze(1),
            self._replace(tgt_cpy, -1, 0)
        ]
       
        tgt = torch.where((tgt_op.eq(CPY_ID) | tgt_op.eq(PRO_ID)).unsqueeze(-1).expand_as(mem), permuted_mem, 0) \
            + torch.where(tgt_op.eq(INS_ID).unsqueeze(-1).expand_as(mem), self.plh, 0) \
            + torch.where(tgt_op.eq(PRO_ID).unsqueeze(-1).expand_as(mem), self.done, 0) \
            + torch.where(tgt_op.eq(EOS_ID).unsqueeze(-1).expand_as(mem), mem, 0) \

        return self.positional_embedding(tgt, d=1)
    
    def forward_op(self, src, tgt_op, tgt_cpy, src_pad_mask, *_):
        encoder_masks = {'src_key_padding_mask': src_pad_mask}
        decoder_masks = {'tgt_mask': self.causal_mask, 'memory_key_padding_mask': src_pad_mask}
        
        mem = self.encoder(src, **encoder_masks)
        tgt = self.tgt_op(mem, tgt_op, tgt_cpy)
        h = self.decoder(tgt, mem, **decoder_masks)
        l_op = self.op_head(h)
        l_cpy = self.cpy_head(h) 
        
        return l_op, l_cpy, tgt
    
    def tgt_par(self, mem, tgt_par):
        B, _, _ = mem.shape 
        
        permuted_mem = mem[
            torch.arange(B, device=mem.device).unsqueeze(1),
            torch.where(tgt_par == -1, 0, tgt_par)
        ]
        
        tgt = mem \
            + torch.where((tgt_par > 0).unsqueeze(-1).expand_as(mem), permuted_mem, 0)
        
        return tgt
    
    def forward_par(self, src, tgt_par, _, src_pad_mask, tgt_pad_mask):
        encoder_masks = {'src_key_padding_mask': src_pad_mask}
        decoder_masks = {'tgt_mask': self.causal_mask, 'tgt_key_padding_mask': tgt_pad_mask, 'memory_key_padding_mask': src_pad_mask}
        
        mem = self.encoder(src, **encoder_masks)
        tgt = self.tgt_par(mem, tgt_par)
        h = self.decoder(tgt, mem, **decoder_masks)
        l = self.par_head(h)
        
        # TODO -- constrain?
        
        return l, tgt
    
    def tgt_gen(self, mem, tgt_gen):
        embeds = self.embedding(self._replace(tgt_gen, -1, 0))
        return mem + torch.where(~tgt_gen.eq(-1).unsqueeze(-1).expand_as(mem), embeds, 0)
    
    def forward_gen(self, src, tgt_gen, _, src_pad_mask, tgt_pad_mask):
        encoder_masks = {'src_key_padding_mask': src_pad_mask}
        decoder_masks = {'tgt_mask': self.causal_mask, 'tgt_key_padding_mask': tgt_pad_mask, 'memory_key_padding_mask': src_pad_mask}
        
        mem = self.encoder(src, **encoder_masks)
        tgt = self.tgt_gen(mem, tgt_gen)
        h = self.decoder(tgt, mem, **decoder_masks)
        l = self.v_head(h)
        
        return l, tgt
    
    def forward(
        self, src, pad_masks,
        tgt_op, tgt_cpy, tgt_par,
        tgt_rel, tgt_pos, tgt_tok
    ):
        # procreate
        l_op, l_cpy, src = self.forward_op(src, tgt_op, tgt_cpy, *pad_masks)
        
        # child rearing
        l_par, src = self.forward_par(src, tgt_par, *pad_masks)
       
        # infancy, adolescence, and adulthood
        l_rel, src = self.forward_gen(src, tgt_rel, *pad_masks)
        l_pos, src = self.forward_gen(src, tgt_pos, *pad_masks)
        l_tok, src = self.forward_gen(src, tgt_tok, *pad_masks)
        
        return l_op, l_cpy, l_par, l_rel, l_pos, l_tok
    
    def _record(self, ls, step):
        prefix = 'train' if self.training else 'eval'
        log({
            f'{prefix}/op': ls[0],
            f'{prefix}/cpy': ls[1],
            f'{prefix}/par': ls[2],
            f'{prefix}/rel': ls[3],
            f'{prefix}/pos': ls[4],
            f'{prefix}/tok': ls[5]
        }, step=step)
    
    def xent(self, l, t, ignore=-1):
        l = l[:, :-1]
        t = t[:, 1:]
        mask = t != ignore
        return -F.log_softmax(l[mask], dim=-1).gather(1, t[mask].unsqueeze(1))
   
    '''
    TODO -- write some documentation for when i inevitably forget how this works
            lots of mental overhead with the way i handle pad masks, etc. here
    ''' 
    def traj_loss(self, root, tgts, step=None):
        T, _, B, N = tgts.shape
        
        tot = [0 for _ in range(6)]
        num = [0 for _ in range(6)]
        
        src = self.root(*root)
        init_pad_mask = torch.full((B, N), True)
        init_pad_mask[:, :3] = False
        
        for i in range(T):
            src_pad_mask = tgts[i, 0].eq(-1)
            tgt_pad_mask = tgts[i, 3].eq(-1)
            tgt_pad_mask[:, 0] = False
            
            pad_masks = (init_pad_mask, src_pad_mask, tgt_pad_mask)
            ls = self.forward(src, pad_masks, *(t for t in tgts[i]))
            
            for j, (l, t) in enumerate(zip(ls, tgts[i])):
                loss = self.xent(l, t)
                tot[j] += torch.sum(loss)
                num[j] += torch.numel(loss)
            
            init_pad_mask = src_pad_mask
            
        loss = [(t/n if n > 0 else 0) for t, n in zip(tot, num)]
        self._record(loss, step)
        return sum(loss)
    
    def _save(self, step, optim):
        path = f'{REMOTE_PREFIX}/checkpoints' if self.device == 'cuda' else 'checkpoints'
        torch.save({
            'step': step,
            'model': self.state_dict(),
            'optim': optim.state_dict(),
            'wandb_run_id': None if wandb.run is None else wandb.run.id
        }, f'{path}/{self.name}-{step+1}.pt')
        
    def _eval(self, eval_loader, eval_steps):
        self.eval()
      
        tot = n = 0
        for step, (root, tgts) in enumerate(eval_loader):
            if step >= eval_steps: break
            tgts = tuple(map(lambda x: x.to(self.device), tgts))
            tot += self.traj_loss(root, *tgts)
            n += tgts[0].shape[0] + torch.sum(tgts[0] != -1)
    
        return tot / n
    
    def _train(
        self, optim, train_loader, eval_loader,
        train_steps, eval_steps, grad_accum_steps,
        checkpoint_at, eval_at,
        start_step=0
    ):
        for step, (root, tgts) in tqdm(
            enumerate(train_loader, start=start_step),
            total=train_steps
        ):
            if step >= train_steps: break
            tgts = tgts.to(self.device)
            
            self.train()
            loss = self.traj_loss(root, tgts, step=step)
            log({'train/loss': loss}, step=step)
            
            loss.backward()
            if step % grad_accum_steps == 0:
                optim.step()
                optim.zero_grad()
            
            if (step + 1) % checkpoint_at == 0:
                self._save(step, optim)
                
            if (step + 1) % eval_at == 0:
                s = time.time()
                loss = self._eval(eval_loader, eval_steps)
                log({'eval/loss': loss, 'eval/time': time.time()-s}, step=step)
                
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--local', action='store_true')
    return parser.parse_args()

def init_run(config, name, local):
    
    with open(config['tok_v'], 'r') as f: tok_v = len(f.readlines())
    with open(config['pos_v'], 'r') as f: pos_v = len(f.readlines())
    with open(config['rel_v'], 'r') as f: rel_v = len(f.readlines())
    
    model = DependencyEvolver(
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        N=config['N'],
        tok_v=tok_v,
        rel_v=rel_v,
        pos_v=pos_v,
        name=name
    )
    
    model = model.to(model.device)
    
    optim = AdamW(model.parameters(), lr=config['lr'])
    
    start_step = 0
    wandb_run_id = None 
    if 'from_checkpoint' in config:
        logger.info(f'loading from {config["from_checkpoint"]}')
        checkpoint = torch.load(config['from_checkpoint'])
        model.load_state_dict(checkpoint['model'])

        if 'resume' in config:
            optim.load_state_dict(checkpoint['optim'])
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
    
    return model, optim, start_step

def parse_model_id(s):
    _, name = os.path.split(s)
    id = '.'.join(name.split('.')[:-1]) or name
    return id
                
def main():
    args = parse_args()
    
    with open(args.config, 'r') as f: config = json.load(f)
    
    prefix = parse_model_id(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'{prefix}_{timestamp}'
    
    model, optim, start_step = init_run(config, name, args.local)
    train_loader = single_loader(config['train'])
    eval_loader = single_loader(config['eval'])
    
    model._train(
        optim=optim,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_steps=config['train_steps'],
        eval_steps=config['eval_steps'],
        grad_accum_steps=config['grad_accum_steps'],
        checkpoint_at=config['checkpoint_at'],
        eval_at=config['eval_at'],
        start_step=start_step
    )
                
if __name__ == '__main__':
    main()