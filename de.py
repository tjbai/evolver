import os
import time
import math

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T
from tqdm import tqdm

from constants import *
from model import SinusoidalEmbedding

INS_ID = 0
CPY_ID = 1
PRO_ID = 2
EOS_ID = 3

REMOTE_PREFIX = os.environ.get('REMOTE_PREFIX', '/scratch4/jeisner1')

def log(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: print(f"step {step}: {data}")

class DependencyEvolver(nn.Module):
    
    def __init__(
        self,
        d_model=512, dim_feedforward=2048, nhead=8, dropout=0.1, N=64,
        encoder_layers=6, decoder_layers=6,
        tok_v=VOCAB_SIZE, rel_v=0, pos_v=0,
        pad_token_id=PAD_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        name='test',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.N = N
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.name = name
        self.device = device
        self.causal_mask = T.generate_square_subsequent_mask(N, dtype=torch.bool)
        self.rel_offset = tok_v
        self.pos_offset = tok_v + rel_v
        
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
        
        self.embedding = nn.Embedding(tok_v+rel_v+pos_v, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model, max_len=N)
        
        self.op_head = nn.Linear(d_model, 4)
        self.cpy_head = nn.Linear(d_model, 512)
        self.par_head = nn.Linear(d_model, 512)
        self.v_head = nn.Linear(d_model, tok_v+rel_v+pos_v)
       
        self.done = nn.Parameter(torch.zeros(d_model))
        self.plh = nn.Parameter(torch.zeros(d_model))
        self.init_params()
    
    def init_params(self):
        for param in [self.done, self.plh]:
            nn.init.trunc_normal_(param, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        
    def root(self, tok, pos, N):
        embeds = self.embedding(torch.tensor([self.bos_token_id, tok, self.eos_token_id]))
        embeds[0] += self.embedding.weight[pos]
        embeds = torch.cat([embeds, torch.full((N-3, self.d_model), 0)])
        return embeds.unsqueeze(0)
    
    def _replace(self, t, a, b):
        return torch.where(t == a, b, t)
    
    def tgt_op(self, mem, tgt_op, tgt_cpy):
        B, N, _ = mem.shape
        
        # permuted_mem = self.positional_embedding(mem, d=-1)[
        permuted_mem = mem[
            torch.arange(B, device=mem.device).unsqueeze(1),
            self._replace(tgt_cpy, -1, 0)
        ]
       
        tgt = torch.where(tgt_op.eq(INS_ID).unsqueeze(-1).expand_as(mem), self.plh, 0) \
            + torch.where(tgt_op.eq(PRO_ID).unsqueeze(-1).expand_as(mem), self.done, 0) \
            + torch.where((tgt_op.eq(CPY_ID) | tgt_op.eq(PRO_ID)).unsqueeze(-1).expand_as(mem), permuted_mem, 0)

        # return self.positional_embedding(tgt, d=1) 
        return tgt
    
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
   
    # op, cpy, par, rel, pos, tok
    def traj_loss(self, root, *tgts, step=None):
        op_traj, _, _, rel_traj, *_ =tgts 
        
        T, B, N = tgts[0].shape
        tot = [0 for _ in range(6)]
        num = [0 for _ in range(6)]
        
        src = self.root(*root, self.N)
        init_pad_mask = torch.full((B, N), True)
        init_pad_mask[:, :3] = False
        
        for i in range(T):
            src_pad_mask = op_traj[i].eq(-1)
            tgt_pad_mask = rel_traj[i].eq(-1)
            tgt_pad_mask[:, 0] = False
            
            pad_masks = (init_pad_mask, src_pad_mask, tgt_pad_mask)
            ls = self.forward(src, pad_masks, *(t[i] for t in tgts))
            
            for j, (l, t) in enumerate(zip(ls, tgts)):
                loss = self.xent(l, t[i])
                tot[j] += torch.sum(loss)
                num[j] += torch.numel(loss)
            
            init_pad_mask = src_pad_mask
            
        # self._record([t / n for t, n in zip(tot, num)], step)
        return sum((t / n) for t, n in zip(tot, num))
    
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
            tgts = tuple(map(lambda x: x.to(self.device), tgts))
            
            self.train()
            loss = self.traj_loss(root, *tgts, step=step)
            # log({'train/loss': loss}, step=step)
            
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
