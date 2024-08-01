import os
import json
import time
import math
import pickle
import logging
import argparse
from datetime import datetime
from collections import defaultdict

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import Transformer as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import SinusoidalEmbedding
from data import StratifiedInfiniteSampler

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
    
def save_model(model, step, optim):
    path = f'{REMOTE_PREFIX}/checkpoints' if model.device == 'cuda' else 'checkpoints'
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'wandb_run_id': None if wandb.run is None else wandb.run.id
    }, f'{path}/{model.name}-{step+1}.pt')
    
def _replace(t, a, b):
    return torch.where(t == a, b, t)

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
        device='cuda' if torch.cuda.is_available() else 'cpu',
        *_
    ):
        super().__init__()
        
        self.N = N
        self.d_model = d_model
        self.name = name
        self.device = device
        self.rel_offset = tok_v
        self.pos_offset = tok_v + rel_v
        self.vocab_size = tok_v + rel_v + pos_v + 1 # add one for UNK
        self.causal_mask = T.generate_square_subsequent_mask(N, dtype=torch.bool, device=device)
        
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
        self.cpy_head = nn.Linear(d_model, N)
        self.par_head = nn.Linear(d_model, N)
        self.v_head = nn.Linear(d_model, tok_v+rel_v+pos_v)
       
        self.done = nn.Parameter(torch.zeros(d_model))
        self.plh = nn.Parameter(torch.zeros(d_model))
        self.root_bos = nn.Parameter(torch.zeros(d_model))
        self.root_eos = nn.Parameter(torch.zeros(d_model))
        self.init_params()
        
    def get_loader(self, path):
        dataset = ParseDataset.from_pkl(path) 
        return DataLoader(
            dataset,
            batch_size=1,
            sampler=StratifiedInfiniteSampler(dataset, 1),
            collate_fn=collate_single,
        )
    
    def init_params(self):
        for param in [self.done, self.plh, self.root_bos, self.root_eos]:
            nn.init.trunc_normal_(param, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        
    def root(self, tok, rel, pos):
        root = (self.embedding.weight[tok] + self.embedding.weight[rel] + self.embedding.weight[pos]).unsqueeze(0)
        embeds = torch.cat([self.root_bos.unsqueeze(0), root, self.root_eos.unsqueeze(0)], dim=0)
        embeds = torch.cat([embeds, torch.full((self.N-3, self.d_model), 0, device=self.device)])
        return embeds.unsqueeze(0)
    
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
        B, N, _ = mem.shape 
        
        if torch.any(tgt_par >= N): raise Exception("tgt_par has illegal indices")
        
        permuted_mem = mem[
            torch.arange(B, device=mem.device).unsqueeze(1),
            torch.where(tgt_par == -1, 0, tgt_par)
        ]
        
        tgt = mem \
            + torch.where(~tgt_par.eq(-1).unsqueeze(-1).expand_as(mem), permuted_mem, 0)
        
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
        prefix = 'train'
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
   
    def traj_loss(self, root, tgts, step=None, reduce=True):
        T, _, B, N = tgts.shape
        
        tot = [0 for _ in range(6)]
        num = [0 for _ in range(6)]
        
        src = self.root(*root)
        init_pad_mask = torch.full((B, N), True, device=self.device)
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
        if self.training: self._record(loss, step)
        return sum(loss) if reduce else sum(tot)
    
    def _eval(self, eval_loader, eval_steps):
        self.eval()
      
        tot = n = 0
        for step, (root, tgts) in enumerate(eval_loader):
            if step >= eval_steps: break
            tgts = tgts.to(self.device)
            tot += self.traj_loss(root, tgts, reduce=False)
            n += torch.sum(tgts[-1, 0] != -1)
            
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
                save_model(self, step, optim)
                
            if (step + 1) % eval_at == 0:
                s = time.time()
                loss = self._eval(eval_loader, eval_steps)
                log({'eval/loss': loss, 'eval/time': time.time()-s}, step=step)
   
    ### incomplete inference utilities
    
    def generate_op(self, src, src_pad_mask, prev_len):
        tgt_op = torch.full((1, self.N), -1, device=self.device)
        tgt_op[:, 0] = CPY_ID
        tgt_cpy = torch.full((1, self.N), -1, device=self.device)
        
        for i in tqdm(range(1, self.N)):
            if torch.all(tgt_op[:, i-1] == EOS_ID): break
            l_op, l_cpy, src = self.forward_op(src, tgt_op, tgt_cpy, src_pad_mask, None, None)
            probs_op = F.log_softmax(l_op[:, i-1], dim=-1)
            probs_cpy = F.log_softmax(l_cpy[:, i-1, :prev_len], dim=-1)
            op = torch.multinomial(torch.exp(probs_op), 1)
            cpy = torch.multinomial(torch.exp(probs_cpy), 1)
            tgt_op[:, i] = op.item()
            tgt_cpy[:, i] = cpy.item()
            
        return src, tgt_op, tgt_cpy
    
    def generate_par(self, src, src_pad_mask, tgt_pad_mask, is_orphan):
        tgt_par = torch.full((1, self.N), -1, device=self.device)
        for i in tqdm(1, range(self.N)):
            if not torch.all(is_orphan[:, i]): continue
            l_par, src = self.forward_par(src, tgt_par, None, src_pad_mask, tgt_pad_mask)
            probs_par = F.log_softmax(l_par[:, i-1], dim=-1)
            par = torch.multinomial(torch.exp(probs_par), 1)
            tgt_par[:, i] = par.item()
            
        return src, tgt_par
    
    def generate(self, root, max_depth=5):
        src = self.root(*root)
        prev_len = 3
        src_pad_mask = torch.full((1, self.N), True, device=self.device)
        src_pad_mask[:, :3] = False
        
        for _ in range(max_depth):
            # this is going to be _very_ slow for now
            src, tgt_op, tgt_cpy = self.generate_op(src, src_pad_mask, prev_len)
           
            src_pad_mask = tgt_op.eq(-1)
            src_pad_mask[:, 0] = False
            is_orphan = tgt_op.eq(INS_ID)
            tgt_pad_mask = ~is_orphan
            tgt_pad_mask[:, 0] = False
            
            src, tgt_par = self.generate_par(src, src_pad_mask, tgt_pad_mask, is_orphan)
            
            # TODO -- finish this
    
class SimpleParseDataset(Dataset):
    
    @classmethod
    def from_pkl(cls, path, *args, **kwargs):
        with open(path, 'rb') as f:
            samples = pickle.load(f)
            return cls(samples, *args, **kwargs)
        
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        seq, traj = self.samples[i]
        return seq, tuple(map(lambda x: torch.tensor(x), zip(*traj)))
    
class SimpleDependencyEvolver(nn.Module):
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        encoder_layers=6,
        N=64,
        K=64,
        tok_v=None,
        pos_v=None,
        rel_v=None,
        name='test',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **_
    ):
        super().__init__()
       
        self.d_model = d_model 
        self.dim_feedforward = dim_feedforward
        self.N = N
        self.K = K
        self.name = name
        self.device = device
        
        codec_params = {
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'dropout': dropout,
            'batch_first': True
        }
        
        encoder_layer = nn.TransformerEncoderLayer(**codec_params)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
       
        self.load_vocab(tok_v, pos_v, rel_v)
        self.tok_embedding = nn.Embedding(len(self.tok_v), d_model)
        self.pos_embedding = nn.Embedding(len(self.pos_v), d_model)
        self.rel_embedding = nn.Embedding(len(self.rel_v), d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model, max_len=N)
        
        self.ins_head = nn.Linear(2*d_model, K)
        self.par_head = nn.Linear(d_model, N)
        self.rel_head = nn.Linear(d_model, len(self.rel_v))
        self.pos_head = nn.Linear(d_model, len(self.pos_v))
        self.tok_head = nn.Linear(d_model, len(self.tok_v))
        
        self.encode_par = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.d_model)
        )
        
        self.root_bos = nn.Parameter(torch.zeros(1, d_model))
        self.root_eos = nn.Parameter(torch.zeros(1, d_model))
        self.plh = nn.Parameter(torch.zeros(1, d_model))
        self.init_params()
            
    def init_params(self):
        for param in [self.root_bos, self.root_eos]:
            nn.init.trunc_normal_(param, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        
    def load_vocab(self, tok_v, pos_v, rel_v):
        with open(tok_v, 'r') as f: self.tok_v = {tok.strip(): i for i, tok in enumerate(f.readlines())}
        with open(pos_v, 'r') as f: self.pos_v = {pos.strip(): i for i, pos in enumerate(f.readlines())}
        with open(rel_v, 'r') as f: self.rel_v = {rel.strip(): i for i, rel in enumerate(f.readlines())}
        
        self.detok = {v: k for k, v in self.tok_v.items()}
        self.depos = {v: k for k, v in self.pos_v.items()}
        self.derel = {v: k for k, v in self.rel_v.items()}
        
    def get_loader(self, path):
        dataset = SimpleParseDataset.from_pkl(path)
        return DataLoader(
            dataset,
            batch_size=1,
            sampler=StratifiedInfiniteSampler(dataset, 1),
            pin_memory=True
        )
            
    @property
    def vocab_size(self):
        return len(self.tok_v) + len(self.pos_v) + len(self.rel_v)
 
    @staticmethod
    def lbl_tree(parsed):
        # return "level-by-level" representation of parse tree
        
        children = defaultdict(list)
        for t in parsed:
            if t['head'] is None: return None
            if t['head'] != 0: children[t['head']].append(t['id'])
           
        seqs = []
        inserted = set()
        cur_leaves = [next(t for t in parsed if t['head'] == 0)['id']]
        
        while cur_leaves:
            seq = []
            next_leaves = []
            
            for t in parsed:
                if t['id'] in cur_leaves or t['id'] in inserted:
                    inserted.add(t['id'])
                    seq.append(t)
                    if t['id'] in cur_leaves: next_leaves.extend(children[t['id']])
                    
            seqs.append(seq)
            cur_leaves = next_leaves
            
        return seqs
   
    def get_labels(self, input_tokens, output_tokens, pad=True):
        if len(output_tokens) > self.N-2: return None
        
        # tokens are represented as a tuple of tokens
        # we return a tuple of list[ins], list[par], list[rel], list[pos], list[tok]
        tgt_ins, tgt_par, tgt_rel, tgt_pos, tgt_tok = [], [], [], [] , []
        
        seen = set(t['id'] for t in input_tokens)
        idx = dict((t['id'], i+1) for i, t in enumerate(output_tokens))
        
        tgt_par = [-1] + [-1 if t['id'] in seen else idx[t['head']] for t in output_tokens] + [-1]
        tgt_tok = [-1] + [-1 if t['id'] in seen else self.tok_v[t['form']] for t in output_tokens] + [-1]
        tgt_pos = [-1] + [-1 if t['id'] in seen else self.pos_v[t['upos']] for t in output_tokens] + [-1]
        tgt_rel = [-1] + [-1 if t['id'] in seen else self.rel_v[t['deprel']] for t in output_tokens] + [-1]
        
        tgt_ins = [idx[input_tokens[0]['id']]-1] \
                + [idx[input_tokens[i+1]['id']]-idx[input_tokens[i]['id']]-1 for i in range(len(input_tokens)-1)] \
                + [len(output_tokens)-idx[input_tokens[-1]['id']]]
                
        if pad:
            tgt_par.extend(-1 for _ in range(self.N-len(tgt_par)))
            tgt_tok.extend(-1 for _ in range(self.N-len(tgt_tok)))
            tgt_pos.extend(-1 for _ in range(self.N-len(tgt_pos)))
            tgt_rel.extend(-1 for _ in range(self.N-len(tgt_rel)))
            tgt_ins.extend(0 for _ in range(self.N-len(tgt_ins)-1))
        
        return tgt_ins, tgt_par, tgt_rel, tgt_pos, tgt_tok
    
    def tokenize(self, input_tokens):
        return [(self.tok_v[t['form']], self.pos_v[t['upos']], self.rel_v[t['deprel']]) for t in input_tokens]
    
    def detokenize(self, input_tokens):
        return [(self.detok[a], self.depos[b], self.derel[c]) for (a, b, c) in input_tokens]
                
    def embed(self, input_tokens, pad=True):
        tok, pos, rel = tuple(map(
            lambda x: torch.tensor(x, device=self.device, dtype=torch.long),
            list(zip(*input_tokens))
        ))
        
        assert tok.shape == pos.shape == rel.shape
        seq_embed = self.tok_embedding(tok) + self.pos_embedding(pos) + self.rel_embedding(rel)
        root = torch.cat([self.root_bos, seq_embed, self.root_eos], dim=0)
        if pad: root = torch.cat([root, torch.full((self.N-root.shape[0], self.d_model), 0, device=self.device)], dim=0)
        return root.unsqueeze(0)
    
    def _ins(self, src, src_pad_mask):
        h = self.encoder(src, src_key_padding_mask=src_pad_mask)
        h_cat = torch.cat([h[:, :-1], h[:, 1:]], dim=-1)
        l = self.ins_head(h_cat)
        return h, l
    
    def _next_pad_mask(self, cpy_indices):
        max_non_pad = torch.max(cpy_indices, dim=-1).values.unsqueeze(1)
        return torch.arange(self.N, device=self.device).unsqueeze(0) > max_non_pad

    def _ins_plh(self, h, src_pad_mask, ins_count):
        B, N, _ = h.shape
        assert ins_count.shape == (B, N-1)
        
        res = torch.zeros_like(h, device=self.device)
        h = self.positional_embedding(h, d=-1)
        
        cumsum = torch.cat([torch.zeros(B, 1, device=self.device), torch.cumsum(ins_count, dim=1)], dim=1)
        cpy_indices = torch.where(src_pad_mask, 0, torch.arange(N, device=self.device).unsqueeze(0) + cumsum)
        if torch.any(cpy_indices[src_pad_mask] >= N): raise Exception('insertion would cause sequence overflow')
        res.scatter_(1, cpy_indices.long().unsqueeze(-1).expand_as(h), h)
         
        plh_mask = torch.ones((B, N), device=self.device, dtype=torch.bool)
        plh_mask.scatter_(1, cpy_indices.long(), False)
        res[plh_mask] = self.plh
        
        res = self.positional_embedding(res, d=1)
        return res, self._next_pad_mask(cpy_indices)
   
    def _par(self, src, src_pad_mask):
        h = self.encoder(src, src_key_padding_mask=src_pad_mask)
        l = self.par_head(h)
        return l
    
    def _add_par(self, h, par_idx):
        par = self.encode_par(h)
        h = h + torch.where((par_idx != -1).unsqueeze(-1).expand_as(h), par, 0)
        return h
    
    def _add_embeds(self, h, embeds):
        return h + self.embedding(_replace(embeds, -1, 0))
    
    def forward(
        self, src, src_pad_mask,
        tgt_ins, tgt_par,
        tgt_rel, tgt_pos, tgt_tok
    ):
        h, l_ins = self._ins(src, src_pad_mask)
        h, src_pad_mask = self._ins_plh(h, src_pad_mask, tgt_ins)
        
        l_par = self._par(h, src_pad_mask)
        h = self._add_par(h, tgt_par)
        
        l_rel = self.rel_head(h)
        h = h + self.rel_embedding(_replace(tgt_rel, -1, 0))
        
        l_pos = self.pos_head(h)
        h = h + self.pos_embedding(_replace(tgt_pos, -1, 0))
        
        l_tok = self.tok_head(h)
        h = h + self.tok_embedding(_replace(tgt_tok, -1, 0))
        
        return (l_ins, l_par, l_rel, l_pos, l_tok), h, src_pad_mask
 
    def step(self, seq, traj, step=None, reduce=True):
        B, T, N = traj[1].shape
        
        tot = [0 for _ in range(5)]
        num = [0 for _ in range(5)]
        
        src = self.embed(seq[0])
        src_pad_mask = torch.full((B, N), True, device=self.device)
        src_pad_mask[:, :3] = False
        
        for i in range(T):
            ls, src, src_pad_mask = self.forward(
                src, src_pad_mask,
                *map(lambda x: x[:, i], traj)
            )
            
            for j, (l, t) in enumerate(zip(ls, traj)):
                loss = self._xent(l, t[:, i])
                tot[j] += torch.sum(loss)
                num[j] += torch.numel(loss)
                
        loss = [(t/n if n > 0 else 0) for t, n in zip(tot, num)]
        if self.training: self._record(loss, step)
        return sum(loss) if reduce else sum(tot)
            
    def _train(
        self, optim, train_loader, eval_loader,
        train_steps, eval_steps, grad_accum_steps,
        checkpoint_at, eval_at, start_step=0
    ):
        for step, (seq, traj) in tqdm(
            enumerate(train_loader, start=start_step),
            total=train_steps
        ):
            if step >= train_steps: break
            traj = tuple(map(lambda x: x.to(self.device), traj))
            
            self.train()
            loss = self.step(seq, traj, step=step)
            log({'train/loss': loss}, step=step)

            loss.backward()
            if step % grad_accum_steps == 0:
                optim.step()            
                optim.zero_grad()
                
            if (step + 1) % checkpoint_at == 0:
                save_model(self, step, optim)
            
            if (step + 1) % eval_at == 0:
                s = time.time()
                loss = self._eval(eval_loader, eval_steps) 
                log({'eval/loss': loss, 'eval/time': time.time()-s}, step=step)
    
    def _eval(self, eval_loader, eval_steps):
        self.eval()
        
        tot = n = 0
        with torch.no_grad():
            for step, (seq, traj) in enumerate(eval_loader):
                if step >= eval_steps: break
                traj = tuple(map(lambda x: x.to(self.device), traj))
                traj_ins, *_ = traj
                tot += self.step(seq, traj, reduce=False)
                # second to last seq length + everything we insert in the last step
                n += len(seq[-1]) + torch.sum(traj_ins[:, -1])
            
        return tot / n
        
    def _xent(self, l, t, ignore=-1):
        mask = t != ignore
        return -F.log_softmax(l[mask], dim=-1).gather(1, t[mask].unsqueeze(1))

    def _record(self, ls, step):
        prefix = 'train'
        log({
            f'{prefix}/ins': ls[0],
            f'{prefix}/par': ls[1],
            f'{prefix}/rel': ls[2],
            f'{prefix}/pos': ls[3],
            f'{prefix}/tok': ls[4]
        }, step=step)
                
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def init_run(config, name, local, device):
    Model = (SimpleDependencyEvolver if name.startswith('simple') else DependencyEvolver)
    logger.info(f'starting run with model: {Model}')
    model = Model(
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config.get('decoder_layers', 0),
        N=config['N'],
        K=config.get('K', 0),
        tok_v=config['tok_v'],
        pos_v=config['pos_v'],
        rel_v=config['rel_v'],
        name=name,
        device=device
    ).to(device)
   
    # enables cpu as fallback for nested tensors in inference
    if device == 'mps': os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = str(1)
    
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
        logger.info(f'starting new run: {name}')
    
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
    
    model, optim, start_step = init_run(config, name, args.local, args.device)
    train_loader = model.get_loader(config['train'])
    eval_loader = model.get_loader(config['eval'])
    
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
