import os
import copy
import json
import wandb
import logging
import argparse
import time
from functools import wraps

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from const import *
from run import pf_trajectory, apply_edits
from utils import parse_model_id, get_name
from embed import (
    SinusoidalEmbedding,
    LearnedEmbedding,
    DepthEmbedding,
    IdentityEmbedding,
    RotaryEmbedding
)
from trans import (
    TransformerEncoderLayer,
    AdaptiveTransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
    CausalTransformerDecoderLayer,
    CausalTransformerDecoder,
    MultiheadPointer
)
from data import (
    TrajectoryDataset,
    SequenceDataset,
    StratifiedInfiniteSampler,
    InfiniteSampler,
    unsupervised_loader,
    supervised_loader,
    elaborate
)

logging.basicConfig()
logger = logging.getLogger('train')

REMOTE_PREFIX = os.environ.get('REMOTE_PREFIX', '/scratch4/jeisner1')

def log(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f"step {step}: {data}")

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info(f'{f.__name__}: {te-ts}') 
        return result
    return wrap

def xent(logprobs, tgts, ignore=-1):
    loss = torch.sum(logprobs * tgts, dim=-1)
    keep_mask = torch.argmax(tgts, dim=-1) != ignore
    loss = loss * keep_mask
    tot = torch.sum(loss)
    n = torch.sum(keep_mask)
    return -tot, n

class Evolver(nn.Module):
    
    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return model
    
    def __init__(
        self,
        d_model=512,
        dim_feedforward=2048,
        nhead=8,
        dropout=0.1,
        layer_norm_eps=1e-5,
        encoder_layers=6,
        decoder_layers=6,
        num_encoders=1,
        num_decoders=1,
        op_scale=1,
        tok_scale=1,
        idx_scale=1,
        max_len=512,
        vocab_size=VOCAB_SIZE,
        pad_token_id = PAD_TOKEN_ID,
        bos_token_id = BOS_TOKEN_ID,
        eos_token_id = EOS_TOKEN_ID,
        positional_embeddings='sinu',
        static_embeddings=False,
        depth_embeddings=False,
        device='cpu',
        name=None,
        **_,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.max_len = max_len
        self.device = device
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.static_embeddings = static_embeddings
        self.name = name
      
        self.positional_embedding = {
            'sinu': SinusoidalEmbedding,
            'learned': LearnedEmbedding,
            'identity': IdentityEmbedding,
            'rope': RotaryEmbedding
        }[positional_embeddings](d_model=d_model, max_len=max_len)
        self.depth_embedding = DepthEmbedding(d_model=d_model) if depth_embeddings else None
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.op_scale = op_scale
        self.tok_scale = tok_scale
        self.idx_scale = idx_scale
        
        codec_params = {
            'd_model': d_model,
            'nhead': nhead,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps
        }
        
        encoder_layer = AdaptiveTransformerEncoderLayer if depth_embeddings else TransformerEncoderLayer
        encoder_layer = encoder_layer(**codec_params)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.encoder_list = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(num_encoders)])
        
        decoder_layer = CausalTransformerDecoderLayer(**codec_params)
        self.decoder = CausalTransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_list = nn.ModuleList([copy.deepcopy(self.decoder) for _ in range(num_decoders)])
       
        self.op_head = nn.Linear(d_model, 5)
        self.tok_head = nn.Linear(d_model, self.vocab_size)
        self.tok_head.weight = self.token_embedding.weight # tie weights?
        self.idx_head = nn.Linear(d_model, self.max_len)
       
    def compute_tgt_static(self, input_ids, edit_tgts, *_):
        if len(input_ids.shape) == 1: input_ids = input_ids.unsqueeze(0)
        op_ids, tok_ids, idx_ids = tuple(map(lambda x: torch.argmax(x, dim=-1), edit_tgts))
        B, N = op_ids.shape
        
        output_ids = torch.zeros(B, N, device=self.device, dtype=torch.long)
        
        ins_mask = op_ids.eq(INS_ID) | op_ids.eq(SUB_ID)
        output_ids[ins_mask] = tok_ids[ins_mask]
        
        cpy_mask = op_ids.eq(CPY_ID)
        output_ids[cpy_mask] = input_ids[torch.arange(B, device=self.device).unsqueeze(1), idx_ids][cpy_mask]
        
        eos_mask = op_ids.eq(EOS_ID)
        output_ids[eos_mask] = self.eos_token_id
        
        src, _ = self.get_src(output_ids)
        return src
        
    def compute_tgt(self, input_ids, edit_tgts, memory):
        if len(input_ids.shape) == 1: input_ids = input_ids.unsqueeze(0).expand(B, -1)
        op_ids, tok_ids, idx_ids = tuple(map(lambda x: torch.argmax(x, dim=-1), edit_tgts))
        B, N = op_ids.shape
        
        tgt = torch.zeros(B, N, self.d_model, device=self.device)
        memory = self.positional_embedding(memory, d=-1)
        
        permuted_memory = memory[torch.arange(B, device=self.device).unsqueeze(1), idx_ids]
        permuted_input_ids = input_ids[torch.arange(B, device=self.device).unsqueeze(1), idx_ids]
        
        ins_mask = op_ids.eq(INS_ID)
        if torch.any(ins_mask):
            tgt[ins_mask] = self.token_embedding(tok_ids[ins_mask]) * np.sqrt(self.d_model)
            
        cpy_mask = op_ids.eq(CPY_ID)
        if torch.any(cpy_mask):
            cpy_mask = cpy_mask.unsqueeze(-1).expand_as(tgt)
            tgt[cpy_mask] = permuted_memory[cpy_mask]
        
        sub_mask = op_ids.eq(SUB_ID)
        if torch.any(sub_mask):
            old_embeds = self.token_embedding(permuted_input_ids[sub_mask]) * np.sqrt(self.d_model)
            new_embeds = self.token_embedding(tok_ids[sub_mask]) * np.sqrt(self.d_model)
            tgt[sub_mask] = permuted_memory[sub_mask] - old_embeds + new_embeds
        
        eos_mask = op_ids.eq(EOS_ID)
        if torch.any(eos_mask):
            tgt[eos_mask] = self.token_embedding.weight[self.eos_token_id] * np.sqrt(self.d_model)
        
        tgt = self.positional_embedding(tgt, d=1)
        return tgt
    
    def get_src(self, x):
        pad_mask = x.eq(self.pad_token_id)
        x = self.token_embedding(x) * np.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x, pad_mask
   
    def _get_probs(self, edit_logits, pad_mask):
        *_, idx_logits = edit_logits
        idx_logits[pad_mask.unsqueeze(1).expand_as(idx_logits)] = -1e9
        return tuple(map(lambda x: F.log_softmax(x, dim=-1), edit_logits))
    
    def _get_codec(self, t):
        if t is None: return self.encoder, self.decoder
        return self.encoder_list[t % len(self.encoder_list)], self.decoder_list[t % len(self.decoder_list)]
  
    def forward(
        self, input_ids, edit_tgts,
        src=None, t=None,
        memory=None, cache=None
    ):
        B, _ = input_ids.shape
        
        if self.training and memory is not None: raise Exception()
        if self.training and cache is not None: raise Exception()
        
        _src, pad_mask = self.get_src(input_ids)
        src = src or _src
        
        encoder, decoder = self._get_codec(t)
        depth_embed = self.depth_embedding(torch.full((B,), t, device=self.device)) if self.depth_embedding else None
        memory = memory or encoder(src, depth_embed=depth_embed, src_key_padding_mask=pad_mask)
        tgt = self.compute_tgt_static(input_ids, edit_tgts) if self.static_embeddings else self.compute_tgt(input_ids, edit_tgts, memory)
        output, _, cache = decoder(tgt, memory, cache=cache, memory_key_padding_mask=pad_mask)
        
        op_logits = self.op_head(output)
        tok_logits = self.tok_head(output)
        idx_logits = self.idx_head(output)
        
        probs = self._get_probs((op_logits, tok_logits, idx_logits), pad_mask)
        return probs, tgt, memory, cache
   
    def loss(self, edit_probs, edit_tgts):
        return sum((
            xent(p[:, :-1], t[:, 1:], ignore=i)
            for p, t, i in zip(edit_probs, edit_tgts, [PAD_ID, PAD_TOKEN_ID, 0]
        )), ())

    def step(self, batch, step=None):
        traj_input_ids, traj_edit_tgts = batch
        _, T, _ = traj_input_ids.shape
        tot, n = [0, 0, 0], [0, 0, 0]
       
        src = None 
        for t in range(T-1):
            input_ids = traj_input_ids[:, t]
            edit_tgts = tuple(map(lambda x: x[:, t], traj_edit_tgts))
            edit_probs, src, *_ = self.forward(input_ids, edit_tgts, src, t)
          
            loss = self.loss(edit_probs, edit_tgts)
            for i in range(3):
                tot[i] += loss[2*i]
                n[i] += loss[2*i+1]
      
        log({
            'train/per_occ_op_loss': tot[0] / n[0],
            'train/per_occ_tok_loss': tot[1] / n[1],
            'train/per_occ_idx_loss': tot[2] / n[2],
        }, step=step)
        
        traj_loss = (
            (tot[0] / n[0] * self.op_scale)  +
            (tot[1] / n[1] * self.tok_scale) +
            (tot[2] / n[2] * self.idx_scale)
        )
    
        return traj_loss
    
    def prepare_batch(self, batch, step, pf_params):
        traj_input_ids, _, traj_edit_tgts, _ = batch
        traj_input_ids = traj_input_ids.to(self.device)
        
        if traj_edit_tgts is not None:
            traj_edit_tgts = tuple(map(lambda x: x.to(self.device), traj_edit_tgts))
        else:
            s = time.time()
            self.eval()
            traj_edit_tgts, _ = pf_trajectory(self, traj_input_ids, **pf_params)
            log({'train/e_time': time.time()-s}, step=step)
            
        return traj_input_ids, traj_edit_tgts
    
    def _ll(self, traj_input_ids):
        return pf_trajectory(self, traj_input_ids, num_particles=1, temperature=0.5)[1]
    
    def run_eval(self, eval_loader, eval_steps):
        return elbo(self, eval_loader, eval_steps) 
    
class PointerStyleEvolver(Evolver):
    '''
    changes:
    - use context embedding, decoder input, and decoder output to predict edits (3*d_model)
    - use cross attention weights to construct index distribution
    
    things staying the same:
    - predict a special EOS op to terminate generation (to keep interfaces consistent)
    - uses dynamic input embeddings (but preserves ability to ablate with compute_tgt_static)
    - "learns to point" i.e. explicit supervision over the correct operation rather than a mixture distribution
    - learns "separate" distributions for op and tok/idx, i.e. p(z|x) * p(y|z,x)
    '''
    
    def __init__(self, pointer_attn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_head = nn.Linear(3*self.d_model, 5)
        self.pointer_attn = pointer_attn
        if self.pointer_attn:
            self.pointer = MultiheadPointer(
                embed_dim=self.d_model,
                num_heads=self.nhead,
                dropout=self.dropout,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False
            )
           
        # NOTE -- cheating again, revert 
        self.idx_ffn = nn.Linear(self.d_model, self.max_len)
    
    def _to_idx_logits(self, attn_weights, eps=1e-7):
        return torch.log(torch.clamp(attn_weights, eps, 1-eps))
    
    def _to_op_logits(self, attn_weights, mem, tgt, h):
        if not self.training:
            tgt = tgt[:, -1:]
            h = h[:, -1:]
            
        c = torch.bmm(attn_weights, mem)
        return self.op_head(torch.cat([c, tgt, h], dim=-1))
        
    def forward(self, input_ids, edit_tgts, src=None, t=None, mem=None, cache=None):
        '''
        in a sense this implements the evolver interface by returning a distribution over op/tok/idx
        ''' 
        
        if self.training and mem is not None: raise Exception() 
        if self.training and cache is not None: raise Exception()
        
        _src, src_pad_mask = self.get_src(input_ids)
        src = _src if src is None else src
       
        encoder, decoder = self._get_codec(t)
        mem = encoder(src, src_key_padding_mask=src_pad_mask) if mem is None else mem
        tgt = self.compute_tgt(input_ids, edit_tgts, mem)
        h, (*_, attn_weights), cache = decoder(tgt, mem, cache=cache, memory_key_padding_mask=src_pad_mask)

        op_logits = self._to_op_logits(attn_weights, mem, tgt, h)
        tok_logits = self.tok_head(h)
        
        # what if we just cheated?
        # idea: should just be able to look at the positional embedding subspace
        # cheating_tgt = self.positional_embedding(torch.zeros_like(tgt, device=self.device), d=1)
        # idx_logits = self.idx_ffn(cheating_tgt)
        idx_weights = self.pointer(tgt, mem, key_padding_mask=src_pad_mask) if self.pointer_attn else attn_weights
        idx_logits = self._to_idx_logits(idx_weights)
        
        probs =  (
            F.log_softmax(op_logits, dim=-1),
            F.log_softmax(tok_logits, dim=-1),
            F.log_softmax(idx_logits, dim=-1)
        )
        
        return probs, tgt, mem, cache
    
class PGNStyleEvolver(PointerStyleEvolver):
    '''
    changes:
    - inherits changes from PointerStyleEvolver relative to baseline
    - trained as a mixture distribution between vocab and previous seq tokens, i.e. p * ins + (1-p) * cpy
    - can only be supervised with INS/CPY ops and treats EOS as INS(eos) or CPY(n)
    - in the static embedding case, this is equivalent to an iterated PGN. otherwise, we use a pooling op over prev. occurrences
    
    problem:
    - pooling should probably be weighted over the xattn weights
    '''
    
    def compute_tgt(self, input_ids, edit_tgts, mem):
        # we only use edit_tgts here to reconstruct output_ids
        # this isn't perfect but it's a simple temp solution to use the same loader
        output_ids = apply_edits(input_ids, edit_tgts)
        
        # to keep training fast, we just apply mean-pooling over previous occurrences
        # a more principled approach would be to use idx_weights?
        mem = self.positional_embedding(mem, d=-1)
        mask = input_ids[:, :, None] == output_ids[:, None, :] # (B, N_in, N_out)
        counts = mask.sum(1) # (B, N_out)
    
        tgt = (mem.unsqueeze(2) * mask.unsqueeze(-1)).sum(1) # (B, N_in, N_out, D) -> (B, N_out, D)
        tgt = tgt / (counts.unsqueeze(-1) + 1e-10)
        
        ins_mask = counts.eq(0)
        if torch.any(ins_mask):
            tgt[ins_mask] = self.token_embedding(output_ids[counts == 0]) * np.sqrt(self.d_model)
            
        tgt = self.positional_embedding(tgt, d=1)
        return tgt
    
    def forward(self, input_ids, edit_tgts, src=None, t=None, mem=None, cache=None):
        pass
    
class Transformer(nn.Module):
    '''
    standard transformer used either as:
    - a decoder-only autoregressive model for one-step generation
    - an encoder-decoder seq2seq model trained on denoising pairs (x_t, x_{t+1})
    '''
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        encoder_layers=6,
        decoder_layers=6,
        max_len=512,
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_TOKEN_ID,
        device='cpu',
        name=None,
        **_
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.device = device
        self.name = name
      
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len)
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.tok_head = nn.Linear(self.d_model, self.vocab_size)
        self.tok_head.weight = self.token_embedding.weight # weight tying
        
        self.codec_params = {
            'd_model': d_model,
            'nhead': nhead,
            'dropout': dropout,
            'dim_feedforward': dim_feedforward,
            'batch_first': True,
        }
     
        encoder_layer = nn.TransformerEncoderLayer(**self.codec_params)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(**self.codec_params)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        
    def embed(self, x):
        pad_mask = x.eq(self.pad_token_id)
        x = self.token_embedding(x) * np.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x, pad_mask
    
    def forward(self, input_ids, output_ids):
        causal_mask = T.generate_square_subsequent_mask(self.max_len, self.device, dtype=torch.bool)
        
        if self.decoder_layers > 0:
            src, src_pad_mask = self.embed(input_ids)
            tgt, tgt_pad_mask = self.embed(output_ids)
            
            h = self.encoder(src, src_key_padding_mask=src_pad_mask)
        
            h = self.decoder(
                tgt, h,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask
            )
            
        else:
            src, src_pad_mask = self.embed(output_ids)
            
            h = self.encoder(
                src,
                is_causal=True,
                mask=causal_mask,
                src_key_padding_mask=src_pad_mask
            )
        
        tok_logits = self.tok_head(h)
        tok_probs = F.log_softmax(tok_logits, dim=-1)
        
        return tok_probs
    
    def prepare_batch(self, batch, *_):
        input_ids, output_ids = batch
        if self.decoder_layers > 0:
            return input_ids.to(self.device), output_ids.to(self.device)
        return None, output_ids.to(self.device)
    
    def step(self, inputs, _):
        input_ids, output_ids = inputs
        tok_probs = self.forward(input_ids, output_ids)
        loss, n = xent(
            tok_probs[:, :-1],
            F.one_hot(output_ids[:, 1:], num_classes=self.vocab_size),
            ignore=self.pad_token_id
        )
        return loss / n
    
    def _ll(self, traj_input_ids):
        tl = 0
        for t in range(traj_input_ids.shape[1]-1):
            input_ids = traj_input_ids[:, t]
            output_ids = traj_input_ids[:, t+1]
            tok_probs = self.forward(input_ids, output_ids)
            ll = xent(
                tok_probs[:, :-1],
                F.one_hot(output_ids[:, 1:], num_classes=self.vocab_size),
                ignore=self.pad_token_id
            )[0]
            tl -= ll
        return tl
    
    def run_eval(self, eval_loader, eval_steps):
        if self.decoder_layers > 0: return elbo(self, eval_loader, eval_steps)
        
        tot_loss = 0
        tot_n = 0
        for step, batch in enumerate(eval_loader):
            if step >= eval_steps: break
            input_ids, output_ids = self.prepare_batch(batch)
            tok_probs = self.forward(input_ids, output_ids)
            loss, n = xent(
                tok_probs[:, :-1],
                F.one_hot(output_ids[:, 1:], num_classes=self.vocab_size),
                ignore=self.pad_token_id
            )
            tot_loss += loss
            tot_n += n
        return tot_loss / tot_n
    
class DenoisingTransformer(Transformer):
    '''
    Standard autoregressive denoising transformer, except we build in explicit
    "message passing" through cross-attention over the previous timestep's sequence
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.encoder_layers == 0:
            raise Exception('denoising transformer requires > 0 encoder layers')
        
        # note the name mismatch here
        encoder_layer = TransformerDecoderLayer(**self.codec_params)
        self.encoder = TransformerDecoder(encoder_layer, num_layers=self.encoder_layers)
        
    def forward(
        self, input_ids, output_ids,
        prev_mem=None, prev_mask=None,
    ):
        B, N = input_ids.shape
        src, src_pad_mask = self.embed(input_ids)
        tgt, tgt_pad_mask = self.embed(output_ids)
        
        mem = self.encoder(
            src, prev_mem,
            tgt_key_padding_mask=src_pad_mask,
            memory_key_padding_mask=prev_mask
        )[0]
        
        h = self.decoder(
            tgt, mem,
            tgt_mask=T.generate_square_subsequent_mask(N, device=self.device, dtype=torch.bool),
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        
        l = self.tok_head(h)
        l = F.log_softmax(l, dim=-1)
        
        return l, mem, src_pad_mask
    
    def prepare_batch(self, batch, *_):
        traj_input_ids, *_ = batch
        return traj_input_ids.to(self.device)
    
    def step(self, batch, reduce=True):
        traj_input_ids = batch
        
        traj_loss = 0
        prev_mem = None
        prev_mask = None
        
        for t in range(traj_input_ids.shape[1]-1):
            input_ids = traj_input_ids[:, t]
            output_ids = traj_input_ids[:, t+1]
            
            tok_probs, prev_mem, prev_mask = self.forward(
                input_ids, output_ids,
                # prev_mem, prev_mask
                # NOTE -- experiment
                None, None
            )
            
            loss = xent(
                tok_probs[:, :-1],
                F.one_hot(output_ids[:, 1:], num_classes=self.vocab_size),
                ignore=self.pad_token_id
            )[0]
            
            traj_loss += loss
            
        if reduce:
            N = torch.sum(~(traj_input_ids.eq(self.pad_token_id)[:, 1:, :]))
            return traj_loss / N
        
        return traj_loss
    
    def _ll(self, traj_input_ids):
        return -self.step(traj_input_ids, reduce=False)
    
    def run_eval(self, eval_loader, eval_steps):
        return elbo(self, eval_loader, eval_steps)
            
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)
    logger.info(f'RUNNING ON: {gpu_properties.name}')

def get_memory():
    return torch.cuda.memory_allocated() / 1024**2

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
    
def checkpoint_model(model, optim, lr_scheduler, step):
    save_path = REMOTE_PREFIX if model.device == 'cuda' else '.'
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'wandb_run_id': None if wandb.run is None else wandb.run.id
    }, f'{save_path}/checkpoints/{model.name}-{(step or -1)+1}.pt')

@torch.no_grad()
def elbo(model, eval_loader, eval_steps):
    loss = 0
    toks = 0
    for step, (traj_input_ids, post, _, n) in enumerate(eval_loader):
        if step >= eval_steps: break
        traj_input_ids = traj_input_ids.to(model.device)
        post = post.to(model.device)
        ll = model._ll(traj_input_ids)
        loss += torch.sum(ll - post)
        toks += n
    return loss / toks

def train(
    model, optim, lr_scheduler, train_loader, eval_loader,
    train_steps, eval_steps, grad_accum_steps, clip_gradients,
    checkpoint_at, eval_at, pf_params, start_step=0
):
    '''
    model needs to implement:
    - prepare_batch(self, batch, step, pf_params)
    - step(self, inputs, step)
    - run_eval(self, eval_loader, eval_steps)
    '''
    
    for step, batch in tqdm(
        enumerate(train_loader, start=start_step),
        total=train_steps,
        disable=wandb.run is None
    ):
        if step >= train_steps: break
        
        batch = model.prepare_batch(batch, step, pf_params)
        model.train() 
        loss = model.step(batch, step)
        loss.backward()

        if step % grad_accum_steps == 0:
            optim.step()
            optim.zero_grad()

        if lr_scheduler:
            lr_scheduler.step()

        if (step + 1) % checkpoint_at == 0:
            checkpoint_model(model, optim, lr_scheduler, step)

        if (step + 1) % eval_at == 0:
            s = time.time()
            model.eval()
            eval_loss = model.run_eval(eval_loader, eval_steps)
            log({'eval/loss': eval_loss, 'eval/time': time.time() - s}, step=step)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--from-checkpoint', default=None)
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()

def init_run(name, config):
    model = \
        Transformer if name.startswith('ar') \
        else DenoisingTransformer if name.startswith('den') \
        else PointerStyleEvolver if name.startswith('ps') \
        else PGNStyleEvolver if name.startswith('pgn') \
        else Evolver
        
    logger.info(f'using {model}')
    model = model(
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        num_encoders=config.get('num_encoders', 1),
        num_decoders=config.get('num_decoders', 1),
        max_len=config['max_len'],
        op_scale=config.get('op_scale', 1),
        tok_scale=config.get('tok_scale', 1),
        idx_scale=config.get('idx_scale', 1),
        positional_embeddings=config.get('positional_embeddings', 'sinu'),
        static_embeddings=config.get('static_embeddings', False),
        depth_embeddings=config.get('depth_embeddings', False),
        pointer_attn=config.get('pointer_attn', False),
        device=config['device'],
        name=name
    ).to(config['device'])
    
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

    if not config['local']:
        wandb.init(
            id=wandb_run_id,
            project='evolver',
            name=name,
            config=config,
            resume='allow',
            notes=config.get('notes', 'N/A')
        )
    
    return model, optim, lr_scheduler, start_step

def init_loaders(name, config):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sampler = StratifiedInfiniteSampler if 'num_epochs' not in config else InfiniteSampler
    logger.info(f'using {sampler} in loader')
    kwargs = {'max_len': config['max_len'], 'tokenizer': tokenizer, 'batch_size': config['batch_size'], 'sampler': sampler}

    if name.startswith('ar'):
        logger.info('using ar loaders')
        train_dataset = SequenceDataset.from_trajectories(path=config['train'], denoising=name.startswith('ar-d'), **kwargs)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler(train_dataset, config['batch_size']))
        eval_loader = (
            unsupervised_loader(path=config['eval'], **kwargs)
            if name.startswith('ar-d') else
            DataLoader(SequenceDataset.from_trajectories(path=config['eval'], denoising=False, **kwargs), batch_size=config['batch_size'], shuffle=True)
        )
    elif 'unsup' in name or name.startswith('den'):
        logger.info('using unsupervised loaders')
        train_loader = unsupervised_loader(path=config['train'], **kwargs)
        eval_loader = unsupervised_loader(path=config['eval'], **kwargs)
    else:
        cache_prefix = config.get('cache_prefix', name.split('.')[0].split('_')[0])
        logger.info(f'using supervised loaders with cache prefix {cache_prefix}')
        train_loader = supervised_loader(path=config['train'], cache_prefix=cache_prefix, all_tokens=config.get('all_tokens', True), **kwargs)
        eval_loader = unsupervised_loader(path=config['eval'], **kwargs)

    return train_loader, eval_loader

def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level))
    
    with open(args.config, 'r') as f: config = json.load(f)
    config['device'] = args.device
    config['local'] = args.local
    
    prefix = parse_model_id(args.config)
    name = get_name(args.config)
    
    train_loader, eval_loader = init_loaders(name, config)
    train_steps = config.get('train_steps', 0)
    if train_steps == 0:
        num_samples = len(train_loader.dataset)
        train_steps = config.get('num_epochs', 1) * num_samples // config['batch_size']
    config['train_steps']  = train_steps
    logger.info(f'starting run for {train_steps} steps')
    
    if config['eval_at'] < 1:
        config['eval_at'] = int(train_steps * config['eval_at'])
    logger.info(f'eval every {config["eval_at"]} steps')
        
    if config['checkpoint_at'] < 1:
        config['checkpoint_at'] = int(train_steps * config['checkpoint_at'])
    logger.info(f'checkpoint every {config["checkpoint_at"]} steps')
    
    model, optim, lr_scheduler, start_step = init_run(name, config)
   
    pf_params = None if 'unsup' not in prefix else {
        'num_particles': config['num_particles'],
        'threshold': config['threshold'],
        'temperature': config['temperature'],
        'resample_at': config['resample_at'],
    }
    
    train(
        model, optim, lr_scheduler,
        train_loader, eval_loader,
        train_steps=train_steps,
        eval_steps=config['eval_steps'],
        grad_accum_steps=config['grad_accum_steps'],
        clip_gradients=config['clip_gradients'],
        checkpoint_at=config['checkpoint_at'],
        eval_at=config['eval_at'],
        pf_params=pf_params,
        start_step=start_step
    )
    
    checkpoint_model(model, optim, lr_scheduler, None)

if __name__ == '__main__':
    main()

### broken/deprecated ATM

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
        if input_is_tgt: tot_loss, n = model.step(output_ids)
        else: tot_loss, n = model.step(input_ids, output_ids)
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

@torch.no_grad()
def ar_elbo(model, eval_loader: TrajectoryDataset, eval_steps, device):
    model.eval()
    tot_loss = 0
    tot_n = 0
    
    for step, (traj_input_ids, log_posterior, _, n) in enumerate(eval_loader):
        if step >= eval_steps: break
        
        traj_input_ids = traj_input_ids.to(device)
        log_posterior = log_posterior.to(device)
        
        # log_likelihood = traj_likelihood(model, traj_input_ids)
        ll = model.step(traj_input_ids)
        tot_loss += ll - torch.sum(log_posterior)
        tot_n += n
       
    return tot_loss / tot_n

def traj_likelihood(model, traj_input_ids): 
    _, T, _ = traj_input_ids.shape
    tot_loss = tot_n = 0
    
    for i in range(T-1):
        input_ids = traj_input_ids[:, i]
        output_ids = traj_input_ids[:, i+1]
        
        loss, _ = model.loss(input_ids, output_ids)
        tot_loss += loss
        
    return -tot_loss

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
