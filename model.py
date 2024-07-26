import wandb
import logging
from time import time
from functools import wraps

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T

from constants import *
from embedding import *
from transformer import *

logging.basicConfig()
logger = logging.getLogger('train')

def log(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: print(f"Step {step}: {data}")

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'{f.__name__}: {te-ts}') 
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
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        encoder_layers=6,
        decoder_layers=6,
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
        **_,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.max_len = max_len
        self.device = device
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.static_embeddings = static_embeddings
      
        PositionalEmbedding = SinusoidalEmbedding if positional_embeddings == 'sinu' else LearnedEmbedding
        self.positional_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.depth_embedding = DepthEmbedding(d_model=d_model) if depth_embeddings else None
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.op_scale = op_scale
        self.tok_scale = tok_scale
        self.idx_scale = idx_scale
        
        transformer_params = {
            'd_model': d_model,
            'nhead': nhead,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'batch_first': True
        }
        
        EncoderLayer = AdaptiveTransformerEncoderLayer if depth_embeddings else nn.TransformerEncoderLayer
        encoder_layer = EncoderLayer(**transformer_params)
        decoder_layer = CausalTransformerDecoderLayer(**transformer_params)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = CausalTransformerDecoder(decoder_layer, num_layers=decoder_layers)
       
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
            ins_embeds = self.token_embedding(tok_ids[ins_mask]) * np.sqrt(self.d_model)
            tgt[ins_mask] = ins_embeds
            
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
  
    def forward(
        self, input_ids, edit_tgts,
        src=None, t=None,
        memory=None, cache=None
    ):
        B, N = input_ids.shape 
        
        if self.training and memory is not None: raise Exception()
        if self.training and cache is not None: raise Exception()
        
        src_0, pad_mask = self.get_src(input_ids)
        src = src_0 if src is None else src
       
        depth_embed = self.depth_embedding(torch.full((B,), t, device=self.device)) if self.depth_embedding else None
        memory = self.encoder(src, depth_embed=depth_embed, src_key_padding_mask=pad_mask) if memory is None else memory
        tgt = self.compute_tgt_static(input_ids, edit_tgts) if self.static_embeddings else self.compute_tgt(input_ids, edit_tgts, memory)
        output, cache = self.decoder(tgt, memory, cache=cache, memory_key_padding_mask=pad_mask)
        
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

    def traj_loss(self, traj_input_ids, traj_edit_tgts, step=None):
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
    
        N = torch.sum(~(traj_input_ids.eq(PAD_TOKEN_ID)[:, 1:, :])) 
        return traj_loss, tot[0] / N, tot[1] / N, tot[2] / N

class NoShareEvolver(Evolver):

    def __init__(self, *args, num_encoders=1, num_decoders=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(num_encoders)])
        self.decoders = nn.ModuleList([copy.deepcopy(self.decoder) for _ in range(num_decoders)])

    def forward(
        self, input_ids, edit_tgts,
        src=None, t=None,
        memory=None, cache=None
    ):
        B, N = input_ids.shape
        
        if self.training and memory is not None: raise Exception()
        if self.training and cache is not None: raise Exception()
        if t is None: raise Exception('need depth in no share evolver')

        src_0, pad_mask = self.get_src(input_ids)
        src = src_0 if src is None else src

        cur_encoder = self.encoders[t % self.num_encoders]
        cur_decoder = self.decoders[t % self.num_decoders]

        memory = cur_encoder(src, depth_embed=None, src_key_padding_mask=pad_mask) if memory is None else memory
        tgt = self.compute_tgt_static(input_ids, edit_tgts) if self.static_embeddings else self.compute_tgt(input_ids, edit_tgts, memory)
        output, cache = cur_decoder(tgt, memory, cache=cache, memory_key_padding_mask=pad_mask)

        op_logits = self.op_head(output)
        tok_logits = self.tok_head(output)
        idx_logits = self.idx_head(output)

        probs = self._get_probs((op_logits, tok_logits, idx_logits), pad_mask)
        return probs, tgt, memory, cache
    
class Transformer(nn.Module):
    
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
        device='cpu', **_
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
      
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len)
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.tok_head = nn.Linear(self.d_model, self.vocab_size)
        self.tok_head.weight = self.token_embedding.weight # weight tying
        
        transformer_parameters = {
            'd_model': d_model,
            'nhead': nhead,
            'dropout': dropout,
            'dim_feedforward': dim_feedforward,
            'batch_first': True,
        }
     
        encoder_layer = nn.TransformerEncoderLayer(**transformer_parameters)
        decoder_layer = nn.TransformerDecoderLayer(**transformer_parameters)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        
    def get_src(self, x):
        pad_mask = x.eq(self.pad_token_id)
        x = self.token_embedding(x) * np.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x, pad_mask
    
    def forward(
        self,
        src, tgt,
        src_pad_mask,
        tgt_pad_mask,
    ):
        if self.encoder_layers == 0 and src is not None:
            raise Exception('src found when encoder_layers == 0')
        
        causal_mask = T.generate_square_subsequent_mask(self.max_len, self.device).eq(-torch.inf)
       
        output = self.encoder(
            src,
            is_causal=(self.decoder_layers == 0),
            mask=None if (self.decoder_layers > 0) else causal_mask,
            src_key_padding_mask=src_pad_mask
        )
        
        if self.decoder_layers > 0:
            output = self.decoder(
                tgt, output,
                tgt_is_causal=True,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_is_causal=False,
                memory_mask=None,
                memory_key_padding_mask=src_pad_mask
            )
        
        tok_logits = self.tok_head(output)
        tok_probs = F.log_softmax(tok_logits, dim=-1)
        return tok_probs
    
    def loss(self, input_ids, output_ids=None):
        src, src_pad_mask = self.get_src(input_ids)
        
        if output_ids is None:
            tok_probs = self.forward(src, None, src_pad_mask, None)
            labels = F.one_hot(input_ids[:, 1:], num_classes=self.vocab_size)
        
        else:
            tgt, tgt_pad_mask = self.get_src(output_ids)
            tok_probs = self.forward(src, tgt, src_pad_mask, tgt_pad_mask)
            labels = F.one_hot(output_ids[:, 1:], num_classes=self.vocab_size)
       
        return xent(tok_probs[:, :-1], labels, ignore=self.pad_token_id)
