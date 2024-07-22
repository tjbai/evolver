import wandb
import logging
from time import time
from functools import wraps

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T

from constants import (
    VOCAB_SIZE,
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
    INS_ID, CPY_ID, SUB_ID, EOS_ID, PAD_ID
)

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

class SinusoidalEmbedding(nn.Module):
    
    def __init__(self, d_model=512, max_len=10):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10_000) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    
    def forward(self, x, d):
        return x + d * self.pe[:, :x.shape[-2], :]
    
class LearnedSpatialEmbedding(nn.Module):
    
    def __init__(self, d_model=512, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        
        # NOTE -- just initialize to sinusoidal for now
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10_000) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
       
        self.embedding.weight.data.copy_(pe)
        self.embedding.requires_grad_ = False
        
    def forward(self, x, d, pos=None):
        if pos is None:
            B, N, _ = x.shape
            pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        return x + d * self.embedding(pos)
    
class IdentityEmbedding(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x, **kwargs):
        return x
    
class RotaryEmbedding(nn.Module):
    
    def __init__(self):
        pass
    
    def forward(self, x, dir):
        pass
    
class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    
    def forward(
        self,
        tgt, memory,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        if self.training:
            return super().forward(
                tgt, memory,
                tgt_mask=T.generate_square_subsequent_mask(tgt.size(1), tgt.device).eq(-torch.inf),
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            
        tgt_last_tok = tgt[:, -1:, :]

        # self attn
        tmp_tgt = self.self_attn(
            tgt_last_tok, tgt, tgt,
            attn_mask=None, # not needed because we only care about the last token
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # cross attn
        tmp_tgt = self.multihead_attn(
            tgt_last_tok, memory, memory,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
        tgt_last_tok = self.norm2(tgt_last_tok)

        # last ffn
        tmp_tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt_last_tok))))
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        
        return tgt_last_tok
    
class CausalTransformerDecoder(nn.TransformerDecoder):

    def forward(
        self,
        tgt, memory,
        cache=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt

        if self.training:
            for decoder_layer in self.layers:
                x = decoder_layer(
                    x, memory,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                
            return x, None
        
        new_cache = []
        for i, decoder_layer in enumerate(self.layers):
            x = decoder_layer(x, memory)
            new_cache.append(x)
            if cache is not None: x = torch.cat([cache[i], x], dim=1)
            
        if cache is not None: new_cache = torch.cat([cache, torch.stack(new_cache, dim=0)], dim=2)
        else: new_cache = torch.stack(new_cache, dim=0)

        return x, new_cache

class Evolver(nn.Module):
    
    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return model
    
    def __init__(
        self,
        d_model=512, nhead=8, max_len=10,
        dropout=0.1, dim_feedforward=2048,
        encoder_layers=6, decoder_layers=6,
        vocab_size=VOCAB_SIZE,
        op_scale=1, tok_scale=1, idx_scale=1,
        pos_embeddings='sinu',
        static_embeddings=False,
        device='cpu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.max_len = max_len
        self.device = device
        self.vocab_size = vocab_size
        self.static_embeddings = static_embeddings
      
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.positional_embedding = \
            (SinusoidalEmbedding if pos_embeddings == 'sinu' else LearnedSpatialEmbedding)(
                d_model=d_model,
                max_len=max_len
            )
            
        # self.depth_embedding = \
        #     (LearnedSpatialEmbedding if depth_embeddings else IdentityEmbedding)(
        #         d_model=d_model,
        #         max_len=max_depth
        #     )
        
        self.pad_token_id = PAD_TOKEN_ID
        self.bos_token_id = BOS_TOKEN_ID # use [CLS] as BOS
        self.eos_token_id = EOS_TOKEN_ID # use [SEP] as EOS

        self.op_scale = op_scale
        self.tok_scale = tok_scale
        self.idx_scale = idx_scale
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        decoder_layer = CausalTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = CausalTransformerDecoder(decoder_layer, num_layers=decoder_layers)
       
        self.op_head = nn.Linear(d_model, 5)
        self.tok_head = nn.Linear(d_model, self.vocab_size)
        self.tok_head.weight = self.embedding.weight # tie weights
        self.idx_head = nn.Linear(d_model, self.max_len)
       
    # would be equivalent to just calling get_src(output_ids)
    def compute_tgt_static(self, input_ids, edit_tgts, *_):
        if len(input_ids.shape) == 1: input_ids = input_ids.unsqueeze(0)
        op_ids, tok_ids, idx_ids = tuple(map(lambda x: torch.argmax(x, dim=-1), edit_tgts))
        B, _ = op_ids.shape
        
        output_ids = torch.zeros_like(input_ids)
        
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
        # memory = self.depth_embedding(memory, d=-1, pos=depth)
        
        permuted_memory = memory[torch.arange(B, device=self.device).unsqueeze(1), idx_ids]
        permuted_input_ids = input_ids[torch.arange(B, device=self.device).unsqueeze(1), idx_ids]
        # depth = depth[torch.arange(B, device=self.device).unsqueeze(1), idx_ids]
        
        ins_mask = op_ids.eq(INS_ID)
        if torch.any(ins_mask):
            ins_embeds = self.embedding(tok_ids[ins_mask]) * np.sqrt(self.d_model)
            tgt[ins_mask] = ins_embeds
            
        cpy_mask = op_ids.eq(CPY_ID)
        if torch.any(cpy_mask):
            cpy_mask = cpy_mask.unsqueeze(-1).expand_as(tgt)
            tgt[cpy_mask] = permuted_memory[cpy_mask]
            # "promote" copied tokens and reset others 
            # depth[cpy_mask] += 1
            # depth[~cpy_mask] = 0
        
        sub_mask = op_ids.eq(SUB_ID)
        if torch.any(sub_mask):
            old_embeds = self.embedding(permuted_input_ids[sub_mask]) * np.sqrt(self.d_model)
            new_embeds = self.embedding(tok_ids[sub_mask]) * np.sqrt(self.d_model)
            tgt[sub_mask] = permuted_memory[sub_mask] - old_embeds + new_embeds
        
        eos_mask = op_ids.eq(EOS_ID)
        if torch.any(eos_mask):
            tgt[eos_mask] = self.embedding.weight[self.eos_token_id] * np.sqrt(self.d_model)
        
        tgt = self.positional_embedding(tgt, d=1)
        # tgt = self.depth_embedding(tgt, d=1, pos=depth)
        
        return tgt
    
    def get_src(self, x):
        pad_mask = x.eq(self.pad_token_id)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x, pad_mask
   
    def get_probs(self, edit_logits, pad_mask):
        *_, idx_logits = edit_logits
        idx_logits[pad_mask.unsqueeze(1).expand_as(idx_logits)] = -1e9
        return tuple(map(lambda x: F.log_softmax(x, dim=-1), edit_logits))
  
    def forward(
        self, input_ids, edit_tgts,
        src=None, memory=None, cache=None,
    ):
        if self.training and memory is not None: raise Exception()
        if self.training and cache is not None: raise Exception()
        
        src_0, pad_mask = self.get_src(input_ids)
        src = src_0 if src is None else src
        
        memory = self.encoder(src, src_key_padding_mask=pad_mask) if memory is None else memory
        tgt = self.compute_tgt_static(input_ids, edit_tgts) if self.static_embeddings \
         else self.compute_tgt(input_ids, edit_tgts, memory)
      
        output, cache = self.decoder(
            tgt, memory,
            cache=cache,
            memory_key_padding_mask=pad_mask,
        )
        
        op_logits = self.op_head(output)
        tok_logits = self.tok_head(output)
        idx_logits = self.idx_head(output)
        
        probs = self.get_probs((op_logits, tok_logits, idx_logits), pad_mask)
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
        for i in range(T-1):
            input_ids = traj_input_ids[:, i]
            edit_tgts = tuple(map(lambda x: x[:, i], traj_edit_tgts))
            edit_probs, src, *_ = self.forward(input_ids, edit_tgts, src)
          
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
    
class Transformer(nn.Module):
    
    def __init__(
        self,
        d_model=512, nhead=8, max_len=10,
        encoder_layers=6, decoder_layers=6,
        dropout=0.1, dim_feedforward=2048,
        vocab_size=VOCAB_SIZE, # NOTE -- dangerous
        device='cpu', **_
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.device = device
        
        self.pad_token_id = PAD_TOKEN_ID # TODO -- shouldn't have these hardcoded
      
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.tok_head = nn.Linear(self.d_model, self.vocab_size)
        self.tok_head.weight = self.embedding.weight # weight tying
        
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len)
     
        # despite naming, the encoder layers are used for the decoder-only baseline
        # this is my fault for using the pytorch transformer in the first place
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
       
        # standard layers so that we can efficiently evaluate perplexity without extra engineering
        # afterwards we can load these weights into the Causal* variants for efficient inference
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        
    def get_src(self, x):
        pad_mask = x.eq(self.pad_token_id)
        x = self.embedding(x) * np.sqrt(self.d_model)
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
