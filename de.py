import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T

from constants import *
from model import SinusoidalEmbedding

INS_ID = 0
CPY_ID = 1

class DependencyEvolver(nn.Module):
    
    def __init__(
        self,
        d_model=512, dim_feedforward=2048, nhead=8, dropout=0.1, N=64,
        encoder_layers=6, decoder_layers=6,
        tok_v=VOCAB_SIZE, rel_v=0, pos_v=0,
        pad_token_id=PAD_TOKEN_ID
    ):
        super().__init__()
        
        self.N = N
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.causal_mask = T.generate_square_subsequent_mask(N)
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
        
        self.op_head = nn.Linear(d_model, 2)
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
        
    def root(self):
        pass
    
    def _replace(self, t, a, b):
        return torch.where(t == a, b, t)
    
    def tgt_op(self, mem, tgt_op, tgt_cpy):
        B, N, _ = mem.shape
        
        permuted_mem = self.positional_embedding(mem, d=-1)[
            torch.arange(B, device=mem.device).unsqueeze(1),
            torch.where(tgt_cpy.eq(-1), torch.arange(N).expand(B, -1), tgt_cpy)
        ]
       
        tgt = torch.where(~tgt_op.eq(0).unsqueeze(-1).expand_as(mem), permuted_mem, 0) \
            + torch.where(tgt_op.eq(0).unsqueeze(-1).expand_as(mem), self.plh, 0)

        return self.positional_embedding(tgt, d=1) 
    
    def forward_op(self, src, tgt_op, tgt_cpy, src_pad_mask):
        encoder_masks = {'src_key_padding_mask': src_pad_mask}
        decoder_masks = {'tgt_mask': self.causal_mask, 'tgt_key_padding_mask': tgt_op.eq(-1), 'memory_key_padding_mask': src_pad_mask}
        
        mem = self.encoder(src, **encoder_masks)
        tgt = self.tgt_op(mem, tgt_op, tgt_cpy)
        h = self.decoder(tgt, mem, **decoder_masks)
        l = self.head_op(h)
        
        return l, tgt, decoder_masks['tgt_key_padding_mask']
    
    def tgt_par(self, mem, tgt_par, is_leaf):
        B, _, _ = mem.shape 
        
        permuted_mem = mem[
            torch.arange(B, device=mem.device).unsqueeze(1),
            torch.where(tgt_par == -1, 0, tgt_par)
        ]
        
        tgt = mem \
            + torch.where((tgt_par > 0).unsqueeze(-1).expand_as(mem), permuted_mem, 0) \
            + torch.where(is_leaf.unsqueeze(-1).expand_as(mem), self.done, 0)
        
        return tgt
    
    def forward_par(self, src, tgt_par, src_pad_mask, is_leaf):
        encoder_masks = {'src_key_padding_mask': src_pad_mask}
        decoder_masks = {'tgt_mask': self.causal_mask, 'tgt_key_padding_mask': tgt_par.eq(-1), 'memory_key_padding_mask': src_pad_mask}
        
        mem = self.encoder(src, **encoder_masks)
        tgt = self.tgt_par(mem, tgt_par, is_leaf)
        h = self.decoder(tgt, mem, **decoder_masks)
        l = self.par_head(h)
        
        return l, tgt
    
    def tgt_gen(self, mem, tgt_gen):
        embeds = self.embedding(self._replace(tgt_gen, -1, 0))
        return mem + torch.where(~tgt_gen.eq(-1).unsqueeze(-1).expand_as(mem), embeds, 0)
    
    def forward_gen(self, src, tgt_gen, src_pad_mask):
        encoder_masks = {'src_key_padding_mask': src_pad_mask}
        decoder_masks = {'tgt_mask': self.causal_mask, 'tgt_key_padding_mask': tgt_gen.eq(-1), 'memory_key_padding_mask': src_pad_mask}
        
        mem = self.encoder(src, **encoder_masks)
        tgt = self.tgt_rel(mem, tgt_gen)
        h = self.decoder(tgt, mem, **decoder_masks)
        l = self.v_head(h)
        
        return l, tgt
    
    def forward(
        self, input_ids, src,
        tgt_op, tgt_cpy, is_leaf, tgt_par,
        tgt_rel, tgt_pos, tgt_tok
    ):
        # procreate
        l_op, src, src_pad_mask = self.forward_op(input_ids, src, tgt_op, tgt_cpy, is_leaf)
        
        # who are my parents?
        l_par, src = self.forward_par(src, tgt_par, src_pad_mask)
       
        # character development
        l_rel, src = self.forward_gen(src, tgt_rel, src_pad_mask)
        l_pos, src = self.forward_gen(src, tgt_pos, src_pad_mask)
        l_tok, src = self.forward_gen(src, tgt_tok, src_pad_mask)
        
    def loss(self, ls, tgts):
        return sum(F.cross_entropy(l[:, :-1], t[:, 1:], ignore_index=-1) for l, t in zip(ls, tgts))
        