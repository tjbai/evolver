import logging
from time import time
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T

import numpy as np
from transformers import BertModel

from constants import (
    VOCAB_SIZE,
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
    INS_ID, CPY_ID, SUB_ID, EOS_ID, PAD_ID
)

logging.basicConfig()
logger = logging.getLogger('train')

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'{f.__name__}: {te-ts}') 
        return result
    return wrap

def section(s):
    print(f'\n### {s}:')

def xent(logprobs, tgts, ignore=-1):
    loss = torch.sum(logprobs * tgts, dim=-1)
    keep_mask = torch.argmax(tgts, dim=-1) != ignore
    loss = loss * keep_mask
    tot = torch.sum(loss)
    n = max(torch.sum(keep_mask), 1)
    return -tot, n

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model=512, dropout=0.1, max_len=10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) 
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10_000) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        pe[:, 0, 1::2] = torch.cos(pos * div)
        
        # batch first
        pe = pe.transpose(0, 1) 
        
        self.register_buffer('pe', pe)
    
    def forward(self, x, dir):
        x = x + dir * self.pe[:, :x.shape[-2], :]
        return self.dropout(x) if dir > 0 else x
    
# Borrowed from: https://github.com/alex-matton/causal-transformer-decoder and modified for our use
# Implements attention key-value cache and implicit causal target mask
    
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

# This part is adapted from the official Pytorch implementation

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
    
### Core model

class Evolver(nn.Module):
    
    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return model
    
    def __init__(
        self,
        d_model=512, nhead=12, max_len=10,
        encoder_layers=6, decoder_layers=6,
        use_bert_embeddings=True,
        device='cpu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.max_len = max_len
        self.device = device
      
        if self.use_bert_embeddings: 
            bert = BertModel.from_pretrained('bert-base-multilingual-cased')
            self.embedding = bert.embeddings.word_embeddings
            _, self.d_embed = self.embedding.weight.shape
            self.ff_embedding = nn.Linear(self.d_embed, self.d_model)
        else:
            self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        self.pad_token_id = PAD_TOKEN_ID
        self.bos_token_id = BOS_TOKEN_ID # use [CLS] as BOS
        self.eos_token_id = EOS_TOKEN_ID # use [SEP] as EOS
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        decoder_layer = CausalTransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = CausalTransformerDecoder(decoder_layer, num_layers=decoder_layers)
       
        self.op_head = nn.Linear(d_model, 5)
        self.tok_head = nn.Linear(d_model, self.vocab_size)
        self.idx_head = nn.Linear(d_model, self.max_len)
    
    def compute_tgt(self, input_ids, memory, edit_tgts):
        op_ids, tok_ids, idx_ids = tuple(map(lambda x: torch.argmax(x, dim=-1), edit_tgts))
        B, cur_len = op_ids.shape
        
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0).repeat(B, 1)
        
        tgt = torch.zeros(B, cur_len, self.d_model).to(self.device)
       
        # subtract old positional encodings 
        memory = self.positional_encoding(memory, dir=-1)
        
        # permuted[i, j, :] = prev[i, idx_ids[i, j], :]
        permuted_memory = memory[torch.arange(B).unsqueeze(1).to(self.device), idx_ids]
        permuted_input_ids = input_ids[torch.arange(B).unsqueeze(1).to(self.device), idx_ids]
        
        # INS: add new embeddings
        ins_mask = op_ids.eq(INS_ID)
        if torch.any(ins_mask):
            ins_embeds = self.embedding(tok_ids[ins_mask])
            ins_embeds = self.ff_embedding(ins_embeds)
            tgt[ins_mask] = ins_embeds
            
        # CPY: copy over old embeddings
        cpy_mask = op_ids.eq(CPY_ID)
        if torch.any(cpy_mask):
            _cpy_mask = cpy_mask.unsqueeze(-1).expand_as(tgt)
            tgt[_cpy_mask] = permuted_memory[_cpy_mask]
        
        # SUB: subtract old embeddings and add new embeddings
        sub_mask = op_ids.eq(SUB_ID)
        if torch.any(sub_mask):
            old_embeds = self.embedding(permuted_input_ids[sub_mask])
            old_embeds = self.ff_embedding(old_embeds) 
            new_embeds = self.embedding(tok_ids[sub_mask])
            new_embeds = self.ff_embedding(new_embeds)
            tgt[sub_mask] = permuted_memory[sub_mask] - old_embeds + new_embeds
        
        # EOS: broadcast EOS embedding
        eos_mask = op_ids.eq(EOS_ID)
        tgt[eos_mask] = self.ff_embedding(self.embedding.weight[self.eos_token_id])
        
        # add new positional encodings 
        tgt = self.positional_encoding(tgt, dir=1)
        
        return tgt
    
    def get_src(self, x):
        x = self.embedding(x) * np.sqrt(self.d_model)
        if self.d_embed != self.d_model:
            x = self.ff_embedding(x)
            x = F.relu(x)
        x = self.positional_encoding(x, dir=1)
        return x, x.eq(self.pad_token_id) 
  
    def forward(
        self,
        input_ids,          # source-side token ids, used to compute target-side embeddings
        src,                # source-side embeddings, static if first in trajectory but dynamic otherwise
        edit_tgts,          # target edits, either teacher-forced or PF-sampled
        src_pad_mask,       # source-side pad token mask
        tgt_pad_mask,       # target-side pad token mask (shouldn't matter because of causal decoder)
        memory=None,
        cache=None
    ): 
        if self.training and memory is not None:
            raise Exception('encoder memory found during training') 
        
        if self.training and cache is not None:
            raise Exception('kv cache found during training')
        
        if memory is None:
            memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
            
        tgt = self.compute_tgt(input_ids, memory, edit_tgts)
        cur_len = tgt.shape[1]
      
        output, new_cache = self.decoder(
            tgt, memory,
            cache=cache,
            tgt_key_padding_mask=tgt_pad_mask[:, :cur_len],
            memory_key_padding_mask=src_pad_mask,
        )
        
        op_logits = self.op_head(output)
        tok_logits = self.tok_head(output)
        idx_logits = self.idx_head(output)
        
        return (
            (op_logits, tok_logits, idx_logits),
            tgt, memory, new_cache
        )
   
    @staticmethod
    def get_probs(edit_logits, pad_mask):
        *_, idx_logits = edit_logits
        idx_logits[pad_mask.unsqueeze(1).expand_as(idx_logits)] = -1e9
        return tuple(map(lambda x: F.log_softmax(x, dim=-1), edit_logits))
    
    def loss(self, edit_logits, edit_tgts, pad_mask):
        op_tgts, tok_tgts, idx_tgts = tuple(map(lambda x: x[:, 1:, :], edit_tgts))
        assert op_tgts.shape[1] == self.max_len - 1
        
        edit_probs = self.get_probs(edit_logits, pad_mask)
        op_probs, tok_probs, idx_probs = tuple(map(lambda x: x[:, :-1, :], edit_probs))
        assert op_tgts.shape[1] == self.max_len - 1
     
        logger.info(f'MASK: {pad_mask}')
        logger.info(f'PROB: {op_probs}')
        
        op_tot, op_n = xent(op_probs, op_tgts, ignore=PAD_ID)
        tok_tot, tok_n = xent(tok_probs, tok_tgts, ignore=PAD_TOKEN_ID)
        idx_tot, idx_n = xent(idx_probs, idx_tgts, ignore=0)
        
        return (op_tot, op_n, tok_tot, tok_n, idx_tot, idx_n)

    def traj_loss(self, traj_input_ids, traj_edit_tgts):
        traj_op_tot = traj_tok_tot = traj_idx_tot = 0 
        
        traj_src, traj_pad_mask = self.get_src(traj_input_ids)
        src = traj_src[:, 0, :]
        
        traj_loss = 0
        T = traj_input_ids.shape[1]
        
        for i in range(T-1):
            input_ids = traj_input_ids[:, i, :]
            src_pad_mask = traj_pad_mask[:, i, :]
            tgt_pad_mask = traj_pad_mask[:, i+1, :]
            edit_tgts = tuple(map(lambda x: x[:, i, :], traj_edit_tgts))
            
            edit_logits, src, *_ = self.forward(input_ids, src, edit_tgts, src_pad_mask, tgt_pad_mask)
            op_tot, op_n, tok_tot, tok_n, idx_tot, idx_n = self.loss(edit_logits, edit_tgts, src_pad_mask)
            
            traj_loss += (op_tot / op_n) + (tok_tot / tok_n) + (idx_tot / idx_n)
            traj_op_tot += op_tot
            traj_tok_tot += tok_tot
            traj_idx_tot += idx_tot
           
        # optim.zero_grad()
        # (-traj_loss).backward()
        # optim.step()
      
        # we backpropagate over the per-occurrence loss for each of op, tok, and idx
        # but, we want to report the total operation loss averaged across all indices
        N = torch.sum(~traj_pad_mask[:, 1:, :])
        return traj_loss, traj_op_tot / N, traj_tok_tot / N, traj_idx_tot / N
    
### Baseline

class Transformer(nn.Module):
    
    def __init__(
        self,
        d_model=512, nhead=8, max_len=10, vocab_size=10,
        encoder_layers=6, decoder_layers=6
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        self.pad_token_id = -1 # TODO -- shouldn't have these hardcoded
      
        # bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        # self.embedding = bert.embeddings.word_embeddings
        # self.ff_embedding = nn.Linear(768, self.d_model)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.tok_head = nn.Linear(self.d_model, self.vocab_size)
        self.tok_head.weight = self.embedding.weight # weight tying
        
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.encoder_layers = encoder_layers
        
        # decoder_layer = CausalTransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        # self.decoder = CausalTransformerDecoder(decoder_layer, num_layers=decoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_layers = decoder_layers
        
    def get_src(self, x):
        pad_mask = x.eq(self.pad_token_id)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.positional_encoding(x, dir=1)
        return x, pad_mask
    
    def forward(
        self,
        src, tgt,
        src_pad_mask, tgt_pad_mask,
        memory=None, cache=None
    ):
        # if self.training and memory is not None:
        #     raise Exception('encoder memory found during training') 
        
        # if self.training and cache is not None:
        #     raise Exception('kv cache found during training')
        
        # if memory is None and self.encoder_layers > 0:
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
            
        output = self.decoder(
            tgt, memory,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        
        tok_logits = self.tok_head(output)
        
        return tok_logits, memory
    
    def step(self, input_ids, output_ids):
        if input_ids is not None: src, src_pad_mask = self.get_src(input_ids)
        else: src, src_pad_mask = None, None
        tgt, tgt_pad_mask = self.get_src(output_ids)
        tok_logits, *_ = self.forward(src, tgt, src_pad_mask, tgt_pad_mask)
        tok_probs = F.log_softmax(tok_logits, dim=-1)
        return xent(
            tok_probs[:, :-1],
            F.one_hot(output_ids[:, 1:], num_classes=self.vocab_size),
            ignore=PAD_TOKEN_ID
        )