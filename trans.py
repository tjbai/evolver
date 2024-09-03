import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T

class TransformerEncoderLayer(nn.Module):
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        **_
    ):
        super().__init__() 
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, bias=True)
        
        self.fc_1 = nn.Linear(d_model, dim_feedforward, bias=True)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(dim_feedforward, d_model, bias=True)
       
        self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=True)
        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        self.activation = F.relu
       
    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
        x = self.ln_2(x + self._ff_block(x))
        return x
        
    def _sa_block(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False):
        return self.dropout_1(self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal)[0])
    
    def _ff_block(self, x):
        return self.dropout_2(self.fc_2(self.dropout_fc(self.activation(self.fc_1(x)))))
    
class AdaptiveTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ln_1 = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps, elementwise_affine=False)
        self.ln_2 = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps, elementwise_affine=False)
        
        self.adaln = nn.Sequential(nn.ReLU(), nn.Linear(self.d_model, 6 * self.d_model, bias=True))
        nn.init.constant_(self.adaln[-1].weight, 0)
        nn.init.constant_(self.adaln[-1].bias, 0)
       
    @staticmethod 
    def _mod(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
    def forward(self, x, depth_embed, *args, **kwargs):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln(depth_embed).chunk(6, dim=1)
        x = x + gate_attn.unsqueeze(1) * self._sa_block(self._mod(self.ln_1(x), shift_attn, scale_attn), *args, **kwargs)
        x = x + gate_mlp.unsqueeze(1) * self._ff_block(self._mod(self.ln_1(x), shift_mlp, scale_mlp))
        return x
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src, depth_embed=None, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src 
        for mod in self.layers:
            if depth_embed is not None: x = mod(x, depth_embed, src_mask, src_key_padding_mask, is_causal=is_causal)
            else: x = mod(x, src_mask, src_key_padding_mask, is_causal=is_causal)
        return x
    
class TransformerDecoderLayer(nn.Module):
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        **_
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, bias=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, bias=True)
        
        self.fc_1 = nn.Linear(d_model, dim_feedforward, bias=True)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(dim_feedforward, d_model, bias=True)
        
        self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=True)
        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=True)
        self.ln_3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.activation = F.relu

    def forward(
        self,
        tgt, mem,
        tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        tgt_is_causal=False, memory_is_causal=False
    ):
        x = tgt
       
        res = self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        x = self.ln_1(x + res)
        
        if mem is not None: 
            res, attn_weights = self._xa_block(x, mem, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = self.ln_2(x + res)
        else:
            attn_weights = None
            x = self.ln_2(x)
        
        res = self._ff_block(x)
        x = self.ln_3(x + res)
        
        return x, attn_weights

    def _sa_block(self, x, tgt_mask, tgt_key_padding_mask, is_causal):
        return self.dropout_1(self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False, is_causal=is_causal)[0])
    
    def _xa_block(self, x, mem, src_mask, src_key_padding_mask, is_causal=False):
        x, attn_weights = self.cross_attn(x, mem, mem, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, is_causal=is_causal, need_weights=True)
        return self.dropout_2(x), attn_weights
    
    def _ff_block(self, x):
        return self.dropout_3(self.fc_2(self.dropout_fc(self.activation(self.fc_1(x)))))
    
class TransformerDecoder(nn.Module):
   
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, tgt, memory, **kwargs):
        x = tgt
        attn_weights = []
        for mod in self.layers:
            x, layer_attn_weights = mod(x, memory, **kwargs) 
            attn_weights.append(layer_attn_weights)
        return x, attn_weights
    
class CausalTransformerDecoderLayer(TransformerDecoderLayer):
    
    def forward(self, tgt, mem, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        causal_mask = T.generate_square_subsequent_mask(tgt.size(1), tgt.device, dtype=torch.bool)
        
        if self.training:
            return super().forward(
                tgt, mem,
                tgt_mask=causal_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
        x = tgt[:, -1:, :]
      
        res = self._causal_sa_block(x, tgt, None, tgt_key_padding_mask)
        x = self.ln_1(x + res)
        
        res, attn_weights = self._xa_block(x, mem, memory_mask, memory_key_padding_mask)
        x = self.ln_2(x + res)
       
        res = self._ff_block(x)
        x = self.ln_3(x + res)
        
        return x, attn_weights
   
    # same as before, but we only compute keys for the last token! 
    def _causal_sa_block(self, x, tgt, tgt_mask, tgt_key_padding_mask):
        return self.dropout_1(self.self_attn(x, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0])
    
class CausalTransformerDecoder(TransformerDecoder):
    
    def forward(self, tgt, mem, cache=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        masks = {
            'memory_mask': memory_mask,
            'tgt_key_padding_mask': tgt_key_padding_mask,
            'memory_key_padding_mask': memory_key_padding_mask
        }
        
        x = tgt
        
        if self.training:
            attn_weights = []
            for mod in self.layers:
                x, layer_attn_weights = mod(x, mem, **masks)
                attn_weights.append(layer_attn_weights)
            return x, attn_weights, None
        
        x = tgt
        attn_weights = []
        new_cache = []
        for i, mod in enumerate(self.layers):
            x, layer_attn_weights = mod(x, mem, **masks)
            attn_weights.append(layer_attn_weights)
            new_cache.append(x)
           
            # cache[i]: (B, N, D)
            if cache is not None:
                x = torch.cat([cache[i], x], dim=1)
                
        new_cache = torch.stack(new_cache, dim=0)
        if cache is not None:
            new_cache = torch.cat([cache, new_cache], dim=2)
            
        return x, attn_weights, new_cache

class MultiheadPointer(nn.MultiheadAttention):
    '''
    like multihead attention but only returns attn_weights
    this creates useless value parameters but is quick and dirty
    ''' 
    
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def forward(self, query, key, attn_mask=None, key_padding_mask=None):
        if query.dim() == 3:
            query = query.transpose(0, 1)
        if key.dim() == 3:
            key = key.transpose(0, 1) 
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.transpose(0, 1)
        
        return F.multi_head_attention_forward(
            query, key, key,
            self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            need_weights=True,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[1]
