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
        # TODO -- flash attention?
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
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        layer_norm_eps
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
        x = self.ln_1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
        x = self.ln_2(x + self._xa_block(x, mem, memory_mask, memory_key_padding_mask, memory_is_causal))
        x = self.ln_3(x + self._ff_block(x))
        return x

    def _sa_block(self, x, src_mask, src_key_padding_mask, is_causal):
        return self.dropout_1(self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal)[0])
    
    def _xa_block(self, x, mem, src_mask, src_key_padding_mask, is_causal=False):
        return self.dropout_2(self.cross_attn(x, mem, mem, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, is_causal=is_causal, need_weights=False)[0]) 
    
    def _ff_block(self, x):
        return self.dropout_3(self.fc_2(self.dropout_fc(self.activation(self.fc_1(x)))))

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
