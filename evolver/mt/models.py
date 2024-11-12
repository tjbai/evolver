import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from ..embed import SinusoidalEmbedding
from ..trans import TransformerEncoderLayer, TransformerEncoder, CausalTransformerDecoderLayer, CausalTransformerDecoder, MultiheadPointer

class Evolver(nn.Module):

    def __init__(
        self,
        d_model, dim_feedforward, nhead, dropout, layer_norm_eps,
        encoder_layers, decoder_layers,
        vocab_size, max_len, bos_token_id, eos_token_id, pad_token_id, name
    ):
        super().__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = -1 # NOTE -- static
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.name = name
        
        t_params = {
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps}
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len+1)
       
        encoder_layer = TransformerEncoderLayer(**t_params)
        decoder_layer = CausalTransformerDecoderLayer(**t_params)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = CausalTransformerDecoder(decoder_layer, decoder_layers)
        
        self.op_head = nn.Linear(t_params['d_model'], 3)
        self.tok_head = nn.Linear(t_params['d_model'], vocab_size)
        
        self.pointer = MultiheadPointer(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False)
        
    def embed(self, x):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x
    
    def forward(self, batch, cache=None):
        src_ids = batch['src_ids']
        input_ids = batch['input_ids']
        tgt_ids = batch['tgt_ids']
        
        src_embed = self.embed(src_ids)
        input_embed = self.embed(input_ids)
        tgt = self.embed(tgt_ids)

        src = torch.cat([src_embed, input_embed], dim=1)
        pad_mask = torch.cat([src_ids, input_ids], dim=1).eq(self.pad_token_id)
        mem = self.encoder(src, src_key_padding_mask=pad_mask)
        h, (*_, idx_weights), cache = self.decoder(tgt, mem, memory_key_padding_mask=pad_mask, cache=cache)
        
        op_probs = F.log_softmax(self.op_head(h), dim=-1)
        tok_probs = F.log_softmax(self.tok_head(h), dim=-1)
        
        idx_weights = torch.log(torch.clamp(idx_weights, 1e-7, 1-1e-7))[:, :, src_ids.shape[1]:]
        idx_probs = F.log_softmax(idx_weights, dim=-1)
        
        return (op_probs, tok_probs, idx_probs), cache
    
    def compute_tgt(self, tgt_ids, edit_ids, mem):
        pass
    
    def step(self, batch):
        (op_probs, tok_probs, idx_probs), _ = self.forward(batch)
        return (
            F.nll_loss(op_probs[:, :-1].transpose(1, 2), batch['edit_ids'][0][:, 1:], ignore_index=-1, reduction='mean'),
            F.nll_loss(tok_probs[:, :-1].transpose(1, 2), batch['edit_ids'][1][:, 1:], ignore_index=-1, reduction='mean'),
            F.nll_loss(idx_probs[:, :-1].transpose(1, 2), batch['edit_ids'][2][:, 1:], ignore_index=-1, reduction='mean')
        )
    
    @classmethod 
    def apply_edits(self, input_ids, edit_ids):
        op_ids, tok_ids, idx_ids = edit_ids
        B = input_ids.shape[0]
        res = torch.zeros(B, op_ids.shape[1], dtype=torch.long, device=input_ids.device)

        ins_mask = op_ids.eq(0) | op_ids.eq(2)
        res[ins_mask] = tok_ids[ins_mask]
        
        cpy_mask = op_ids.eq(1)
        permuted_inputs = input_ids[torch.arange(B).view(-1, 1), idx_ids]
        
        res[cpy_mask] = permuted_inputs[cpy_mask]
        
        return res
    
    def _generate(self, batch, temp):
        src_ids = batch['src_ids'] 
        input_ids = batch['input_ids']
        
        B = src_ids.shape[0]
        device = src_ids.device
        
        tgt_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        op_ids = torch.full((B, 1), 1, dtype=torch.long, device=device)
        tok_ids = torch.full((B, 1), -1, dtype=torch.long, device=device)
        idx_ids = torch.full((B, 1), 0, dtype=torch.long, device=device)
        
        alive = torch.ones(B, dtype=torch.bool)
       
        cache = None
        for _ in range(self.max_len):
            if not alive.any(): break
            
            batch = {
                'src_ids': src_ids,
                'input_ids': input_ids,
                'tgt_ids': tgt_ids,
                'edit_ids': (op_ids, tok_ids, idx_ids)}

            probs, cache = self.forward(batch, cache=cache)
            next_op, next_tok, next_idx = tuple(map(lambda x: F.softmax(x[:, -1] / temp, dim=-1), probs))
            
            op_id = torch.multinomial(next_op, num_samples=1)
            tok_id = torch.multinomial(next_tok, num_samples=1)
            idx_id = torch.multinomial(next_idx, num_samples=1)
            
            idx_id[op_id.eq(0)] = -1
            tok_id[op_id.eq(1)] = -1

            op_ids = torch.cat([op_ids, op_id], dim=1)
            tok_ids = torch.cat([tok_ids, tok_id], dim=1)
            idx_ids = torch.cat([idx_ids, idx_id], dim=1)

            tgt_ids = self.apply_edits(input_ids, (op_ids, tok_ids, idx_ids))
            alive[tgt_ids[:, -1] == self.eos_token_id] = False
            
        return tgt_ids
    
    # TODO -- rollout should start from empty string
    def rollout(self, batch, T=10, temp=1, verbose=False):
        pass
        # self.decoder.set_causal()
        # traj = []
        # for _ in tqdm(range(T), desc='rolling out', disable=not verbose):
        #     batch = {'src_ids': batch['src_ids'], 'input_ids': traj[-1]}
        #     traj.append(self._generate(batch, temp=temp))
        # self.decoder.set_parallel()
        # return traj
    
class Transformer(nn.Module):
    
    def __init__(
        self,
        d_model, dim_feedforward, nhead, dropout, layer_norm_eps,
        encoder_layers, decoder_layers, vocab_size, max_len,
        pad_token_id, bos_token_id, eos_token_id, name
    ):
        super().__init__() 
        
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.name = name
        
        t_params = {
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps
        }
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len+1)
       
        encoder_layer = TransformerEncoderLayer(**t_params)
        decoder_layer = CausalTransformerDecoderLayer(**t_params)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = CausalTransformerDecoder(decoder_layer, decoder_layers)
        
        self.tok_head = nn.Linear(d_model, vocab_size)
        
    def embed(self, x):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x
    
    def forward(self, batch, cache=None):
        src_ids = batch['src_ids']
        tgt_ids = batch['tgt_ids']
        
        src = self.embed(src_ids)
        tgt = self.embed(tgt_ids)
        
        pad_mask = src_ids.eq(self.pad_token_id)
        mem = self.encoder(src, src_key_padding_mask=pad_mask)
        h, _, cache = self.decoder(tgt, mem, memory_key_padding_mask=pad_mask, cache=cache)
        
        return self.tok_head(h), cache
       
    def step(self, batch, reduce=True):
        h, _ = self.forward(batch)
        tok_probs = F.log_softmax(h, dim=-1)
        return F.nll_loss(tok_probs[:, :-1].transpose(1, 2), batch['tgt_ids'][:, 1:], ignore_index=self.pad_token_id, reduction='mean' if reduce else 'sum')
    
    @torch.no_grad()
    def generate(self, src_ids, beam_size=4, temperature=1.0, **_):
        B = src_ids.shape[0]
        device = src_ids.device
        self.decoder.set_causal()
        
        tgt_ids = torch.full((B * beam_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        src_ids = src_ids.repeat_interleave(beam_size, dim=0)
        
        beam_scores = torch.zeros((B, beam_size), device=device)
        beam_scores[:, 1:] = -1e9
        
        finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)
        
        cache = None
        for _ in range(self.max_len):
            logits, cache = self.forward({'src_ids': src_ids, 'tgt_ids': tgt_ids}, cache=cache)
            next_tok_logits = logits[:, -1, :] / temperature
            
            vocab_size = next_tok_logits.shape[-1]
            log_probs = F.log_softmax(next_tok_logits, dim=-1)
            
            log_probs = log_probs.view(B, beam_size, -1)
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            next_scores, next_tokens = next_scores.view(B, -1).topk(beam_size, dim=1)
            beam_scores = next_scores
            
            beam_indices = next_tokens // vocab_size
            token_indices = next_tokens % vocab_size
            
            tgt_ids = tgt_ids.view(B, beam_size, -1)
            tgt_ids = torch.cat([tgt_ids[torch.arange(B).unsqueeze(1), beam_indices], token_indices.unsqueeze(-1)], dim=-1)
            tgt_ids = tgt_ids.view(B * beam_size, -1)
            
            finished |= (token_indices == self.eos_token_id).view(-1)
            
            if finished.all(): break
        
        tgt_ids = tgt_ids.view(B, beam_size, -1)
        best_beam = beam_scores.argmax(dim=1)
        tgt_ids = tgt_ids[torch.arange(B), best_beam]
        
        self.decoder.set_parallel()
        return tgt_ids
        
class Teacher(nn.Module):
    
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        max_len,
        d_model=512,
        dim_feedforward=2048,
        nhead=8,
        dropout=0,
        layer_norm_eps=1e-5,
        encoder_layers=6,
        decoder_layers=6,
        name=None
    ):
        
        super().__init__()
        
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.name = name
        
        t_params = {
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps}
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len+1)
       
        encoder_layer = TransformerEncoderLayer(**t_params)
        decoder_layer = CausalTransformerDecoderLayer(**t_params)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = CausalTransformerDecoder(decoder_layer, decoder_layers)
        
        self.tok_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, batch, cache=None):
        src_ids = batch['src_ids']
        input_ids = batch['input_ids']
        tgt_ids = batch['tgt_ids']
        
        src = torch.cat([self.embed(src_ids), self.embed(input_ids)], dim=1)
        pad_mask = torch.cat([src_ids, input_ids], dim=1).eq(self.pad_token_id)
        tgt = self.embed(tgt_ids)
        
        mem = self.encoder(src, src_key_padding_mask=pad_mask)
        h, _, cache = self.decoder(tgt, mem, memory_key_padding_mask=pad_mask, cache=cache)
        
        return self.tok_head(h), cache
    
    def embed(self, x):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x
    
    def step(self, batch, reduce=True):
        h, _ = self.forward(batch)
        tok_probs = F.log_softmax(h, dim=-1)
        return F.nll_loss(tok_probs[:, :-1].transpose(1, 2), batch['tgt_ids'][:, 1:], ignore_index=self.pad_token_id, reduction='mean' if reduce else 'sum')
    
    @torch.no_grad()
    def _generate(self, batch, temp=0.7, **_):
        src_ids = batch['src_ids']
        input_ids = batch['input_ids']
        
        B = src_ids.shape[0]
        device = src_ids.device
        
        tgt_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        cache = None
        for _ in range(self.max_len):
            logits, cache = self.forward({
                'src_ids': src_ids.to(device),
                'input_ids': input_ids.to(device),
                'tgt_ids': tgt_ids.to(device)}, cache=cache)

            next_tok_logits = logits[:, -1, :] / temp
            next_tok = torch.multinomial(F.softmax(next_tok_logits, dim=-1), num_samples=1)
            tgt_ids = torch.cat([tgt_ids, next_tok], dim=-1)

            finished |= (next_tok.squeeze(-1) == self.eos_token_id)
            if finished.all(): break
        
        return tgt_ids
        
    def rollout(self, batch, T=10, verbose=False):
        self.decoder.set_causal()
        traj = [batch['root_ids']]
        for _ in tqdm(range(T), desc='rolling out', disable=not verbose):
            batch = {'src_ids': batch['src_ids'], 'input_ids': traj[-1]}
            traj.append(self._generate(batch))
        self.decoder.set_parallel()
        return traj
