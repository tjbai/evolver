import os
import json
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from const import VOCAB_SIZE
from evo import SinusoidalEmbedding
from transformer import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer
)

REMOTE_PREFIX = os.environ.get('REMOTE_PREFIX', '/scratch4/jeisner1')

class PointerGenerator(pl.LightningModule):
    def __init__(
        self,
        d_model=512,
        dim_feedforward=2048,
        nhead=8,
        dropout=0.1,
        layer_norm_eps=1e-5,
        encoder_layers=6,
        decoder_layers=6,
        N=512,
        vocab_size=VOCAB_SIZE,
        pad_token_id=-1,
        tie_weights=False,
        name='test'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.N = N
        self.name = name
        self.pad_token_id = pad_token_id
        
        self.codec_params = {
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps
        }
        
        self.encoder = TransformerEncoder(TransformerEncoderLayer(**self.codec_params), num_layers=encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(**self.codec_params), num_layers=decoder_layers)
        self.ins_fc = nn.Linear(3*d_model, 1)
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model, N)
        
        self.tok_head = nn.Linear(d_model, vocab_size)
        if tie_weights: self.tok_head.weight = self.embedding.weight
        
        self.causal_mask = T.generate_square_subsequent_mask(N, dtype=torch.bool)
        
        self.save_hyperparameters()
        
    def _embed(self, ids):
        pad_mask = ids.eq(self.pad_token_id)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x, pad_mask
    
    def _compute_p_ins(self, attn_weights, mem, input_tgt, output_tgt):
        # attn_weights: (B, N_out, N_in)
        # mem: (B, N_in, D)
        # input_tgt: (B, N_out, D)
        # output_tgt: (B, N_out, D)
       
        # from Bafna et al. (2024), p_ins = σ(W[c;i;o] + B)
        c = torch.bmm(attn_weights, mem)
        p = F.sigmoid(self.ins_fc(torch.cat([c, input_tgt, output_tgt], dim=-1))) 
        return torch.clamp(p, 1e-9, 1-1e-9)

    def forward(self, input_ids, output_ids):
        src, src_pad_mask = self._embed(input_ids)
        tgt, tgt_pad_mask = self._embed(output_ids)
        
        mem = self.encoder(src, src_key_padding_mask=src_pad_mask)
        h, (*_, attn_weights) = self.decoder(tgt, mem, memory_key_padding_mask=src_pad_mask, tgt_mask=self.causal_mask, tgt_key_padding_mask=tgt_pad_mask)
       
        ins_logits = F.log_softmax(self.tok_head(h), dim=-1)
        cpy_logits = torch.log(attn_weights)
        
        p_ins = self._compute_p_ins()
        ins_dist = torch.log(p_ins) + ins_logits
        cpy_dist = torch.log(1 - p_ins) + cpy_logits
        return torch.logsumexp(torch.stack([ins_dist, cpy_dist], dim=-1), dim=-1)
    
    def on_train_epoch_start(self):
        self.train_loss = 0
        self.train_toks = 0

    def on_train_epoch_end(self):
        loss = self.train_loss / self.train_toks if self.train_num_tokens > 0 else 0
        self.log('train/epoch_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/epoch_ppl', torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def _nll_loss(self, logits, output_ids):
        logits = logits[:, :-1].view(-1, self.vocab_size)
        output_ids = output_ids[:, 1:].view(-1)
        loss = F.nll_loss(logits, output_ids, ignore_index=self.pad_token_id, reduction='sum')
        toks = torch.sum(output_ids != self.pad_token_id)
        return loss, toks

    def training_step(self, batch, _):
        input_ids, output_ids = batch
        logits = self.forward(input_ids, output_ids)
        loss, toks = self._nll_loss(self, logits, output_ids)
        self.train_loss += loss
        self.train_toks += toks
        self.log('train/loss', loss / toks, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train/ppl', torch.exp(loss / toks), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss / toks
    
    def on_validation_epoch_start(self):
        self.eval_loss = 0
        self.eval_toks = 0
        self.eval_elbo_loss = 0
        self.eval_elbo_toks = 0
        
    def on_validation_epoch_end(self):
        loss = self.eval_loss / self.eval_toks if self.eval_toks > 0 else 0
        elbo = self.eval_elbo_loss / self.eval_elbo_toks if self.eval_elbo_toks > 0 else 0
        self.log('eval/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('eval/ppl', torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('eval/elbo', elbo, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, _):
        # traj_input_ids: (B, T, N)
        # traj_likelihhood: (B,)
        traj_input_ids, traj_likelihood = batch
        
        for i in range(T-1):
            input_ids = traj_input_ids[:, i]
            output_ids = traj_input_ids[:, i+1]
            logits = self.forward(input_ids, output_ids)
            loss, toks = self._nll_loss(logits, output_ids)
            
            assert loss >= 0
            self.eval_loss += loss
            self.eval_toks += toks
            self.eval_elbo_loss -= loss
            self.eval_elbo_toks += toks if i == T-2 else 0
            
        self.eval_elbo_loss -= torch.sum(traj_likelihood)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=3e-4)
        return {'optimizer': optim}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--local')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
def main():
    args = parse_args()
    with open(args.config, 'r') as f: config = json.load(f)
   
    logger = None 
    if not args.local:
        wandb.init(project='evolver', config=config)
        logger = WandbLogger(project='evolver')
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{REMOTE_PREFIX if args.device=="cuda" else "."}/checkpoints'
        filename=f'pgn-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
        
    
if __name__ == '__main__':
    main()
