import os
import json
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torch.nn import Transformer as T
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer

from const import VOCAB_SIZE, PAD_TOKEN_ID, INS_ID, CPY_ID, SUB_ID, EOS_ID
from utils import get_name, replace, log1mexp, check_nan
from embed import SinusoidalEmbedding
from data import SequenceDataset, StratifiedInfiniteSampler, TrajectoryDataset, collate_unsupervised
from transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

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
        pad_token_id=PAD_TOKEN_ID,
        tie_weights=False,
        name='test'
    ):
        super().__init__()

        self.d_model = d_model
        self.N = N
        self.vocab_size = vocab_size
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
        
        self.save_hyperparameters()
        
    def _train_loader(self, train_path, tokenizer, batch_size):
        dataset = SequenceDataset.from_trajectories(path=train_path, max_len=self.N, tokenizer=tokenizer, denoising=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=StratifiedInfiniteSampler(dataset, batch_size))
        return loader
        
    def _eval_loader(self, eval_path, tokenizer, batch_size):
        dataset = TrajectoryDataset.from_disk(eval_path, max_len=self.N, tokenizer=tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=StratifiedInfiniteSampler(dataset, batch_size), collate_fn=collate_unsupervised)
        return loader
        
    def _embed(self, ids):
        pad_mask = ids.eq(self.pad_token_id)
        x = self.embedding(ids) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x, pad_mask
    
    def _compute_p_ins(self, attn_weights, mem, input_tgt, output_tgt):
        # attn_weights: (B, N_out, N_in)
        # mem: (B, N_in, D)
        # input_tgt: (B, N_out, D)
        # output_tgt: (B, N_out, D)
        # p_ins = σ(W[c;i;o] + B)
        c = torch.bmm(attn_weights, mem)
        return F.logsigmoid(self.ins_fc(torch.cat([c, input_tgt, output_tgt], dim=-1)))
    
    def _aggregate_dist(self, weights, ids, eps=1e-7):
        # ids: (B, N_ins)
        # weights: (B, N_out, N_ins)
        ids = F.one_hot(replace(ids, self.pad_token_id, 0).long(), num_classes=self.vocab_size).float()
        return torch.log(torch.clamp(torch.bmm(weights, ids), eps, 1-eps))

    def forward(self, input_ids, output_ids):
        src, src_pad_mask = self._embed(input_ids)
        tgt, tgt_pad_mask = self._embed(output_ids)
        check_nan(src, 'src')
        check_nan(tgt, 'tgt')
        
        mem = self.encoder(src, src_key_padding_mask=src_pad_mask)
        check_nan(mem, 'mem')
        
        causal_mask = T.generate_square_subsequent_mask(output_ids.shape[1], dtype=torch.bool, device=tgt.device)
        h, (*_, attn_weights) = self.decoder(tgt, mem, memory_key_padding_mask=src_pad_mask, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_pad_mask)
        check_nan(h, 'h')
        check_nan(attn_weights, 'attn_weights')
       
        ins_logits = F.log_softmax(self.tok_head(h), dim=-1)
        cpy_logits = self._aggregate_dist(attn_weights, input_ids)
        check_nan(ins_logits, 'ins_logits')
        check_nan(cpy_logits, 'cpy_logits')
        
        p_ins = self._compute_p_ins(attn_weights, mem, tgt, h)
        check_nan(p_ins, 'p_ins')
        ins_dist = p_ins + ins_logits
        cpy_dist = log1mexp(p_ins) + cpy_logits
        
        return torch.logsumexp(torch.stack([ins_dist, cpy_dist], dim=-1), dim=-1)
    
    def on_train_epoch_start(self):
        self.train_loss = 0
        self.train_toks = 0

    def on_train_epoch_end(self):
        loss = self.train_loss / self.train_toks if self.train_toks > 0 else 0
        self.log('train/epoch_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/epoch_ppl', torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def _nll_loss(self, logits, output_ids):
        logits = logits[:, :-1].reshape(-1, self.vocab_size)
        output_ids = output_ids[:, 1:].reshape(-1)
        loss = F.nll_loss(logits, output_ids, ignore_index=self.pad_token_id, reduction='sum')
        toks = torch.sum(output_ids != self.pad_token_id)
        return loss, toks

    def training_step(self, batch, _):
        input_ids, output_ids = batch
        logits = self.forward(input_ids, output_ids)
        check_nan(logits, 'logits')
        loss, toks = self._nll_loss(logits, output_ids)
        check_nan(loss, 'loss')
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

    def validation_step(self, batch, batch_idx):
        traj_input_ids, traj_likelihood, *_ = batch
        
        traj_toks = []
        for i in range(traj_input_ids.shape[1]-1):
            input_ids = traj_input_ids[:, i]
            output_ids = traj_input_ids[:, i+1]
            logits = self.forward(input_ids, output_ids)
            loss, toks = self._nll_loss(logits, output_ids)
            self.eval_loss += loss
            self.eval_elbo_loss -= loss
            traj_toks.append(toks)
        
        self.eval_toks += sum(traj_toks)
        self.eval_elbo_toks += traj_toks[-1]
        self.eval_elbo_loss -= torch.sum(traj_likelihood)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=3e-4)
        
        def nan_grad_hook(mod, grad_in, _):
            if any(torch.isnan(gi).any() for gi in grad_in if gi is not None):
                print(f'nan grad in {mod.__class__.__name__}')
            
        for name, mod in self.named_modules():
            mod.register_backward_hook(nan_grad_hook)
        
        return {'optimizer': optim}
    
class PointerGeneratorEvolver(pl.LightningModule):
    
    def _ins_loss(self, p_ins):
        pass
    
    def forward(self, traj_input_ids):
        pass
    
    def _apply_edits(self, op_tgts, idx_tgts, mem):
        # op_tgts: (B, N)
        # mem: (B, N, D)
        
        mem = self.positional_embedding(mem, d=-1)
        
        B, N = op_tgts.shape
        permuted_mem = mem[torch.arange(B).unsqueeze(1), ]
        
        # return (B, N, D)
        
        return None
    
    def training_step(self, batch, _):
        traj_input_ids, _, (op_tgts, _, idx_tgts), _ = batch
        
        # NOTE -- quirk because we're reusing labelhttps://www.clsp.jhu.edu/workshops/s
        op_tgts = torch.argmax(op_tgts, dim=-1)
        op_tgts = replace(op_tgts, SUB_ID, INS_ID)
        op_tgts = replace(op_tgts, EOS_ID, CPY_ID)
        idx_tgts = torch.argmax(idx_tgts, dim=-1)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()
    
def main():
    args = parse_args()
    with open(args.config, 'r') as f: config = json.load(f)
    name = get_name(args.config)
   
    logger = None
    if not args.local:
        wandb.init(project='evolver', config=config)
        logger = WandbLogger(project='evolver', name=name, log_model='all')
       
    model = PointerGenerator(
        d_model=config['d_model'],
        dim_feedforward=config['dim_feedforward'],
        nhead=config['nhead'],
        dropout=config['dropout'],
        encoder_layers=config['encoder_layers'],
        N=config['N'],
        name=name
    )
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader = model._train_loader(config['train'], tokenizer, config['batch_size'])
    eval_loader = model._eval_loader(config['eval'], tokenizer, config['batch_size'])
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{REMOTE_PREFIX if args.device=="cuda" else "."}/checkpoints',
        filename=name+'-{step:08d}',
        save_top_k=3,
        monitor='eval/elbo',
        mode='max',
        every_n_train_steps=config['eval_at'],
        save_on_train_epoch_end=False
    )
    
    trainer = pl.Trainer(
        max_steps=config['train_steps'],
        limit_val_batches=config['eval_steps'],
        val_check_interval=config['eval_at'],
        check_val_every_n_epoch=None,
        accumulate_grad_batches=config['grad_accum_steps'],
        callbacks=[checkpoint_callback],
        accelerator=args.device,
        logger=logger,
    )
    
    trainer.fit(model, train_loader, eval_loader)
    
if __name__ == '__main__':
    main()
