import os
import json
import math
import random
import logging
import argparse
from datetime import datetime

import spacy
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sacrebleu import corpus_bleu
from tqdm import tqdm
from datasets import load_dataset

from embed import SinusoidalEmbedding
from trans import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, MultiheadPointer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_wandb(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f'step {step}: {data}')

class SpacyTokenizer:
    
    def __init__(self):
        self.de_nlp = spacy.load('de_core_news_sm')
        self.en_nlp = spacy.load('en_core_web_sm')
        
        self.vocab = {}
        with open('vocab/wmt14_de_en.vocab', 'r') as f:
            i = 0
            for t in enumerate(f, start=3):
                if t not in self.vocab:
                    self.vocab[t] = i
                    i += 1
                
        self.vocab['BOS'] = 0
        self.vocab['EOS'] = 1
        self.vocab['PAD'] = 2
        self.vocab['UNK'] = len(self.vocab)
        
        self.id_to_tok = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
    def encode(self, text, lang, skip_special_tokens=False):
        doc = {'de': self.de_nlp, 'en': self.en_nlp}[lang](text)
        tokens = [self.vocab.get(token.text, self.vocab['UNK']) for token in doc]
        if not skip_special_tokens: tokens = [self.vocab['BOS']] + tokens + [self.vocab['EOS']]
        return tokens
    
    def decode(self, tok_ids, skip_special_tokens=False):
        tokens = []
        for id in tok_ids:
            token = self.id_to_tok.get(id, 'UNK')
            if skip_special_tokens and token in {'PAD', 'BOS', 'EOS'}: continue
            tokens.append(token)
        return ' '.join(tokens)
    
class MTDataset(Dataset):

    def __init__(self, split='train', max_len=256, buffer_size=1000, tokenizer=SpacyTokenizer()):
        self.dataset = load_dataset('wmt14', 'de-en', split=split)
        self.max_len = max_len
        
        # NOTE -- not a great tokenizer, huge vocab
        self.tokenizer = tokenizer
        
        self.buffer = []
        self.buffer_index = 0
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, _):
        if self.buffer_index >= len(self.buffer): self.refill_buffer()
        item = self.buffer[self.buffer_index]
        self.buffer_index += 1
        
        src_ids = self.tokenizer.encode(item['de'], lang='de')[:self.max_len]
        tgt_ids = self.tokenizer.encode(item['en'], lang='en')[:self.max_len]
        
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids} 
    
    def refill_buffer(self):
        start_idx = random.randint(0, len(self.dataset) - self.buffer_size)
        self.buffer = self.dataset[start_idx:start_idx + self.buffer_size]['translation']
        random.shuffle(self.buffer)
        self.buffer_index = 0
        
    def collate_fn(self, batch):
        src_ids = [item['src_ids'] for item in batch]
        tgt_ids = [item['tgt_ids'] for item in batch]
        
        src_ids_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in src_ids],
            batch_first=True,
            padding_value=self.tokenizer.vocab['PAD']
        )
        
        tgt_ids_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in tgt_ids],
            batch_first=True,
            padding_value=self.tokenizer.vocab['PAD']
        )
        
        return {'src_ids': src_ids_padded, 'tgt_ids': tgt_ids_padded}

class MTEvolver(nn.Module):

    def __init__(
        self,
        d_model, dim_feedforward, nhead, dropout, layer_norm_eps,
        encoder_layers, decoder_layers,
        vocab_size, max_len, bos_token_id, eos_token_id, name
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
            'layer_norm_eps': layer_norm_eps
        }
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len+1)
       
        encoder_layer = TransformerEncoderLayer(**t_params)
        decoder_layer = TransformerDecoderLayer(**t_params)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, decoder_layers)
        
        self.op_head = nn.Linear(t_params['d_model'], 3)
        self.tok_head = nn.Linear(t_params['d_model'], vocab_size)
        
        self.pointer = MultiheadPointer(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False
        )
        
    def embed(self, x, embedding):
        x = embedding(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x
    
    def forward(self, batch, src=None):
        src_ids = batch['src_ids']
        input_ids = batch['input_ids']
        output_ids = batch['output_ids']
        
        B = input_ids.shape[0]
        N_pref = src_ids.shape[1]
        N_out = output_ids.shape[1]
        device = src_ids.device
        
        src_embed = self.embed(src_ids)
        input_embed = self.embed(input_ids)
        src = torch.cat([src_embed, input_embed if src is None else src], dim=1)
        
        pad_mask = torch.cat([src_ids, input_ids], dim=1).eq(self.pad_token_id)
        causal_mask = T.generate_square_subsequent_mask(N_out, dtype=torch.bool, device=device)
        
        mem = self.encoder(src, src_key_padding_mask=pad_mask) if mem is None else mem
        tgt = self.embed(output_ids)
        h, (*_, idx_weights) = self.decoder(tgt, mem, memory_key_padding_mask=pad_mask, tgt_mask=causal_mask)
        # tgt = self.embed(self.apply_edits(input_ids, edit_ids)) if self.static \
        #         else self.compute_tgt(input_ids, edit_ids, mem[:, src_ids.shape[1]:])
        
        op_probs = F.log_softmax(self.op_head(h), dim=-1)
        tok_probs = F.log_softmax(self.tok_head(h), dim=-1)
        
        idx_weights = torch.log(torch.clamp(idx_weights, 1e-7, 1-1e-7))[:, :, N_pref:]
        idx_probs = F.log_softmax(idx_weights, dim=-1)
        
        return op_probs, tok_probs, idx_probs
    
    def compute_tgt(self, tgt_ids, edit_ids, mem):
        pass
    
    def step(self, batch):
        pass
    
    def generate(self, src_ids, max_depth=10, max_steps=200):
        pass
    
class MTTransformer(nn.Module):
    
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
        decoder_layer = TransformerDecoderLayer(**t_params)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, decoder_layers)
        
        self.tok_head = nn.Linear(d_model, vocab_size)
        
    def embed(self, x):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x
    
    def forward(self, batch):
        src_ids = batch['src_ids']
        tgt_ids = batch['tgt_ids']
        
        N = tgt_ids.shape[1] 
        device = tgt_ids.device
        
        src = self.embed(src_ids)
        tgt = self.embed(tgt_ids)
        
        pad_mask = src_ids.eq(self.pad_token_id)
        causal_mask = T.generate_square_subsequent_mask(N, dtype=torch.bool, device=device)
        
        mem = self.encoder(src, src_key_padding_mask=pad_mask)
        h, _ = self.decoder(tgt, mem, tgt_is_causal=True, tgt_mask=causal_mask, memory_key_padding_mask=pad_mask)
        tok_probs = F.log_softmax(self.tok_head(h), dim=-1)
        
        return tok_probs
       
    def step(self, batch, reduce=True):
        tok_probs = self.forward(batch)
        return F.nll_loss(tok_probs[:, :-1].transpose(1, 2), batch['tgt_ids'][:, 1:], ignore_index=self.pad_token_id, reduction='mean' if reduce else 'sum')
    
    @torch.no_grad() 
    def generate(self, src_ids, temperature=1.0, **_):
        B = src_ids.shape[0]
        device = src_ids.device
        
        tgt_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(self.max_len):
            logits = self.forward({'src_ids': src_ids, 'tgt_ids': tgt_ids})
            next_tok_logits = logits[:, -1, :] / temperature
            next_tok = torch.multinomial(F.softmax(next_tok_logits, dim=-1), num_samples=1)
            tgt_ids = torch.cat([tgt_ids, next_tok], dim=-1)
            finished |= (next_tok.squeeze(-1) == self.eos_token_id)
            if finished.all(): break
            
        return tgt_ids
    
def init_model(config, tokenizer):
    if config['model_type'] == 'decoder_only':
        return MTTransformer(
            d_model=config['d_model'],
            dim_feedforward=config['dim_feedforward'],
            nhead=config['nhead'],
            dropout=config['dropout'],
            layer_norm_eps=config['layer_norm_eps'],
            decoder_layers=config['decoder_layers'],
            encoder_layers=config['encoder_layers'],
            vocab_size=tokenizer.vocab_size,
            max_len=config['max_len'],
            pad_token_id=tokenizer.vocab['PAD'],
            bos_token_id=tokenizer.vocab['BOS'],
            eos_token_id=tokenizer.vocab['EOS'],
            name=config['name']
        ).to(config['device'])
        
    raise NotImplementedError()
    
    return Evolver(
        d_model=config['d_model'],
        dim_feedforward=config['dim_feedforward'],
        nhead=config['nhead'],
        dropout=config['dropout'],
        layer_norm_eps=config['layer_norm_eps'],
        decoder_layers=config['decoder_layers'],
        encoder_layers=config['encoder_layers'],
        vocab_size=csg.vocab_size,
        max_len=config['max_len'],
        pad_token_id=csg.tok_to_id['PAD'],
        bos_token_id=csg.tok_to_id['BOS'],
        eos_token_id=csg.tok_to_id['EOS'],
        root_id=csg.tok_to_id['s'],
        csg=csg,
        name=config['name'],
        static=config.get('static', False)
    ).to(config['device'])
    
 
def load_checkpoint(model, optimizer, config):
    if config['from_checkpoint']:
        checkpoint = torch.load(config['from_checkpoint'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step'] + 1
        logger.info(f'resuming from step {start_step}')
        return start_step
    return 0

def save_checkpoint(model, optimizer, step, config):
    save_path = os.path.join(config['checkpoint_dir'], f'{model.name}_{step}.pt')
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
    
def train_step(model, batch, device):
    if isinstance(model, MTTransformer):
        return model.step({k: v.to(device) for k, v in batch.items()})
    raise NotImplementedError()

@torch.no_grad()
def evaluate(model, eval_loader, device, num_eval_steps, tokenizer):
    model.eval()
    
    tot_loss = 0
    hyps = []
    refs = []
    
    for i, batch in enumerate(tqdm(eval_loader, desc="eval...", total=num_eval_steps)):
        if i >= num_eval_steps: break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        if isinstance(model, MTTransformer):
            loss = model.step(batch)
            tot_loss += loss.item()
            generated_ids = model.generate(batch['src_ids'])
            
        else: raise NotImplementedError()
        
        for hyp, ref in zip(generated_ids, batch['tgt_ids']):
            hyp_text = tokenizer.decode(hyp)
            ref_text = tokenizer.decode(ref)
            hyps.append(hyp_text)
            refs.append(ref_text)
    
    return tot_loss / num_eval_steps, corpus_bleu(hyps, [refs]).score

def train(config):
    device = torch.device(config['device'])
    tokenizer = SpacyTokenizer()
    
    train_dataset = MTDataset(split='train', max_len=config['max_len'], buffer_size=config['buffer_size'], tokenizer=tokenizer)
    eval_dataset = MTDataset(split='validation', max_len=config['max_len'], buffer_size=config['buffer_size'], tokenizer=tokenizer)
    logger.info('loaded datasets')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=train_dataset.collate_fn, num_workers=config['num_workers'])
    eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], collate_fn=eval_dataset.collate_fn, num_workers=config['num_workers'])
    
    model = init_model(config, tokenizer)
    optim = AdamW(model.parameters(), lr=config['lr'])
    start_step = load_checkpoint(model, optim, config)
    
    logger.info('eval sanity check')
    evaluate(model, eval_loader, device, 1, tokenizer)
    logger.info('passed!')
    
    model.train()
    for step, batch in tqdm(
        enumerate(train_loader, start=start_step),
        total=config['train_steps'],
        disable=config['local']
    ):
        if step >= config['train_steps']: break
        
        loss = train_step(model, batch, device)
        loss.backward()

        if (step + 1) % config['grad_accum_steps'] == 0:
            optim.step()
            optim.zero_grad()

        if step % config['log_every'] == 0:
            log_to_wandb({'train/loss': loss.item()}, step=step)

        if step % config['eval_every'] == 0:
            eval_loss, bleu_score = evaluate(model, eval_loader, device, config['num_eval_steps'], tokenizer)
            log_to_wandb({'eval/loss': eval_loss, 'eval/bleu': bleu_score}, step=step)
            model.train()

        if step % config['save_every'] == 0:
            save_checkpoint(model, optim, step, config)

    eval_loss, bleu_score = evaluate(model, eval_loader, device, config['num_eval_steps'], tokenizer)
    log_to_wandb({'eval/loss': eval_loss, 'eval/bleu': bleu_score}, step=step)
    save_checkpoint(model, optim, config['train_steps'], config)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--from-checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    config['device'] = args.device
    config['local'] = args.local
    config['from_checkpoint'] = args.from_checkpoint
    config['name'] = f"mt_{config['model_type']}_{config['d_model']}d_{config.get('encoder_layers', 0)}enc_{config['decoder_layers']}dec-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not config['local']: wandb.init(project='mt-evolver', name=config['name'], config=config, resume='allow')
    train(config)

if __name__ == '__main__':
    main()
