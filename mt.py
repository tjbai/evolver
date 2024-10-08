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
from transformers import BertTokenizer as TransformersBertTokenizer

from embed import SinusoidalEmbedding
from trans import (
    TransformerEncoderLayer,
    TransformerEncoder,
    CausalTransformerDecoderLayer,
    CausalTransformerDecoder,
    MultiheadPointer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_wandb(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f'step {step}: {data}')

class SpacyTokenizer:
    
    def __init__(self):
        self.de_nlp = spacy.load('de_core_news_sm')
        self.en_nlp = spacy.load('en_core_web_sm')
        
        self.vocab = {'BOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        with open('vocab/wmt14_de_en.vocab', 'r') as f:
            i = 4
            for _t in f:
                t = _t[:-1]
                if t not in self.vocab:
                    self.vocab[t] = i
                    i += 1
                
        self.id_to_tok = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.unk_token_id = 3
        
    def get_id(self, t):
        return self.vocab.get(t, self.unk_token_id)
        
    def encode(self, text, lang, add_special_tokens=True):
        doc = {'de': self.de_nlp, 'en': self.en_nlp}[lang](text)
        tokens = [self.get_id(token.text) for token in doc]
        if add_special_tokens: tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens
    
    def decode(self, tok_ids, skip_special_tokens=True):
        tokens = []
        for id in tok_ids.tolist():
            token = self.id_to_tok.get(id, 'UNK')
            if not skip_special_tokens or (token not in {'PAD', 'BOS', 'EOS'}): tokens.append(token)
            if token == 'EOS': break
        return ' '.join(tokens)

class BertTokenizer:
    
    def __init__(self):
        self.tokenizer = TransformersBertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 100
        self.bos_token_id = 101
        self.eos_token_id = 102

    def get_id(self, t):
        return self.tokenizer.get_vocab().get(t, self.unk_token_id)

    def encode(self, text, **_):
        return self.tokenizer(text)['input_ids']

    def decode(self, tok_ids, skip_special_tokens=True, **_):
        return self.tokenizer.decode(tok_ids, skip_special_tokens=skip_special_tokens)
    
class MTDataset(Dataset):

    def __init__(self, split='train', max_len=256, buffer_size=1000, tokenizer=SpacyTokenizer()):
        self.dataset = load_dataset('wmt14', 'de-en', split=split)
        self.max_len = max_len
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
            padding_value=self.tokenizer.pad_token_id
        )
        
        tgt_ids_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in tgt_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        return {'src_ids': src_ids_padded, 'tgt_ids': tgt_ids_padded}
    
class MTEditDataset(MTDataset):
    
    def depth(self, doc):
        root = [tok for tok in doc if tok.head == tok] [0]
        def dfs(node):
            r = 1
            for child in node.children: r = max(r, 1 + dfs(child))
            return r
        return dfs(root)
    
    def gen(self, doc):
        get_id = self.tokenizer.get_id
        INS, CPY, SUB = 0, 1, 2 
       
        traj = [['_' for _ in range(len(doc))] for _ in range(2*self.depth(doc))]
        
        def traverse(token, depth):
            for i in range(depth, len(traj)):
                traj[i][token.i] = (token.text if (i > depth+1) else token.pos_, token.i, token.head.i)

            traj[depth+1][token.i] = (token.text, token.i, token.head.i)
            for child in token.children: traverse(child, depth+2)
        
        root = next(token for token in doc if token.head == token)
        traverse(root, 0)
        
        m = {}
        input_ids = [[0, get_id(root.pos_), 1]]
        
        op_ids = []
        tok_ids = []
        idx_ids = []
        
        last_len = 3
        for i, seq in enumerate(traj[1:]):
            cur_edits = [(CPY, -1, 0)]
            
            if i % 2 == 0:
                k = 1
                for t in seq:
                    if t == '_': continue
                    if t[1] in m: cur_edits.append((CPY, -1, k))
                    else: cur_edits.append((SUB, get_id(t[0]), k))
                    m[t[1]] = k
                    k += 1
                    
            else:
                k = 1
                for t in seq:
                    if t == '_': continue
                    if t[1] in m: cur_edits.append((CPY, -1, m[t[1]]))
                    # NOTE -- let's call this INS for now...
                    else: cur_edits.append((INS, get_id(t[0]), m[t[2]]))
            
            input_ids.append([get_id('BOS')] + [get_id(t[0]) for t in seq if t != '_'] + [get_id('EOS')])
            cur_edits.append((CPY, -1, last_len - 1))
            last_len = len(cur_edits)
            
            ops, toks, idxs = zip(*cur_edits)
            op_ids.append(ops)
            tok_ids.append(toks)
            idx_ids.append(idxs)
        
        return input_ids, (op_ids, tok_ids, idx_ids)
    
    def __getitem__(self, _):
        if self.buffer_index >= len(self.buffer): self.refill_buffer()
        item = self.buffer[self.buffer_index]
        self.buffer_index += 1
        
        src_ids = self.tokenizer.encode(item['de'], lang='de')[:self.max_len]
        input_ids, edit_ids = self.gen(self.tokenizer.en_nlp(item['en']))
        
        t = random.randint(0, len(input_ids)-2)
        return {'src_ids': src_ids, 'input_ids': input_ids[t], 'tgt_ids': input_ids[t+1], 'edit_ids': tuple(map(lambda x: x[t], edit_ids))}
    
    def collate_fn(self, batch):
        src_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['src_ids']) for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['input_ids']) for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        tgt_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['tgt_ids']) for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        op_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['edit_ids'][0]) for item in batch],
            batch_first=True,
            padding_value=-1
        )
       
        tok_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['edit_ids'][1]) for item in batch],
            batch_first=True,
            padding_value=-1
        )
        
        idx_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['edit_ids'][2]) for item in batch],
            batch_first=True,
            padding_value=-1
        )
        
        return {'src_ids': src_ids, 'input_ids': input_ids, 'tgt_ids': tgt_ids, 'edit_ids': (op_ids, tok_ids, idx_ids)}

class MTEvolver(nn.Module):

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
            'layer_norm_eps': layer_norm_eps
        }
        
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
            add_zero_attn=False
        )
        
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
    
    def _generate(self, batch, max_steps):
        B = batch['src_ids'].shape[0]
        device = batch['src_ids'].device
        
        op_ids = torch.full((B, 1), 1, dtype=torch.long, device=device)
        tok_ids = torch.full((B, 1), -1, dtype=torch.long, device=device)
        idx_ids = torch.full((B, 1), 0, dtype=torch.long, device=device)
        
        alive = torch.ones(B, dtype=torch.bool)
       
        cache = None 
        for _ in range(max_steps):
            if not alive.any(): break
            
            batch = {'src_ids': batch['src_ids'], 'input_ids': batch['input_ids'], 'tgt_ids': batch['tgt_ids'], 'edit_ids': (op_ids, tok_ids, idx_ids)}
            probs, cache = self.forward(batch, cache=cache)
            next_op, next_tok, next_idx = tuple(map(lambda x: x[:, -1], probs))
            
            op_id = torch.multinomial(next_op.exp(), num_samples=1)
            tok_id = torch.multinomial(next_tok.exp(), num_samples=1)
            idx_id = torch.multinomial(next_idx.exp(), num_samples=1)
            
            idx_id[op_id.eq(0)] = -1
            tok_id[op_id.eq(1)] = -1

            op_ids = torch.cat([op_ids, op_id], dim=1)
            tok_ids = torch.cat([tok_ids, tok_id], dim=1)
            idx_ids = torch.cat([idx_ids, idx_id], dim=1)

            tgt_ids = self.apply_edits(batch['input_ids'], (op_ids, tok_ids, idx_ids))
            alive[tgt_ids[:, -1] == self.eos_token_id] = False
            
        return tgt_ids
    
    def generate(self, batch, max_depth=10, max_steps=128):
        self.decoder.set_causal()
        traj = [batch['input_ids']]
        for _ in range(max_depth):
            batch = {'src_ids': batch['src_ids'], 'input_ids': traj[-1], 'tgt_ids': batch['tgt_ids'], 'edit_ids': batch['edit_ids']}
            traj.append(self._generate(batch, max_steps))
        self.decoder.set_parallel()
        return traj[-1]
    
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
        tok_probs = F.log_softmax(self.tok_head(h), dim=-1)
        
        return tok_probs, cache
       
    def step(self, batch, reduce=True):
        tok_probs, _ = self.forward(batch)
        return F.nll_loss(tok_probs[:, :-1].transpose(1, 2), batch['tgt_ids'][:, 1:], ignore_index=self.pad_token_id, reduction='mean' if reduce else 'sum')
    
    @torch.no_grad()
    def generate(self, src_ids, temperature=1.0, **_):
        B = src_ids.shape[0]
        device = src_ids.device
        self.decoder.set_causal()
        
        tgt_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        cache = None
        for _ in range(self.max_len):
            logits, cache = self.forward({'src_ids': src_ids.to(device), 'tgt_ids': tgt_ids.to(device)}, cache=cache)
            next_tok_logits = logits[:, -1, :] / temperature
            next_tok = torch.multinomial(F.softmax(next_tok_logits, dim=-1), num_samples=1)
            tgt_ids = torch.cat([tgt_ids, next_tok], dim=-1)
            finished |= (next_tok.squeeze(-1) == self.eos_token_id)
            if finished.all(): break
        
        self.decoder.set_parallel()
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
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            name=config['name']
        ).to(config['device'])
        
    return MTEvolver(
        d_model=config['d_model'],
        dim_feedforward=config['dim_feedforward'],
        nhead=config['nhead'],
        dropout=config['dropout'],
        layer_norm_eps=config['layer_norm_eps'],
        decoder_layers=config['decoder_layers'],
        encoder_layers=config['encoder_layers'],
        vocab_size=tokenizer.vocab_size,
        max_len=config['max_len'],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        name=config['name'],
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
    
def train_step(model, batch, device, step=None):
    if isinstance(model, MTTransformer):
        return model.step({k: v.to(device) for k, v in batch.items()})
    else:
        op_loss, tok_loss, idx_loss = model.step({
            'src_ids': batch['src_ids'].to(device),
            'input_ids': batch['input_ids'].to(device),
            'tgt_ids': batch['tgt_ids'].to(device),
            'edit_ids': tuple(map(lambda x: x.to(device), batch['edit_ids']))
        })
        if step is not None:
            log_to_wandb({
                'train/op_loss': op_loss,
                'train/tok_loss': tok_loss,
                'train/idx_loss': idx_loss},
            step=step)
        return op_loss + tok_loss + idx_loss

@torch.no_grad()
def evaluate(model, eval_loader, device, num_eval_steps, tokenizer):
    model.eval()
    
    tot_loss = 0
    hyps = []
    refs = []
    
    for i, batch in enumerate(tqdm(eval_loader, desc="eval...", total=num_eval_steps)):
        if i >= num_eval_steps: break
        
        if isinstance(model, MTTransformer):
            loss = train_step(model, batch, device)
            tot_loss += loss.item()
            generated_ids = model.generate(batch['src_ids'].to(device))
            
        else:
            loss = train_step(model, batch, device)
            tot_loss += loss.item()
            generated_ids = model.generate({
                'src_ids': batch['src_ids'].to(device),
                'input_ids': batch['input_ids'].to(device),
                'tgt_ids': batch['tgt_ids'].to(device),
                'edit_ids': tuple(map(lambda x: x.to(device), batch['edit_ids']))
            })
        
        for hyp, ref in zip(generated_ids, batch['tgt_ids']):
            hyp_text = tokenizer.decode(hyp)
            ref_text = tokenizer.decode(ref)
            hyps.append(hyp_text)
            refs.append(ref_text)
    
    return tot_loss / num_eval_steps, corpus_bleu(hyps, [refs]).score

def train(config):
    device = torch.device(config['device'])
    
    tokenizer = BertTokenizer() if config['model_type'] == 'decoder_only' else SpacyTokenizer()

    dataset = MTDataset if config['model_type'] == 'decoder_only' else MTEditDataset
    train_dataset = dataset(split='train', max_len=config['max_len'], buffer_size=config['buffer_size'], tokenizer=tokenizer)
    eval_dataset = dataset(split='validation', max_len=config['max_len'], buffer_size=config['buffer_size'], tokenizer=tokenizer)
    logger.info('loaded datasets')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=train_dataset.collate_fn, num_workers=config['num_workers'])
    eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], collate_fn=eval_dataset.collate_fn, num_workers=config['num_workers'])
    
    model = init_model(config, tokenizer)
    optim = AdamW(model.parameters(), lr=config['lr'])
    start_step = load_checkpoint(model, optim, config)
    
    model.train()
    for step, batch in tqdm(
        enumerate(train_loader, start=start_step),
        total=config['train_steps'],
        disable=config['local']
    ):
        if step >= config['train_steps']: break
        
        loss = train_step(model, batch, device, step=step)
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
