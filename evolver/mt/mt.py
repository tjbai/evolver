import os
import json
import math
import yaml
import random
import logging
import argparse
from datetime import datetime
from collections import defaultdict

import spacy
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
from sacrebleu import corpus_bleu
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer as TransformersBertTokenizer
from transformers import MarianTokenizer as TransformersMarianTokenizer

from evolver.embed import SinusoidalEmbedding
from evolver.trans import TransformerEncoderLayer, TransformerEncoder, CausalTransformerDecoderLayer, CausalTransformerDecoder, MultiheadPointer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_wandb(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f'step {step}: {data}')
    
def pad(seqs, padding_value, max_len=int(1e10)):
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=padding_value)[:, :max_len]

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
        for id in tok_ids.tolist() if isinstance(tok_ids, torch.Tensor) else tok_ids:
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
    
class MarianTokenizer:
    
    def __init__(self):
        self.tokenizer = TransformersMarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        self.vocab_size = self.tokenizer.vocab_size + 1
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.bos_token_id = self.tokenizer.vocab_size

    def get_id(self, t):
        if t == '<s>': return self.bos_token_id
        return self.tokenizer.get_vocab().get(t, self.unk_token_id)

    def encode(self, text, **_):
        return [self.bos_token_id] + self.tokenizer(text)['input_ids']

    def decode(self, tok_ids, skip_special_tokens=True, **_):
        if skip_special_tokens: return self.tokenizer.decode(tok_ids[1:], skip_special_tokens=True)
        else: return '<s>' + self.tokenizer.decode(tok_ids[1:], skip_special_tokens=False)
        
class Parser:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.marian = TransformersMarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
    
    def get_spans(self, text, tokens):
        spans = []
        i = 0
        for token in tokens:
            while i < len(text) and text[i].isspace(): i += 1
            if i < len(text):
                s = i
                i = text.find(token, i) + len(token)
                spans.append((s, i))
        
        assert len(spans) == len(tokens), f'number of spans does not match number of tokens for: {text}'
        return spans

    def get_alignment(self, text, spacy_tokens, marian_tokens, spacy_spans=None, marian_spans=None):
        if spacy_spans is None: spacy_spans = self.get_spans(spacy_tokens) 
        if marian_tokens is None: marian_spans = self.get_spans(marian_tokens) 
        
        alignment = {} # map marian_tokens[i] to spacy_tokens[j]
        best_overlap = defaultdict(int) # track max overlap
        
        # just bruteforce check (who needs DP?)
        for i, marian_span in enumerate(marian_spans):
            for j, spacy_span in enumerate(spacy_spans):
                overlap = max(0, min(spacy_span[1], marian_span[1]) - max(spacy_span[0], marian_span[0]))
                if overlap > 0 and overlap > best_overlap[i]:
                    alignment[i] = j
                    best_overlap[i] = overlap
                    
        for i, tok in enumerate(marian_tokens):
            if tok == '': alignment[i] = alignment[i+1] # word break
            if tok == '<unk>': alignment[i] = 1 + alignment[i-1] # unk
        
        assert len(alignment) == len(marian_tokens), f'did not find a complete alignment for: {text}'
        return alignment

    def get_spacy_tokens(self, text):
        return [token.text for token in self.nlp(text)]

    def get_marian_tokens(self, text):
        return [self.marian.decode(id) for id in self.marian(text)['input_ids'][:-1]]
    
    def parse(self, text):
        spacy_tokens = self.get_spacy_tokens(text)
        spacy_spans = self.get_spans(text, spacy_tokens)
        normalized_marian_tokens = self.get_marian_tokens(text)
        marian_spans = self.get_spans(text, normalized_marian_tokens)

        raw_marian_tokens = self.marian.tokenize(text)
        doc = self.nlp(text)
        alignment = self.get_alignment(text, spacy_tokens, normalized_marian_tokens, spacy_spans, marian_spans)
    
        # reverse map spacy to marian tokens
        reverse = defaultdict(list)
        for k, v in alignment.items(): reverse[v].append(k)

        # get original root idx
        root_i = next(i for i, tok in enumerate(doc) if tok.head == tok)

        seq = []
        for i, text in enumerate(raw_marian_tokens):
            spacy_tok = doc[alignment[i]]
            seq.append({'text': text, 'pos': spacy_tok.pos_, 'i': i, 'is_head': alignment[i] == root_i})
        
        for i, _ in enumerate(seq):
            spacy_children = doc[alignment[i]].children
            seq[i]['children'] = []
            for child in spacy_children:
                seq[i]['children'].extend(seq[i] for i in reverse[child.i])
            
            ### heuristic for lineage
            # spacy_parent = doc[alignment[i]].head.i
            # inferred_parent = reverse[spacy_parent][0]
            seq[i]['par'] = reverse[doc[alignment[i]].head.i][0]
            
        return seq
    
class WMT(Dataset):

    def __init__(self, split='train', max_len=256, tokenizer=MarianTokenizer(), truncate=None):
        self.dataset = load_dataset('wmt14', 'de-en', split=split)
        self.max_len = max_len
        self.tokenizer = tokenizer
        if truncate is not None: self.dataset = self.dataset.select(range(truncate))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]['translation']
        src_ids = self.tokenizer.encode(item['de'], lang='de')[:self.max_len]
        tgt_ids = self.tokenizer.encode(item['en'], lang='en')[:self.max_len]
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}
        
    def collate_fn(self, batch):
        src_ids = [item['src_ids'] for item in batch]
        tgt_ids = [item['tgt_ids'] for item in batch]
        src_ids = pad([torch.tensor(ids) for ids in src_ids], self.tokenizer.pad_token_id)
        tgt_ids = pad([torch.tensor(ids) for ids in tgt_ids], self.tokenizer.pad_token_id)
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}
    
class WMTForEvolver(WMT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = Parser()
    
    def _get_depth(self, doc):
        root = next(tok for tok in doc if tok['is_head'])
        def dfs(node):
            r = 1
            for child in node['children']: r = max(r, 1 + dfs(child))
            return r
        return dfs(root)
    
    # uses fake spacy api
    def _get_input_traj(self, doc):
        def aux(token, depth):
            for i in range(depth, len(traj)):
                traj[i][token['i']] = (token['text'] if (i > depth+1) else token['pos'], token['i'], token['par'])

            traj[depth+1][token['i']] = (token['text'], token['i'], token['par'])
            for child in token['children']: aux(child, depth+2)
        
        traj = [['_' for _ in range(len(doc))] for _ in range(2*self._get_depth(doc))]
        for tok in doc:
            if tok['is_head']: aux(tok, 0)
        return traj 
   
    def _get_short_input_traj(self, doc):
        return self._get_input_traj(doc)[1::2]
    
    def _get_output_traj(self, doc):
        get_id = self.tokenizer.get_id
        INS, CPY, SUB = 0, 1, 2
        
        # NOTE -- this the reduced version for scaling efficiency
        input_traj = self._get_short_input_traj(doc)

        # prefill empty string x0 
        output_traj = []
        traj_ids = []
        output_traj.append([(INS, self.tokenizer.bos_token_id, -1), (INS, self.tokenizer.eos_token_id, -1)])
        traj_ids.append([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id])
       
        # prefill root sequence 
        par_idx = {} 
        output_traj.append([(INS, self.tokenizer.bos_token_id, -1)])
        traj_ids.append([(self.tokenizer.bos_token_id)])
        k = 1
        for tok in doc:
            if not tok['is_head']: continue
            output_traj[-1].append((INS, get_id(tok['text']), -1))
            par_idx[tok['i']] = k
            k += 1
        output_traj[-1].append((INS, self.tokenizer.eos_token_id, -1))
        traj_ids[-1].append(self.tokenizer.eos_token_id)

        # fill in subsequent steps
        for seq in input_traj[1:]:
            k = 1
            new_par_idx = {}
            cur_edits = [(INS, self.tokenizer.bos_token_id, -1)]
            cur_ids = [self.tokenizer.bos_token_id]
            
            for t in seq:
                if t == '_':
                    continue
                elif t[1] in par_idx:
                    assert par_idx[t[1]] < len(output_traj[-1]), f'{par_idx[t[1]]} is too large for prev seq size of {len(output_traj[-1])}'
                    cur_edits.append((CPY, -1, par_idx[t[1]]))
                else:
                    assert par_idx[t[2]] < len(output_traj[-1]), f'{par_idx[t[2]]} is too large for prev seq size of {len(output_traj[-1])}'
                    cur_edits.append((SUB, get_id(t[0]), par_idx[t[2]]))
                cur_ids.append(get_id(t[0]))
                new_par_idx[t[1]] = k
                k += 1
            
            par_idx = new_par_idx 
            cur_edits.append((INS, self.tokenizer.eos_token_id, -1))
            cur_ids.append(self.tokenizer.eos_token_id)
            output_traj.append(cur_edits)
            traj_ids.append(cur_ids)
        
        return output_traj, traj_ids

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_ids = self.tokenizer.encode(item['translation']['de'])
        
        parsed = self.parser.parse(item['translation']['en'])
        output_traj, traj_ids = self._get_output_traj(parsed)
        
        t = random.randint(0, len(output_traj)-2)
        target_seq = output_traj[t+1]
        edit_ids = ([x[0] for x in target_seq], [x[1] for x in target_seq], [x[2] for x in target_seq])

        return {'src_ids': src_ids, 'input_ids': traj_ids[t], 'tgt_ids': traj_ids[t+1], 'edit_ids': edit_ids}
    
    def collate_fn(self, batch):
        src_ids = pad([torch.tensor(item['src_ids']) for item in batch], self.tokenizer.pad_token_id, self.max_len)
        input_ids = pad([torch.tensor(item['input_ids']) for item in batch], self.tokenizer.pad_token_id, self.max_len)
        tgt_ids = pad([torch.tensor(item['tgt_ids']) for item in batch], self.tokenizer.pad_token_id, self.max_len)
        op_ids = pad([torch.tensor(item['edit_ids'][0]) for item in batch], -1, self.max_len)
        tok_ids = pad([torch.tensor(item['edit_ids'][1]) for item in batch], -1, self.max_len)
        idx_ids = pad([torch.tensor(item['edit_ids'][2]) for item in batch], -1, self.max_len)
        return {'src_ids': src_ids, 'input_ids': input_ids, 'tgt_ids': tgt_ids, 'edit_ids': (op_ids, tok_ids, idx_ids)}

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
    
    # NOTE -- rollout should start from empty string
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

def save_checkpoint(model, optimizer, step, config):
    save_path = os.path.join(config['checkpoint_dir'], f'{model.name}_{step}.pt')
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
    
def train_step(model, batch, device, step=None):
    if isinstance(model, Transformer):
        return model.step({k: v.to(device) for k, v in batch.items()})
    else:
        loss = model.step({
            'src_ids': batch['src_ids'].to(device),
            'input_ids': batch['input_ids'].to(device),
            'tgt_ids': batch['tgt_ids'].to(device),
            'edit_ids': tuple(map(lambda x: x.to(device), batch['edit_ids']))
        })

        if step is not None and isinstance(model, Evolver):
            log_to_wandb({
                'train/op_loss': loss[0],
                'train/tok_loss': loss[1],
                'train/idx_loss': loss[2]}, step=step)

        return sum(loss) if isinstance(model, Evolver) else loss

@torch.no_grad()
def evaluate(model, eval_loader, device, num_eval_steps, tokenizer):
    model.eval()
    
    tot_loss = 0
    hyps = []
    refs = []
    
    for i, batch in enumerate(tqdm(eval_loader, desc="eval...", total=num_eval_steps)):
        if i >= num_eval_steps: break
        
        if isinstance(model, Transformer):
            loss = train_step(model, batch, device)
            tot_loss += loss.item()
            generated = model.generate(batch['src_ids'].to(device))
            
        else:
            loss = train_step(model, batch, device)
            tot_loss += loss.item()
        
        # TODO -- bring back sampling to test
        #     traj = model.rollout({
        #         'src_ids': batch['src_ids'].to(device),
        #         'input_ids': batch['input_ids'].to(device),
        #         'tgt_ids': batch['tgt_ids'].to(device),
        #         'edit_ids': tuple(map(lambda x: x.to(device), batch['edit_ids']))})
        #     *_, generated = traj
        
        # for hyp, ref in zip(generated, batch['tgt_ids']):
        #     hyp_text = tokenizer.decode(hyp)
        #     ref_text = tokenizer.decode(ref)
        #     hyps.append(hyp_text)
        #     refs.append(ref_text)
            
    # logger.info(f'sample hyp: {hyps[0]}')
    # logger.info(f'sample ref: {refs[0]}')
    
    # score = corpus_bleu(hyps, [refs]).score
    score = 0

    return tot_loss / num_eval_steps, score
    
def init_model(config, tokenizer):
    params = {
        'd_model': config['d_model'],
        'dim_feedforward': config['dim_feedforward'],
        'nhead': config['nhead'],
        'dropout': config['dropout'],
        'layer_norm_eps': config['layer_norm_eps'],
        'decoder_layers': config['decoder_layers'],
        'encoder_layers': config['encoder_layers'],
        'vocab_size': tokenizer.vocab_size,
        'max_len': config['max_len'],
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'name': config['name']
    }
    
    if config['model_type'] == 'decoder_only':
        return Transformer(**params).to(config['device'])
    elif config['model_type'] == 'evolver':
        return Evolver(**params).to(config['device'])
    else:
        return Teacher(**params).to(config['device'])
    
def load_checkpoint(model, optimizer, config):
    if config.get('from_checkpoint') is not None:
        checkpoint = torch.load(config['from_checkpoint'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step'] + 1
        logger.info(f'resuming from step {start_step}')
        return start_step
    return 0

def train(config):
    device = torch.device(config['device'])
    
    # tokenizer = MarianTokenizer() if config['model_type'] == 'decoder_only' else SpacyTokenizer()
    tokenizer = MarianTokenizer()
    
    dataset = WMT if config['model_type'] == 'decoder_only' else WMTForEvolver
    train_dataset = dataset(split='train', max_len=config['max_len'], tokenizer=tokenizer, truncate=config.get('truncate'))
    eval_dataset = dataset(split='validation', max_len=config['max_len'], tokenizer=tokenizer, truncate=config.get('truncate'))
    logger.info(f'loaded {dataset} datasets and {type(tokenizer)} tokenizer')
    
    kwargs = {'batch_size': config['batch_size'], 'collate_fn': train_dataset.collate_fn, 'num_workers': config['num_workers']}
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), **kwargs)
    eval_loader = DataLoader(eval_dataset, **kwargs)
   
    model = init_model(config, tokenizer)
    optim = AdamW(model.parameters(), lr=config['lr'])
    step = load_checkpoint(model, optim, config)
    
    if not config['skip']:
        logger.info('eval sanity check')
        evaluate(model, eval_loader, device, 1, tokenizer)
        logger.info('sanity check passed')
    
    model.train()
    for _ in range(config['train_epochs']):
        for batch in tqdm(train_loader):
            if step >= config['train_steps']: break
            
            loss = train_step(model, batch, device, step=step)
            loss.backward()

            if (step + 1) % config['grad_accum_steps'] == 0:
                optim.step()
                optim.zero_grad()

            if (step + 1) % config['log_every'] == 0:
                log_to_wandb({'train/loss': loss.item()}, step=step)

            if (step + 1) % config['eval_every'] == 0:
                eval_loss, bleu_score = evaluate(model, eval_loader, device, config['num_eval_steps'], tokenizer)
                log_to_wandb({'eval/loss': eval_loss, 'eval/bleu': bleu_score}, step=step)
                model.train()

            if (step + 1) % config['save_every'] == 0:
                save_checkpoint(model, optim, step, config)
            
            step += 1

    eval_loss, bleu_score = evaluate(model, eval_loader, device, config['num_eval_steps'], tokenizer)
    log_to_wandb({'eval/loss': eval_loss, 'eval/bleu': bleu_score}, step=step)
    save_checkpoint(model, optim, config['train_steps'], config)

def load_config(config_path):
    with open(config_path, 'r') as f:
        if config_path[-3:] == 'yml':
            return yaml.safe_load(f)
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--skip', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    config['device'] = args.device
    config['local'] = args.local
    config['skip'] = args.skip
    config['name'] = f"mt_{config['model_type']}_{config['d_model']}d_{config.get('encoder_layers', 0)}enc_{config['decoder_layers']}dec-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not config['local']: wandb.init(project='mt-evolver', name=config['name'], config=config, resume='allow')
    train(config)

if __name__ == '__main__':
    main()
