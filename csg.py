import os
import time
import json
import math
import random
import logging
import argparse
from PIL import Image, ImageDraw, ImageChops
from functools import wraps
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import Transformer as T
from torchvision import transforms
from tqdm import tqdm

from const import *
from embed import SinusoidalEmbedding
from trans import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, MultiheadPointer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_wandb(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f'step {step}: {data}')
    
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logger.info(f'{f.__name__}: {te-ts}')
        return result
    return wrap

class CSG:
    def __init__(self):
        self.rules = {
            's': ([['binop'], ['circle'], ['quad']], [2, 1, 1]),
            'binop': ([['(', 'op', 's', 's', ')']], [1]),
            'op': ([['Add'], ['Sub']], [2, 1]),
            'circle': ([['(', 'Circle', 'num', 'num', 'num', ')']], [1]),
            'quad': ([['(', 'Quad', 'num', 'num', 'num',  'num', 'angle', ')']], [1]),
            'num': (list(['(', 'Num', str(i), ')'] for i in range(16)), [1 for _ in range(16)]),
            'angle': (list(['(', 'Angle', str(i), ')'] for i in range(0, 316, 45)), [1 for _ in range(0, 316, 45)])
        }
        
        self._map()
        
    def _map(self):
        self.tok_to_id = dict()
        self.id_to_tok = dict() 
        
        toks = set(self.rules.keys())
        toks.add('BOS')
        toks.add('EOS')
        toks.add('PAD')
        
        for expansions, _ in self.rules.values():
            for expansion in expansions:
                toks.update(expansion)
        
        for i, token in enumerate(sorted(toks)):
            self.tok_to_id[token] = i
            self.id_to_tok[i] = token
            
        self.vocab_size = len(self.tok_to_id)
        
    def generate(self, symbol='s', depth=0, max_depth=5):
        if symbol not in self.rules: return symbol
        if depth >= max_depth and symbol == 's': expansion = random.choice([['circle'], ['quad']])
        else: expansion = random.choices(self.rules[symbol][0], weights=self.rules[symbol][1], k=1)[0]
        return ' '.join(self.generate(const, depth+1) for const in expansion)
    
    def parse(self, program):
        return self._parse(program.split())
    
    def _parse(self, tokens):
        if tokens[1] == 'Circle': return {'type': 's', 'value': self.parse_circle(tokens)}
        if tokens[1] == 'Quad': return {'type': 's', 'value': self.parse_quad(tokens)}
        return {'type': 's', 'value': self.parse_binop(tokens)}

    def find_split(self, tokens):
        count = 0
        for i, token in enumerate(tokens):
            if token == '(': count += 1
            elif token == ')': count -= 1
            if count == 0: return i + 1
    
    def find_all(self, tokens, match, expected):
        res = []
        for i, token in enumerate(tokens):
            if len(res) == expected: break
            if token == match: res.append(i)
        return res
    
    def parse_op(self, token):
        return {'type': 'op', 'value': token}

    def parse_binop(self, tokens):
        i = self.find_split(tokens[2:-1])
        return {
            'type': 'binop',
            'op': self.parse_op(tokens[1]),
            'left': self._parse(tokens[2:2+i]),
            'right': self._parse(tokens[2+i:-1])
        }

    def parse_circle(self, tokens):
        r, x, y = self.find_all(tokens[1:], '(', 3)
        return {
            'type': 'circle',
            'r': self.parse_num(tokens[r+1:r+5]),
            'x': self.parse_num(tokens[x+1:x+5]),
            'y': self.parse_num(tokens[y+1:y+5])
        }

    def parse_quad(self, tokens):
        x, y, w, h, theta = self.find_all(tokens[1:], '(', 5)
        return {
            'type': 'quad',
            'x': self.parse_num(tokens[x+1:x+5]),
            'y': self.parse_num(tokens[y+1:y+5]),
            'w': self.parse_num(tokens[w+1:w+5]),
            'h': self.parse_num(tokens[h+1:h+5]),
            'theta': self.parse_angle(tokens[theta+1:theta+5])
        }
    
    def parse_num(self, tokens):
        return {'type': 'num', 'value': tokens[2]}
    
    def parse_angle(self, tokens):
        return {'type': 'angle', 'value': tokens[2]}
    
    def expand(self, node, cur_depth=0, max_depth=1e9):
        ps = {'cur_depth': cur_depth+1, 'max_depth': max_depth}
        if cur_depth >= max_depth: return node['type']
        if node['type'] == 's': return self.expand(node['value'], **ps)
        if node['type'] == 'op': return node['value']
        if node['type'] == 'binop': return f"( {self.expand(node['op'], **ps)} {self.expand(node['left'], **ps)} {self.expand(node['right'], **ps)} )"
        if node['type'] == 'circle': return f"( Circle {self.expand(node['r'], **ps)} {self.expand(node['x'], **ps)} {self.expand(node['y'], **ps)} )"
        if node['type']  == 'quad': return f"( Quad {self.expand(node['x'], **ps)} {self.expand(node['y'], **ps)} {self.expand(node['w'], **ps)} {self.expand(node['h'], **ps)} )"
        if node['type'] == 'num': return f"( Num {node['value']} )"
        if node['type'] == 'angle': return f"( Angle {node['value']} )"
        
    def render(self, program, size=(128, 128)):
        try: return self._render(self.parse(program), size=size)
        except: return None
    
    def _render(self, tree, size):
        image = Image.new('1', size=size, color='black') 
        return self._draw_tree(image, tree, size)
    
    def _draw_tree(self, image, node, size):
        if node['type']  == 's':
            return self._draw_tree(image, node['value'], size)
        
        if node['type'] == 'binop':
            if node['op']['value'] == 'Add':
                left = self._draw_tree(image.copy(), node['left'], size)
                right = self._draw_tree(image.copy(), node['right'], size)
                image = ImageChops.add(left, right)

            elif node['op']['value'] == 'Sub':
                left = self._draw_tree(Image.new('1', size, color='black'), node['left'], size)
                right = self._draw_tree(Image.new('1', size, color='black'), node['right'], size)
                image = ImageChops.add(image, ImageChops.difference(left, right))
                
        elif node['type'] == 'circle':
            draw = ImageDraw.Draw(image)
            self._draw_circle(draw, node, size)

        elif node['type'] == 'quad':
            draw = ImageDraw.Draw(image)
            self._draw_quad(draw, node, size)

        return image
    
    def _draw_circle(self, draw, node, size):
        r = self._val(node['r']) * min(size) / 32
        x = self._val(node['x']) * size[0] / 16
        y = self._val(node['y']) * size[1] / 16
        draw.ellipse([x-r, y-r, x+r, y+r], fill='white')
    
    def _draw_quad(self, draw, node, size):
        x = self._val(node['x']) * size[0] / 16
        y = self._val(node['y']) * size[1] / 16
        w = self._val(node['w']) * min(size) / 16
        h = self._val(node['h']) * min(size) / 16
        angle = self._val(node['theta']) * math.pi / 180
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        rotated_corners = [(x + (cx * math.cos(angle) - cy * math.sin(angle)), y + (cx * math.sin(angle) + cy * math.cos(angle))) for cx, cy in corners]
        draw.polygon(rotated_corners, fill='white')
        
    def _val(self, node):
        return int(node['value'])
    
    def tokenize(self, program):
        return [self.tok_to_id['BOS']] + list(map(lambda tok: self.tok_to_id[tok], program.split())) + [self.tok_to_id['EOS']]
    
    def detokenize(self, tok_ids, skip_special_tokens=True):
        if not skip_special_tokens: return ' '.join(self.id_to_tok[id] for id in tok_ids)

        try: start = tok_ids.index(self.tok_to_id['BOS'])
        except: start = 0
        
        try: end = tok_ids.index(self.tok_to_id['EOS'], start+1)
        except: end = len(tok_ids)
        
        tok_ids = tok_ids[start+1:end]
        return ' '.join(self.id_to_tok[id] for id in tok_ids)
    
    def detokenize_tensor(self, tok_ids, skip_special_tokens=True):
        return [self.detokenize(item, skip_special_tokens=skip_special_tokens) for item in tok_ids.tolist()] 

class ImageEncoder(nn.Module):
    
    def __init__(self, embed_dim, prefix_tokens):
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.embed_dim = embed_dim
        self.model = timm.create_model('nf_resnet26', in_chans=1)
        self.ffn = nn.Linear(1000, embed_dim*prefix_tokens)
        
    def forward(self, img):
        return self.ffn(self.model(img)).view(-1, self.prefix_tokens, self.embed_dim)
    
class CSGDataset(Dataset):

    def __init__(self, size=(128, 128), max_depth=4, **_):
        self.csg = CSG()
        self.size = size
        self.max_depth = max_depth
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return int(1e9)

    def __getitem__(self, _):
        program = self.csg.generate(max_depth=self.max_depth)
        img = self.csg.render(program, size=self.size)
        img_tensor = self.transform(img).float()
        input_ids = torch.tensor(self.csg.tokenize(program), dtype=torch.long)
        return {'img': img_tensor, 'input_ids': input_ids, 'program': program}

    def collate_fn(self, batch):
        N = max(len(item['input_ids']) for item in batch)
        input_ids = torch.full((len(batch), N), self.csg.tok_to_id['PAD'], dtype=torch.long)
        for i, item in enumerate(batch): input_ids[i, :len(item['input_ids'])] = item['input_ids']
        return {
            'input_ids': input_ids,
            'imgs': torch.stack([item['img'] for item in batch]),
            'attn_mask': input_ids.eq(self.csg.tok_to_id['PAD']),
            'programs': [item['program'] for item in batch]
        }
    
class CSGTreeDataset(CSGDataset):
    
    def to_traj(self, tree):
        INS, CPY, SUB = 0, 1, 2
        t2i = self.csg.tok_to_id
        
        traj = []
        i = 0
        while 1:
            cur = self.csg.expand(tree, max_depth=i)
            if traj and cur == traj[-1]: break
            traj.append(cur)
            i += 1
            
        traj = list(map(lambda x: x.split(), traj))
        traj_op_ids = torch.full((len(traj)-1, len(traj[-1])+2), -1)
        traj_tok_ids = torch.full((len(traj)-1, len(traj[-1])+2), -1)
        traj_idx_ids = torch.full((len(traj)-1, len(traj[-1])+2), -1)
        
        for i in range(len(traj)-1):
            cur = [(CPY, -1, 0)]
            a, b = traj[i], traj[i+1]
            j = k = 0
            
            for j in range(len(a)):
                if a[j] == b[k]:
                    cur.append((CPY, -1, j+1))
                    k += 1
                else:
                    if a[j] == 'binop': step = 5
                    elif a[j] == 'quad': step = 7
                    elif a[j] == 'circle': step = 6
                    elif a[j] in {'s', 'op'}: step = 1
                    elif a[j] in {'num', 'angle'}: step = 4
                    
                    for _ in range(step):
                        cur.append((INS, t2i[b[k]], -1))
                        k += 1
            
            cur.append((CPY, -1, len(a)+1))
            op_ids, tok_ids, idx_ids = map(lambda x: torch.tensor(x), zip(*cur))
            
            traj_op_ids[i, :len(op_ids)] = op_ids
            traj_tok_ids[i, :len(tok_ids)] = tok_ids
            traj_idx_ids[i, :len(idx_ids)] = idx_ids
                
        return traj_op_ids, traj_tok_ids, traj_idx_ids
    
    def collate_fn(self, batch):
        N = max(len(item['input_ids']) for item in batch)
        B = len(batch)
        
        input_ids = torch.full((B, N), self.csg.tok_to_id['PAD'], dtype=torch.long)
        for i, item in enumerate(batch): input_ids[i, :len(item['input_ids'])] = item['input_ids']
        
        ops = []
        toks = []
        idxs = []
        T = -1
        for item in batch:
            tree = self.csg.parse(item['program'])
            op, tok, idx = self.to_traj(tree)
            T = max(T, op.shape[0])
            ops.append(op)
            toks.append(tok)
            idxs.append(idx)
            
        bt_op_ids = torch.full((B, T, N), -1)
        bt_tok_ids = torch.full((B, T, N), -1)
        bt_idx_ids = torch.full((B, T, N), -1)
        
        for i in range(B):
            op, tok, idx = ops[i], toks[i], idxs[i]
            bt_op_ids[i, :op.shape[0], :op.shape[1]] = op
            bt_tok_ids[i, :tok.shape[0], :tok.shape[1]] = tok
            bt_idx_ids[i, :idx.shape[0], :idx.shape[1]] = idx
            
        return {
            'imgs': torch.stack([item['img'] for item in batch]),
            'input_ids': input_ids,
            'edit_ids': (bt_op_ids.long(), bt_tok_ids.long(), bt_idx_ids.long()),
            'programs': [item['program'] for item in batch]
        }

class Evolver(nn.Module):
    
    def __init__(
        self,
        d_model, dim_feedforward, nhead, dropout, layer_norm_eps,
        encoder_layers, decoder_layers, vocab_size, max_len,
        pad_token_id, bos_token_id, eos_token_id, root_id, csg, name,
        static=False
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
        self.root_id = root_id
        self.csg = csg
        self.name = name
        self.static = static
        
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
        self.img_encoder = ImageEncoder(d_model, 4)
        
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
    
    def compute_tgt(self, input_ids, edit_ids, mem):
        op_ids, tok_ids, idx_ids = edit_ids
        B, N = op_ids.shape
        
        ins_mask = op_ids.eq(0)
        cpy_mask = op_ids.eq(1)
        sub_mask = op_ids.eq(2)
        
        tgt = torch.zeros(B, N, self.d_model, device=input_ids.device)
        mem = self.positional_embedding(mem, d=-1)
        
        batch_indices = torch.arange(B, device=input_ids.device).unsqueeze(1)
        permuted_mem = mem[batch_indices, idx_ids]
        permuted_input_ids = input_ids[batch_indices, idx_ids]
            
        tgt[cpy_mask] = permuted_mem[cpy_mask]
        
        if ins_mask.any():
            tgt[ins_mask] = self.token_embedding(tok_ids[ins_mask]) * math.sqrt(self.d_model)
        
        if sub_mask.any():
            old_embeds = self.token_embedding(permuted_input_ids[sub_mask]) * math.sqrt(self.d_model)
            new_embeds = self.token_embedding(tok_ids[sub_mask]) * math.sqrt(self.d_model)
            tgt[sub_mask] = permuted_mem[sub_mask] - old_embeds + new_embeds
        
        tgt = self.positional_embedding(tgt, d=1)
        return tgt
        
    def apply_edits(self, input_ids, edit_ids):
        op_ids, tok_ids, idx_ids = edit_ids
        B = input_ids.shape[0]
        res = torch.zeros(B, op_ids.shape[1], dtype=torch.long, device=input_ids.device)

        ins_mask = op_ids.eq(INS_ID) | op_ids.eq(SUB_ID)
        res[ins_mask] = tok_ids[ins_mask]
        
        cpy_mask = op_ids.eq(CPY_ID)
        permuted_inputs = input_ids[torch.arange(B).view(-1, 1), idx_ids]
        
        res[cpy_mask] = permuted_inputs[cpy_mask]
        
        return res
    
    def forward(self, batch, src=None, mem=None, return_tgt=True):
        imgs, input_ids, edit_ids = batch
        B = input_ids.shape[0]
        N = edit_ids[0].shape[1]
        device = input_ids.device
        
        attn_mask = input_ids.eq(self.pad_token_id)
        concat_attn_mask = torch.cat([torch.full((B, 4), 0, dtype=torch.bool, device=device), attn_mask], dim=1)
        causal_mask = T.generate_square_subsequent_mask(N, dtype=torch.bool, device=device)
        
        img_embedding = self.img_encoder(imgs)
        src = torch.cat([img_embedding, self.embed(input_ids) if src is None else src], dim=1)
        
        mem = self.encoder(src, src_key_padding_mask=concat_attn_mask) if mem is None else mem
        tgt = self.embed(self.apply_edits(input_ids, edit_ids)) if self.static else self.compute_tgt(input_ids, edit_ids, mem[:, 4:])
        h, (*_, idx_weights) = self.decoder(tgt, mem, memory_key_padding_mask=concat_attn_mask, tgt_mask=causal_mask)
        
        op_probs = F.log_softmax(self.op_head(h), dim=-1)
        tok_probs = F.log_softmax(self.tok_head(h), dim=-1)
        
        # constrain to all of the token positions
        # idx_weights = self.pointer(tgt, mem[:, 4:], key_padding_mask=attn_mask)
        idx_weights = torch.log(torch.clamp(idx_weights, 1e-7, 1-1e-7))[:, :, 4:]
        idx_probs = F.log_softmax(idx_weights, dim=-1)
        
        if return_tgt:
            return (op_probs, tok_probs, idx_probs), tgt
        
        return op_probs, tok_probs, idx_probs
    
    def step(self, batch):
        imgs = batch['imgs']
        B, N = batch['input_ids'].shape
        edit_ids = batch['edit_ids']
        T = edit_ids[0].shape[1]
        
        input_ids = torch.full((B, N), self.pad_token_id, device=imgs.device)
        input_ids[:, 0] = self.bos_token_id
        input_ids[:, 1] = self.root_id
        input_ids[:, 2] = self.eos_token_id
        
        src = None
        op_loss = op_n = 0
        tok_loss = tok_n = 0
        idx_loss = idx_n = 0
        
        for i in range(T):
            cur_edit_ids = tuple(map(lambda x: x[:, i].to(imgs.device), edit_ids))
            (op_probs, tok_probs, idx_probs), src = self.forward((imgs, input_ids, cur_edit_ids), src=src, return_tgt=True)
            
            op_loss += F.nll_loss(op_probs[:, :-1].transpose(1, 2), edit_ids[0][:, i, 1:], ignore_index=-1, reduction='sum')
            op_n += torch.sum(edit_ids[0][:, i, 1:] != -1)
            
            tok_loss += F.nll_loss(tok_probs[:, :-1].transpose(1, 2), edit_ids[1][:, i, 1:], ignore_index=-1, reduction='sum')
            tok_n += torch.sum(edit_ids[1][:, i, 1:] != -1)
            
            idx_loss += F.nll_loss(idx_probs[:, :-1].transpose(1, 2), edit_ids[2][:, i, 1:], ignore_index=-1, reduction='sum')
            idx_n += torch.sum(edit_ids[2][:, i, 1:] != -1)
            
            input_ids = self.apply_edits(input_ids, cur_edit_ids)
            
        return op_loss / op_n, tok_loss / tok_n, idx_loss / idx_n
    
    def _generate(self, input_ids, img, src, max_steps, **_):
        device = img.device
        img = img.unsqueeze(0)
        
        op_ids = torch.tensor([[1]], dtype=torch.long, device=device)
        tok_ids = torch.tensor([[-1]], dtype=torch.long, device=device)
        idx_ids = torch.tensor([[0]], dtype=torch.long, device=device)
        
        for _ in range(max_steps):
            batch = (img, input_ids, (op_ids, tok_ids, idx_ids))
            op_probs, tok_probs, idx_probs = self.forward(batch, src=src, return_tgt=False)
            
            next_op = op_probs[:, -1]
            next_tok = tok_probs[:, -1]
            next_idx = idx_probs[:, -1]
            
            op_id = torch.multinomial(next_op.exp(), num_samples=1)
            
            if op_id == 0:
                tok_id = torch.multinomial(next_tok.exp(), num_samples=1)
                idx_id = torch.tensor([[-1]], device=device)
            elif op_id == 1:
                tok_id = torch.tensor([[-1]], device=device)
                idx_id = torch.multinomial(next_idx.exp(), num_samples=1)
            else:
                tok_id = torch.multinomial(next_tok.exp(), num_samples=1)
                idx_id = torch.multinomial(next_idx.exp(), num_samples=1)
                
            op_ids = torch.cat([op_ids, op_id], dim=1)
            tok_ids = torch.cat([tok_ids, tok_id], dim=1)
            idx_ids = torch.cat([idx_ids, idx_id], dim=1)
            
            if idx_id.item() >= 0 and input_ids[0, idx_id.item()] == self.eos_token_id: break
            if tok_id.item() >= 0 and tok_id.item() == self.eos_token_id: break
            
        output_ids = self.apply_edits(input_ids, (op_ids, tok_ids, idx_ids))
        _, tgt = self.forward((img, input_ids, (op_ids, tok_ids, idx_ids)), return_tgt=True)
        
        return output_ids, tgt
    
    def all_terminal(self, tok_ids):
        toks = self.csg.detokenize_tensor(tok_ids, skip_special_tokens=True)[0]
        return all([(tok not in {'s', 'binop', 'op', 'circle', 'quad', 'num', 'angle'}) for tok in toks.split()])
    
    def _generate_unbatched(self, img, max_depth, max_steps):
        src = None
        traj = [torch.tensor([[self.bos_token_id, self.root_id, self.eos_token_id]], dtype=torch.long, device=img.device)]
        
        for _ in range(max_depth):
            output_ids, src = self._generate(traj[-1], img, src, max_steps=max_steps)
            traj.append(output_ids)
            if self.all_terminal(output_ids): break
            
        return traj[-1]
   
    @torch.no_grad() 
    def generate(self, imgs, max_depth=1, max_steps=1, **_):
        output_ids = [self._generate_unbatched(img, max_depth, max_steps).squeeze() for img in imgs]
        N = max(len(ids) for ids in output_ids)
        res = torch.zeros((imgs.shape[0], N), dtype=torch.long, device=imgs.device)
        for i, ids in enumerate(output_ids): res[i, :len(ids)] = ids
        return res
    
class AutoregressiveEvolver(nn.Module):
    
    def __init__(
        self,
        d_model, dim_feedforward, nhead, dropout, layer_norm_eps,
        encoder_layers, decoder_layers, vocab_size, max_len,
        pad_token_id, bos_token_id, eos_token_id, root_id, csg, name,
        static=False
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
        self.root_id = root_id
        self.csg = csg
        self.name = name
        self.static = static
        
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
        self.img_encoder = ImageEncoder(d_model, 4)
        
        self.tok_head = nn.Linear(t_params['d_model'], vocab_size)
        
    def forward(self, batch):
        imgs = batch['imgs']
        input_ids = batch['input_ids']
        output_ids = batch['output_ids']
        
        B, N = input_ids.shape
        device = input_ids.device
        
        pad_mask = torch.cat([torch.full((B, 4), 0, dtype=torch.bool, device=device), input_ids.eq(self.pad_token_id, dtype=torch.bool)])
        causal_mask = T.generate_square_subsequent_mask(N, dtype=torch.bool, device=device)
        
        img_embedding = self.img_encoder(imgs)
        src = torch.cat([img_embedding, self.embed(input_ids)], dim=1)
        
        mem = self.encoder(src, src_key_padding_mask=pad_mask)
        tgt = self.embed(output_ids)
        # h, _ = self.decoder(tgt, mem, memory_key_padding_mask=concat_attn_mask)
        
        raise NotImplemented()
        
class DecoderOnlyTransformer(nn.Module):
    
    def __init__(
        self,
        d_model, dim_feedforward, nhead, dropout, layer_norm_eps,
        decoder_layers, vocab_size, max_len,
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
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len+1)
        
        decoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        self.decoder = TransformerEncoder(decoder_layer, decoder_layers)
        self.img_encoder = ImageEncoder(d_model, 4) # INS, CPY, SUB
        self.tok_head = nn.Linear(d_model, vocab_size)
   
    def embed(self, x):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x
    
    def forward(self, batch):
        imgs = batch['imgs']
        input_ids = batch['input_ids']
        pad_mask = batch['attn_mask']
        
        B, N = input_ids.shape
        device = input_ids.device

        pad_mask = torch.cat([torch.full((B, 4), 0, dtype=torch.bool, device=device), pad_mask], dim=1)
        causal_mask = T.generate_square_subsequent_mask(N+4, dtype=torch.bool, device=device)
        causal_mask[:, :4] = False
        
        img_embedding = self.img_encoder(imgs)
        tok_embedding = self.embed(input_ids)
        src = torch.cat([img_embedding, tok_embedding], dim=1)
        
        h = self.decoder(src, is_causal=True, src_mask=causal_mask, src_key_padding_mask=pad_mask)
        tok_logits = self.tok_head(h)
        tok_probs = F.log_softmax(tok_logits, dim=-1)
        
        # remove first 4 image tokens
        return tok_probs[:, 4:]
    
    def step(self, batch):
        tok_probs = self.forward(batch)
        loss = F.nll_loss(tok_probs[:, :-1].transpose(1, 2), batch['input_ids'][:, 1:], ignore_index=self.pad_token_id)
        return loss
    
    @torch.no_grad() 
    def generate(self, imgs, temperature=1.0, **_):
        B = imgs.shape[0]
        device = imgs.device
        
        input_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(self.max_len):
            logits = self.forward({'imgs': imgs, 'input_ids': input_ids, 'attn_mask': torch.zeros_like(input_ids, dtype=torch.bool, device=device)})
            next_tok_logits = logits[:, -1, :] / temperature
            next_tok = torch.multinomial(F.softmax(next_tok_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            finished |= (next_tok.squeeze(-1) == self.eos_token_id)
            if finished.all(): break
            
        return input_ids
    
def init_model(config, csg):
    if config['model_type'] == 'decoder_only':
        return DecoderOnlyTransformer(
            d_model=config['d_model'],
            dim_feedforward=config['dim_feedforward'],
            nhead=config['nhead'],
            dropout=config['dropout'],
            layer_norm_eps=config['layer_norm_eps'],
            decoder_layers=config['decoder_layers'],
            vocab_size=csg.vocab_size,
            max_len=config['max_len'],
            pad_token_id=csg.tok_to_id['PAD'],
            bos_token_id=csg.tok_to_id['BOS'],
            eos_token_id=csg.tok_to_id['EOS'],
            name=config['name']
        ).to(config['device'])
    
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
    save_path = os.path.join(config['checkpoint_dir'], f"{model.name}_{step}.pt")
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
    
def train_step(model, batch, device, step=None):
    if isinstance(model, DecoderOnlyTransformer):
        batch = {'imgs': batch['imgs'].to(device), 'input_ids': batch['input_ids'].to(device), 'attn_mask': batch['attn_mask'].to(device), 'programs': batch['programs']}
        return model.step(batch)
    
    elif isinstance(model, Evolver):
        # defer moving edit_ids to gpu to save space
        batch = {'imgs': batch['imgs'].to(device), 'input_ids': batch['input_ids'], 'edit_ids': batch['edit_ids']}
        op_loss, tok_loss, idx_loss = model.step(batch)
        if step is not None: log_to_wandb({'train/op_loss': op_loss, 'train/tok_loss': tok_loss, 'train/idx_loss': idx_loss}, step=step)
        return  op_loss + tok_loss + idx_loss
        
    else:
        raise Exception(f'unsupported model type {type(model)}')

def calculate_iou(targets, renders):
    targets = targets.bool()
    renders = renders.bool()
    i = torch.logical_and(targets, renders).sum(dim=(1, 2, 3)).float()
    u = torch.logical_or(targets, renders).sum(dim=(1, 2, 3)).float()
    return torch.where(u > 0, i / u, torch.ones_like(u))

@torch.no_grad()
def evaluate(model, eval_loader, device, num_eval_steps, csg):
    model.eval()
    tt = transforms.ToTensor()
    
    tot_loss = 0
    tot_samples = 0
    err_samples = 0
    tot_iou = 0
    
    for i, batch in enumerate(eval_loader):
        if i >= num_eval_steps: break
        loss = train_step(model, batch, device)
        tot_loss += loss.item()
        
        generated = model.generate(batch['imgs'].to(device))
        programs = csg.detokenize_tensor(generated)
        renders = [csg.render(prog) for prog in programs]
        err = torch.tensor([r is None for r in renders], dtype=torch.bool)

        targets = batch['imgs'][~err].to(device)
        renders = [tt(r) for r in renders if r is not None]

        tot_samples += len(generated)
        err_samples += torch.sum(err)

        if len(renders) > 0:
            renders = torch.stack(renders).to(device)
            tot_iou += torch.sum(calculate_iou(targets, renders))
    
    return (
        tot_loss / num_eval_steps,
        err_samples / tot_samples,
        (tot_iou / (tot_samples - err_samples)) if (err_samples < tot_samples) else 0
    )

def train(config):
    device = config['device']
    csg = CSG()
    
    if config['model_type'] == 'decoder_only': dataset = CSGDataset(size=config['image_size'], max_depth=config['max_depth'])
    elif config['model_type'] == 'evolver': dataset = CSGTreeDataset(size=config['image_size'], max_depth=config['max_depth'])
    else: raise Exception(f"invalid model_type {config['model_type']}")
    
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=dataset.collate_fn, num_workers=config['num_workers'])
    eval_loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=dataset.collate_fn, num_workers=config['num_workers'])

    model = init_model(config, csg)
    optim = AdamW(model.parameters(), lr=config['lr'])

    start_step = load_checkpoint(model, optim, config)
    
    logger.info('eval sanity check')
    evaluate(model, eval_loader, device, 1, csg)
    logger.info('passed!')

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
            eval_loss, err_rate, avg_iou = evaluate(model, eval_loader, device, config['num_eval_steps'], csg)
            log_to_wandb({'eval/loss': eval_loss, 'eval/err_rate': err_rate, 'eval/iou': avg_iou}, step=step)
            model.train()

        if step % config['save_every'] == 0:
            save_checkpoint(model, optim, step, config)

    eval_loss, err_rate, avg_iou = evaluate(model, eval_loader, device, config['num_eval_steps'], csg)
    log_to_wandb({'eval/loss': eval_loss, 'eval/err_rate': err_rate, 'eval/iou': avg_iou}, step=step)
    save_checkpoint(model, optim, config['train_steps'], config)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--from-checkpoint', default=None)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    config['device'] = args.device
    config['local'] = args.local
    config['from_checkpoint'] = args.from_checkpoint
    config['name'] = f"csg_{config['model_type']}_{config['d_model']}d_{config.get('encoder_layers', 0)}enc_{config['decoder_layers']}dec-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not config['local']: wandb.init(project='csg-evolver', name=config['name'], config=config, resume='allow')
    train(config)

if __name__ == '__main__':
    main()
