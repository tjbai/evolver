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
from trans import TransformerEncoder, TransformerEncoderLayer, CausalTransformerDecoder, CausalTransformerDecoderLayer, MultiheadPointer

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
        if tokens[1] == 'Circle': return self.parse_circle(tokens)
        if tokens[1] == 'Quad': return self.parse_quad(tokens)
        return self.parse_binop(tokens)

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

    def parse_binop(self, tokens):
        i = self.find_split(tokens[2:-1])
        return {
            'type': 'binop',
            'op': tokens[1],
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
        if max_depth == 0: return 's'
        if cur_depth >= max_depth: return node['type']
        if node['type'] == 'binop': return f"( {node['op']} {self.expand(node['left'], cur_depth+1, max_depth)} {self.expand(node['right'], cur_depth+1, max_depth)} )"
        if node['type'] == 'circle': return f"( Circle {self.expand(node['r'], cur_depth+1, max_depth)} {self.expand(node['x'], cur_depth+1, max_depth)} {self.expand(node['y'], cur_depth+1, max_depth)} )"
        if node['type']  == 'quad': return f"( Quad {self.expand(node['x'], cur_depth+1, max_depth)} {self.expand(node['y'], cur_depth+1, max_depth)} {self.expand(node['w'], cur_depth+1, max_depth)} {self.expand(node['h'], cur_depth+1, max_depth)} )"
        if node['type'] == 'num': return f"( Num {node['value']} )"
        if node['type'] == 'angle': return f"( Angle {node['value']} )"
        
    def render(self, program, size=(128, 128)):
        return self._render(self.parse(program), size=size)
    
    def _render(self, tree, size):
        image = Image.new('1', size=size, color='black') 
        return self._draw_tree(image, tree, size)
    
    def _draw_tree(self, image, node, size):
        if node['type'] == 'binop':
            if node['op'] == 'Add':
                left = self._draw_tree(image.copy(), node['left'], size)
                right = self._draw_tree(image.copy(), node['right'], size)
                image = ImageChops.add(left, right)
            elif node['op'] == 'Sub':
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
    
    def detokenize(self, tok_ids):
        return ' '.join(self.id_to_tok[id] for id in tok_ids) 

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
    def __init__(self, size=(128, 128), max_depth=4, num_samples=1000):
        self.csg = CSG()
        self.size = size
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        program = self.csg.generate(max_depth=self.max_depth)
        image = self.csg.render(program, size=self.size)
        image_tensor = self.transform(image).float()
        input_ids = torch.tensor(self.csg.tokenize(program), dtype=torch.long)
        return {'image': image_tensor, 'input_ids': input_ids}

    def collate_fn(self, batch):
        images = torch.stack([item['image'] for item in batch])
        N = max(len(item['input_ids']) for item in batch)
        input_ids = torch.full((len(batch), N), self.csg.tok_to_id['PAD'], dtype=torch.long)
        for i, item in enumerate(batch): input_ids[i, :len(item['input_ids'])] = item['input_ids']
        return {'images': images, 'input_ids': input_ids, 'attn_mask': input_ids.eq(self.csg.tok_to_id['PAD'])}

class SyntaxDecoder(nn.Module):
    pass

class Evolver(nn.Module):
    
    def __init__(
        self,
        d_model, dim_feedforward, nhead, dropout, layer_norm_eps,
        encoder_layers, decoder_layers,
        vocab_size, max_len
    ):
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        t_params = {
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps
        }
       
        encoder_layer = TransformerEncoderLayer(**t_params)
        decoder_layer = CausalTransformerDecoderLayer(**t_params)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = CausalTransformerDecoder(decoder_layer, decoder_layers)
        
        self.img_enocder = ImageEncoder(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len+1)
        
        self.op_head = nn.Linear(t_params['d_model'], 5)
        self.tok_head = nn.Linear(t_params['d_model'], vocab_size)
        self.pointer = MultiheadPointer(**t_params)
        
    def get_src(self, x):
        pad_mask = x.eq(PAD_TOKEN_ID)
        x = self.embedding(x) * math.sqrt(self.t_params['d_model'])
        x = self.positional_embedding(x, d=1)
        return x, pad_mask
    
    def forward(
        self,
        input_ids, input_img, edit_tgts,
        src=None, memory=None, cache=None
    ):
        B, N = input_ids.shape
        assert N <= self.max_len
        assert B == input_img.shape[0]
        
        _src, pad_mask = self.get_src(input_ids)
        tok_src = src if src is not None else _src
        img_src = self.img_encoder(input_img)
        src = torch.stack([img_src, tok_src], dim=1)
        
        # TODO -- finish
        
class DecoderOnlyTransformer(nn.Module):
    
    def __init__(
        self,
        d_model, dim_feedforward, nhead, dropout, layer_norm_eps,
        decoder_layers, vocab_size, max_len, pad_token_id, name
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
        self.name = name
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalEmbedding(d_model=d_model, max_len=max_len+1)
        
        decoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        self.decoder = TransformerEncoder(decoder_layer, decoder_layers)
        self.img_encoder = ImageEncoder(d_model, 4)
        self.tok_head = nn.Linear(d_model, vocab_size)
    
    def embed(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x, d=1)
        return x
    
    def forward(self, batch):
        imgs, input_ids, pad_mask = batch.values()
        B = input_ids.shape[0]
        pad_mask = torch.cat([torch.full((B, 4), 0, dtype=torch.bool), pad_mask], dim=1)
        causal_mask = T.generate_square_subsequent_mask(input_ids.shape[1]+4, input_ids.device, dtype=torch.bool)
        causal_mask[:, :4] = False
        
        tok_embedding = self.embed(input_ids)
        img_embedding = self.img_encoder(imgs)
        src = torch.cat([img_embedding, tok_embedding], dim=1)
        
        h = self.decoder(src, is_causal=True, src_mask=causal_mask, src_key_padding_mask=pad_mask)
        tok_logits = self.tok_head(h)
        tok_probs = F.log_softmax(tok_logits, dim=-1)
        
        # remove first 4 image tokens 
        return tok_probs[:, 4:]
    
def init_model(config, csg):
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
        name=config['name']
    ).to(config['device'])
    
def load_checkpoint(model, optimizer, config):
    if config['from_checkpoint']:
        checkpoint = torch.load(config['from_checkpoint'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step'] + 1
        logger.info(f"Resuming from step {start_step}")
        return start_step
    return 0

def save_checkpoint(model, optimizer, step, config):
    save_path = os.path.join(config['checkpoint_dir'], f"{model.name}_{step}.pt")
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
    
@timing
def train_step(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    tok_probs = model(batch)
    loss = F.nll_loss(tok_probs[:, :-1].transpose(1, 2), batch['input_ids'][:, 1:], ignore_index=model.pad_token_id)
    return loss

@timing
@torch.no_grad()
def evaluate(model, eval_loader, device, num_eval_steps):
    model.eval()
    tot = 0
    for i, batch in enumerate(eval_loader):
        if i >= num_eval_steps: break
        loss = train_step(model, batch, device)
        tot += loss.item()
    return tot / num_eval_steps

def train(config):
    device = config['device']
    csg = CSG()
    
    dataset = CSGDataset(size=config['image_size'], max_depth=config['max_depth'], num_samples=config['num_samples'])
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=dataset.collate_fn, num_workers=config['num_workers'])
    eval_loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=dataset.collate_fn, num_workers=config['num_workers'])

    model = init_model(config, csg)
    optim = AdamW(model.parameters(), lr=config['lr'])

    start_step = load_checkpoint(model, optim, config)

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
            eval_loss = evaluate(model, eval_loader, device, config['num_eval_steps'])
            log_to_wandb({'eval/loss': eval_loss}, step=step)
            model.train()

        if step % config['save_every'] == 0:
            save_checkpoint(model, optim, step, config)

    eval_loss = evaluate(model, eval_loader, device, config['num_eval_steps'])
    log_to_wandb({'eval/loss': eval_loss}, step=config['train_steps'])
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
    config['name'] = f"csg_decoder_only_{config['d_model']}d_{config['decoder_layers']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not config['local']: wandb.init(project='csg-evolver', name=config['name'], config=config, resume='allow')
    train(config)

if __name__ == '__main__':
    main()
