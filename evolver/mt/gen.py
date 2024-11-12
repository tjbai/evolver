import os
import json
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mt import Transformer, MarianTokenizer, load_config, init_model, MTDataset, SpacyTokenizer, MTEditDataset

def generate_completions(model, tokenizer, src_texts, num_completions, max_length, device):
    model.to(device)
    completions = []
    
    for src_text in tqdm(src_texts, desc="Generating completions"):
        src_ids = torch.tensor([tokenizer.encode(src_text)]).to(device)
        
        for _ in range(num_completions):
            generated_ids = model.generate(src_ids, max_length=max_length)
            completion = tokenizer.decode(generated_ids[0])
            completions.append((src_text, completion))
    
    return completions

def dump(tokenizer, ref, traj, output_file):
    with open(output_file, 'w') as f:
        f.write(f"ref: {tokenizer.decode(ref)}\n")
        for i, hyp in enumerate(traj):
            f.write(f"step {i+1}: {tokenizer.decode(hyp)}\n")
            f.write("\n" + "="*50 + "\n\n")
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    parser.add_argument('--num_completions')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    config['device'] = args.device
    
    tokenizer = MarianTokenizer() if config['model_type'] == 'decoder_only' else SpacyTokenizer()
    dataset = MTDataset if config['model_type'] == 'decoder_only' else MTEditDataset
    dataset = dataset(split='validation', max_len=config['max_len'], buffer_size=config['buffer_size'], tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn) 
    
    model = init_model(config, tokenizer)
    
    for batch in loader[:10]:
        traj = model.generate({
            'src_ids': batch['src_ids'].to(config['device']),
            'input_ids': batch['input_ids'].to(config['device']),
            'tgt_ids': batch['tgt_ids'].to(config['device']),
            'edit_ids': tuple(map(lambda x: x.to(config['device']), batch['edit_ids']))})
        
        dump()

    print(f"Completions have been saved to {args.output}")

if __name__ == "__main__":
    mTJain()