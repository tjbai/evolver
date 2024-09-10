import argparse
import json
import torch
import os

from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from evo import Transformer, PointerStyleEvolver
from data import SequenceDataset
from run import sample_trajectory, sample_ar
from utils import BT
from data import elaborate

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_model(config, checkpoint_path):
    name = os.path.basename(checkpoint_path)
    model = PointerStyleEvolver if name.startswith('ps') else Transformer
    
    model_params = {
        'd_model': config['d_model'],
        'nhead': config['nhead'],
        'dim_feedforward': config['dim_feedforward'],
        'dropout': config['dropout'],
        'encoder_layers': config['encoder_layers'],
        'decoder_layers': config['decoder_layers'],
        'max_len': config['max_len'],
        'device': config['device'],
        'pointer_attn': config.get('pointer_attn', False)
    }
    
    model = model(**model_params).to(config['device'])
    model.load_state_dict(torch.load(checkpoint_path, map_location=config['device'])['model'])
    model.eval()
    
    return model

def batch_decode(input_ids):
    # batch decode {T|B}xN tensor
    input_ids = input_ids.cpu().tolist()
    return BT.batch_decode(input_ids, skip_special_tokens=True)
    
def run(model, loader, output_file, limit):
    with open(output_file, 'w') as f:
        for i, (input_ids, _) in tqdm(enumerate(loader)):
            if limit is not None and i >= limit: break
            if isinstance(model, PointerStyleEvolver):
                output_ids, _edits = sample_trajectory(model, input_ids, T=6, pf_params={}, verbose=True)
                edits = list(zip(*elaborate(_edits)))
                for i in range(output_ids.shape[1]):
                    traj = batch_decode(output_ids[:, i])
                    json.dump(traj, f)
                    f.write('\n')
                    json.dump(edits[i], f)
                    f.write('\n\n')
            else:
                output_ids = sample_ar(model, input_ids, verbose=True)
                batch = batch_decode(output_ids)
                for str in batch:
                    f.write(str)
                    f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--input-file')
    parser.add_argument('--output-file')
    parser.add_argument('--shuffle')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = load_config(args.config)
    config['device'] = args.device
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    model = load_model(config, args.checkpoint)
    dataset = SequenceDataset.from_trajectories(args.input_file, denoising=False, max_len=config['max_len'], tokenizer=BT)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=args.shuffle)
    
    run(model, loader, args.output_file, args.limit)

if __name__ == '__main__':
    main()
