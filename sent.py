import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from transformers import BertTokenizer

from const import *
from data import get_input_ids
from evo import CosineEmbedding

class StackedClassifier(nn.Module):
    
    def __init__(
        self,
        d_model=512, nhead=8,
        dim_feedforward=2048, encoder_layers=6,
        vocab_size=VOCAB_SIZE, max_len=10,
        stacked=1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.stacked = stacked
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = CosineEmbedding(d_model=d_model, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.0, dim_feedforward=dim_feedforward, batch_first=True) 
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        for param in self.encoder.parameters(): param.requires_grad = False
        
        self.pool = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))
        self.ffn = nn.Linear(d_model, 1)
        
    def forward(self, input_ids):
        pad_mask = input_ids.eq(PAD_TOKEN_ID)
        x = self.embedding(input_ids) * torch.sqrt(self.d_model)
        x = self.positional_encoding(x, dir=1)
       
        for _ in range(self.stacked): x = self.encoder(x, src_key_padding_mask=pad_mask)
        
        scores = self.pool(x).squeeze()
        scores = scores.masked_fill(pad_mask, -1e9)
        scores = F.softmax(scores, dim=-1)
        pooled = (x * scores.unsqueeze(-1)).sum(dim=1)

        return self.ffn(pooled).squeeze()
    
    def loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    
    def accuracy(self, logits, labels):
        predicted = (torch.sigmoid(logits) > 0.5).long()
        correct = torch.sum(predicted == labels)
        return correct / predicted.shape[-1]
    
class SentimentDataset(Dataset):
    
    @staticmethod
    def from_disk(cls, path, **kwargs):
        seqs = []
        labels = []
        
        with open(path, 'r') as f:
            for line in f.readlines():
                seq, label = json.loads(line)
                seqs.append(seq)
                labels.append(label)
                
        return cls(seqs, labels, **kwargs)
            
    def __init__(self, seqs, labels, max_len=512):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
        self.input_ids = get_input_ids(seqs, max_len, tokenizer)
        self.labels = labels
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
   
def train(model, train_loader):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == '__main__':
    main()