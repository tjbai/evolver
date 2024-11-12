import random
import torch
import spacy
from transformers import MarianTokenizer as Marian
from datasets import load_dataset
from collections import defaultdict
from torch.utils.data import Dataset
    
def pad(seqs, padding_value, max_len=int(1e10)):
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=padding_value)[:, :max_len]

class Parser:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.marian = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
    
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
        if spacy_spans is None: spacy_spans = self.get_spans(text, spacy_tokens)
        if marian_spans is None: marian_spans = self.get_spans(text, marian_tokens)
        
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
        return [self.marian.decode(id) for id in self.marian(text)['input_ids'][1:-1]]
    
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
            
            '''
            we use a weak heuristic to determine the lineage if the parent is broken up into several tokens.

            we also encounter a funny edge case here with percentage signs, e.g. 11%
            spacy will tokenize this into 11%, where 11 <- %
            marian will tokenize this into 1 and 1%, essentially blending generations

            jank solution: if the parent is empty then just go one level deeper
            '''
            spacy_par = doc[alignment[i]].head
            while len(reverse[spacy_par.i]) == 0:
                spacy_par = spacy_par.head
            par = reverse[spacy_par.i][0]
            seq[i]['par'] = par
            
        return seq

class MarianTokenizer(Marian):
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = super().from_pretrained(*args, **kwargs)
        tokenizer.add_special_tokens({'bos_token': '<s>'})
        tokenizer.add_bos_token = True
        return tokenizer
    
    def encode(self, text):
        return super().encode(self.bos_token+text)
    
    def __call__(self, text, *args, **kwargs):
        return super().__call__(self.bos_token+text, *args, **kwargs)
    
class WMT(Dataset):

    def __init__(self, split='train', max_len=256, truncate=None):
        self.max_len = max_len
        self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        self.dataset = load_dataset('wmt14', 'de-en', split=split)
        if truncate is not None: self.dataset = self.dataset.select(range(truncate))

        # just for utility
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
    
    def get_id(self, tok):
        return self.tokenizer.get_vocab().get(tok)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]['translation']
        src_ids = self.tokenizer.encode(item['de'])[:self.max_len]
        tgt_ids = self.tokenizer.encode(item['en'])[:self.max_len]
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}
        
    def collate_fn(self, batch):
        src_ids = [item['src_ids'] for item in batch]
        tgt_ids = [item['tgt_ids'] for item in batch]
        src_ids = pad([torch.tensor(ids) for ids in src_ids], self.pad_token_id)
        tgt_ids = pad([torch.tensor(ids) for ids in tgt_ids], self.pad_token_id)
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}
    
class EvolverWMT(WMT):

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
        INS, CPY, SUB = 0, 1, 2
        
        # NOTE -- this the reduced version for scaling efficiency
        input_traj = self._get_short_input_traj(doc)

        # prefill empty string x0 
        output_traj = []
        traj_ids = []
        output_traj.append([(INS, self.bos_token_id, -1), (INS, self.eos_token_id, -1)])
        traj_ids.append([self.bos_token_id, self.eos_token_id])
       
        # prefill root sequence 
        par_idx = {} 
        output_traj.append([(INS, self.bos_token_id, -1)])
        traj_ids.append([(self.bos_token_id)])
        k = 1
        for tok in doc:
            if not tok['is_head']: continue
            output_traj[-1].append((INS, self.get_id(tok['text']), -1))
            par_idx[tok['i']] = k
            k += 1
        output_traj[-1].append((INS, self.eos_token_id, -1))
        traj_ids[-1].append(self.eos_token_id)

        # fill in subsequent steps
        for seq in input_traj[1:]:
            k = 1
            new_par_idx = {}
            cur_edits = [(INS, self.bos_token_id, -1)]
            cur_ids = [self.bos_token_id]
            
            for t in seq:
                if t == '_':
                    continue
                elif t[1] in par_idx:
                    assert par_idx[t[1]] < len(output_traj[-1]), f'{par_idx[t[1]]} is too large for prev seq size of {len(output_traj[-1])}'
                    cur_edits.append((CPY, -1, par_idx[t[1]]))
                else:
                    assert par_idx[t[2]] < len(output_traj[-1]), f'{par_idx[t[2]]} is too large for prev seq size of {len(output_traj[-1])}'
                    cur_edits.append((SUB, self.get_id(t[0]), par_idx[t[2]]))
                cur_ids.append(self.get_id(t[0]))
                new_par_idx[t[1]] = k
                k += 1
            
            par_idx = new_par_idx 
            cur_edits.append((INS, self.eos_token_id, -1))
            cur_ids.append(self.eos_token_id)
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
        src_ids = pad([torch.tensor(item['src_ids']) for item in batch], self.pad_token_id, self.max_len)
        input_ids = pad([torch.tensor(item['input_ids']) for item in batch], self.pad_token_id, self.max_len)
        tgt_ids = pad([torch.tensor(item['tgt_ids']) for item in batch], self.pad_token_id, self.max_len)
        op_ids = pad([torch.tensor(item['edit_ids'][0]) for item in batch], -1, self.max_len)
        tok_ids = pad([torch.tensor(item['edit_ids'][1]) for item in batch], -1, self.max_len)
        idx_ids = pad([torch.tensor(item['edit_ids'][2]) for item in batch], -1, self.max_len)
        return {'src_ids': src_ids, 'input_ids': input_ids, 'tgt_ids': tgt_ids, 'edit_ids': (op_ids, tok_ids, idx_ids)}
