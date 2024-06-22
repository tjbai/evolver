import json
import stanza
import random
import argparse
from copy import deepcopy
from functools import lru_cache
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from ordered_set import OrderedSet

parser = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

def flip(w):
    return random.choices([True, False], weights=[w, 1-w])[0]

@lru_cache
def parse(observed):
    parsed = parser(observed)
    N = len([word for sentence in parsed.sentences for word in sentence.words])
    
    parent = {}
    to_text = {}
    children = defaultdict(set)
    leaves = set(i for i in range(N))
    sent = OrderedSet(i for i in range(N))
    
    ct = 0
    for sentence in parsed.sentences:
        for _i, word in enumerate(sentence.words):
            i = ct + _i
            par = word.head + ct - 1
            parent[i] = par
            to_text[i] = word.text
            children[par].add(i)
            if par in leaves: leaves.remove(par)
        ct += len(sentence.words)
        
    return parent, children, leaves, sent, to_text

def noise_dep(observed):
    parent, children, leaves, sent, to_text = parse(observed)
    
    # make copies so we can cache
    parent = deepcopy(parent)
    children = deepcopy(children)
    leaves = deepcopy(leaves)
    sent = deepcopy(sent)
    to_text = deepcopy(to_text)
    
    log_prob = 0
    traj = [' '.join(to_text[s] for s in sent)]
    
    while sent:
        cand = list(leaves)
        for leaf in cand:
            if flip(0.15):
                log_prob += np.log(0.15)
                continue
            log_prob += np.log(0.85)
            leaves.remove(leaf)
            sent.remove(leaf)
            if leaf not in parent: continue
            children[parent[leaf]].remove(leaf)
            if not children[parent[leaf]]: leaves.add(parent[leaf])
            
        seq = ' '.join(to_text[s] for s in sent)
        if seq and seq != traj[-1]: traj.append(seq)

    return traj[::-1], log_prob

def noise_gpt(observed):
    pass

def load(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def dump(traj_list, path):
    with open(path, 'w') as f:
        for traj in tqdm(traj_list):
            json.dump(traj, f)
            f.write('\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    return parser.parse_args()

def main():
    args = parse_args() 
    dump((noise_forward(obs)[0] for obs in load(args.input)), args.output)
    
if __name__ == '__main__':
    main()
