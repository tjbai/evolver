import json
import random
import argparse
from copy import deepcopy
from functools import lru_cache
from collections import defaultdict

import conllu
import numpy as np
from tqdm import tqdm
from ordered_set import OrderedSet

def flip(w):
    return random.choices([True, False], weights=[w, 1-w])[0]

def parse(observed):
    N = len(observed)
    
    parent = {}
    to_text = {}
    children = defaultdict(set)
    leaves = set(i for i in range(N))
    sent = OrderedSet(i for i in range(N))
    
    for i, token in enumerate(observed):
        par = token['head']
        parent[i] = par
        to_text[i] = token['form']
        children[par].add(i)
        if par in leaves: leaves.remove(par)
        
    return parent, children, leaves, sent, to_text

def noise(observed, w):
    parent, children, leaves, sent, to_text = parse(observed)
    
    log_prob = 0
    traj = [' '.join(to_text[s] for s in sent)]
    
    while leaves:
        cand = list(leaves)
        for node in cand:
            if flip(w):
                log_prob += np.log(w)
                continue
            log_prob += np.log(1-w)
            leaves.remove(node)
            if node in sent: sent.remove(node)
            if node not in parent: continue
            children[parent[node]].remove(node)
            if not children[parent[node]]: leaves.add(parent[node])
            
        seq = ' '.join(to_text[s] for s in sent)
        if seq and seq != traj[-1]: traj.append(seq)

    return traj[::-1], log_prob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--redundant', default=1)
    parser.add_argument('--weight', default=0.2, type=float)
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.input, 'r') as f:
        sentences = conllu.parse(f.read())
       
    traj_list = []
    length = []
    for sent in sentences:
        for _ in range(args.redundant):
            traj_list.append(noise(sent, args.weight)[0])
            length.append(len(traj_list[-1]))
            
    print(f'generated {len(traj_list)} noising trajectories')
    print(f'avg length {sum(length) / len(length)}')
    
    with open(args.output, 'w') as f:
        for traj in traj_list:
            json.dump(traj, f)
            f.write('\n')
    
if __name__ == '__main__':
    main()
