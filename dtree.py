import json
import random
import argparse
from collections import defaultdict

import conllu
import numpy as np
from tqdm import tqdm
from ordered_set import OrderedSet
from nltk.tokenize.treebank import TreebankWordDetokenizer

detok = TreebankWordDetokenizer() 

def flip(w):
    return random.choices([True, False], weights=[w, 1-w])[0]

def parse(observed):
    N = len(observed)
    
    parent = {}
    to_form = {}
    children = defaultdict(set)
    leaves = set(i for i in range(N))
    sent = OrderedSet(i for i in range(N))
    
    for i, token in enumerate(observed):
        if token['head'] is None:
            sent.remove(i)
            continue
        
        par = token['head'] - 1
        parent[i] = par
        to_form[i] = token['form']
        children[par].add(i)
        if par in leaves: leaves.remove(par)
        
    return parent, children, leaves, sent, to_form

def noise(observed, w=0.1):
    parent, children, leaves, sent, to_form = parse(observed)
    traj = [[to_form[s] for s in sent]]
    log_prob = 0
    
    while leaves:
        cand = list(leaves)
        updated = False
        
        for node in cand:
            if flip(w):
                log_prob += np.log(w)
                continue
            
            log_prob += np.log(1-w)
            updated = True
           
            # remove this node 
            leaves.remove(node) 
            if node in sent: sent.remove(node)
           
            # update parent
            if node not in parent: continue
            children[parent[node]].remove(node)
            if not children[parent[node]]: leaves.add(parent[node])
           
        if updated:
            seq = [to_form[s] for s in sent]
            if ' '.join(seq) == ' '.join(traj[-1]): continue
            traj.append(seq)

    return list(map(detok.detokenize, traj[::-1])), log_prob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--redundant', default=1, type=int)
    parser.add_argument('--weight', default=0.2, type=float)
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.input, 'r') as f:
        sentences = conllu.parse(f.read())
      
    traj_list = []
    log_probs = []
    length = []
    
    for sent in tqdm(sentences):
        for _ in range(args.redundant):
            traj, log_prob = noise(sent, args.weight)
            traj_list.append(traj)
            log_probs.append(log_prob)
            length.append(len(traj_list[-1]))
            
    print(f'generated {len(traj_list)} noising trajectories')
    print(f'avg length {sum(length) / len(length)}')
    
    with open(args.output, 'w') as f:
        for traj, log_prob in zip(traj_list, log_probs):
            json.dump([traj, log_prob], f)
            f.write('\n')
    
if __name__ == '__main__':
    main()
