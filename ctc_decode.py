#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:33:04 2020

@author: mtdp
"""

import numpy as np
# example
B = 1
L = 12
V = 5  # 0 for blank, 1 for pandding
target = np.array([[3,3,4,1]])
target_length = np.array([3])
rec_logits = np.random.rand(B, L, V)     # bs 1, text_len 6, voc_num 5
exp = np.exp(rec_logits)
rec_probs = exp / np.sum(exp, axis=2, keepdims=True)


# remove blank and repeat
def remove_blank(raw_str, blank=0):
    res_str = []
    for b in range(len(raw_str)):
        res_b = []
        prev = -1
        for ch in raw_str[b]:
            if ch == prev or ch == blank:
                prev = ch
                continue
            res_b.append(ch)
            prev = ch
        res_str.append(res_b) 
    return res_str


# greedy search
def greedy_decode(rec_probs, blank=0):
    # raw str
    raw_str = np.argmax(rec_probs, axis=2).tolist()
    
    # res str
    res_str = remove_blank(raw_str)

    return raw_str, res_str


# beam search
def beam_decode(rec_probs, beam_size=5):
    B, L, V = rec_probs.shape
    # log for add better than multiple directly
    rec_probs = np.log(rec_probs)
    
    raw_str = []
    res_str = []
    scores = []
    for b in range(B):
        rec_prob_b = rec_probs[b]
        beam = [([], 0)]
        for t in range(L):
            new_beam = []
            for prefix, score in beam:
                for i in range(V):
                    new_prefix = prefix + [i]
                    new_score = score + rec_prob_b[t, i]
                    new_beam.append((new_prefix, new_score))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]
            
        raw_str.append([])
        res_str.append([])
        scores.append([])
        for j in range(len(beam)):
            raw_str[-1].append(beam[j][0])
            res_str[-1].append(remove_blank([beam[j][0]])[0])
            scores[-1].append(beam[j][1])
            
    return raw_str, res_str, scores
 
    


    
raw_str, res_str = greedy_decode(rec_probs)
print('greedy_decode')
print(raw_str)      # shape B, 1
print(res_str)
print()
print('beam_decode')
raw_str, res_str, scores = beam_decode(rec_probs)
print(raw_str)      # shape B, beam_size, 1
print(res_str)
print(scores)
    
    
                
          
# prefix beam search
# confusing
import math
import collections
NEG_INF = -float("inf")

def make_new_beam():
    fn = lambda : (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))    # why ???
    return a_max + lsp

def prefix_beam_decode(rec_probs, beam_size=5, blank=0):
    """
    Performs inference for the given output probabilities.
    Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
            time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.
        blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    B, T, S = rec_probs.shape
    res_str = []
    scores = []
    for b in range(B):
        probs = np.log(rec_probs[b])
    
        # Elements in the beam are (prefix, (p_blank, p_no_blank))
        # Initialize the beam with the empty sequence, a probability of
        # 1 for ending in blank and zero for ending in non-blank
        # (in log space).
        beam = [(tuple(), (0.0, NEG_INF))]
    
        for t in range(T): # Loop over time
            # A default dictionary to store the next step candidates.
            next_beam = make_new_beam()
            
            for s in range(S): # Loop over vocab
                p = probs[t, s]
    
                # The variables p_b and p_nb are respectively the
                # probabilities for the prefix given that it ends in a
                # blank and does not end in a blank at this time step.
                for prefix, (p_b, p_nb) in beam: # Loop over beam
    
                    # If we propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
                    if s == blank:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        continue
    
                    # Extend the prefix by the new character s and add it to
                    # the beam. Only the probability of not ending in blank
                    # gets updated.
                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s,)
                    n_p_b, n_p_nb = next_beam[n_prefix]
                    if s != end_t:
                        n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                    else:
                        # We don't include the previous probability of not ending
                        # in blank (p_nb) if s is repeated at the end. The CTC
                        # algorithm merges characters not separated by a blank.
                        n_p_nb = logsumexp(n_p_nb, p_b + p)
              
                    # *NB* this would be a good place to include an LM score.
                    next_beam[n_prefix] = (n_p_b, n_p_nb)
    
                    # If s is repeated at the end we also update the unchanged
                    # prefix. This is the merging case.
                    if s == end_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = logsumexp(n_p_nb, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
    
            # Sort and trim the beam before moving on to the
            # next time-step.
            beam = sorted(next_beam.items(), key=lambda x : logsumexp(*x[1]), reverse=True)
            beam = beam[:beam_size]
            
        res_str.append([])
        scores.append([])
        for item in beam:
            res_str[-1].append(item[0])
            scores[-1].append(logsumexp(*item[1]))
            
    return res_str, scores
            
res_str, scores = prefix_beam_decode(rec_probs)
print()
print('prefix_beam_decode')
print(res_str)
print(scores)       
            
            
            
            
            
            