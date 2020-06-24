#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:48:53 2020

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
# rec_probs = rec_probs / rec_probs.sum(axis=2).reshape(B, -1, 1)
# print(rec_probs)


# target = np.array([[0,3,0,2,0,4,0,1,0,1,0,1,0], [0,2,0,2,0,1,0,1,0,1,0,1,0]])

# TODO deal with FA
# TODO 先都用 for循环解决吧
def forward_backward_cal(target_i, rec_probs_i, L, is_backward=False):
    if is_backward:
        target_i = target_i[::-1]
        rec_probs_i = rec_probs_i[::-1]
    
    target_length_i = len(target_i)
    FA = np.zeros((L, target_length_i))
    # 初始化 A0
    FA[0, 0] = rec_probs_i[0, 0]
    FA[0, 1] = rec_probs_i[0, target_i[1]]
    
    for t in range(1, L):
        for k in range(max(0, target_length_i-(L-t)*2), min((t+1)*2, target_length_i)):     # recheck
            if k % 2 == 0:  
                # is blank
                if k-1 >= 0:
                    FA[t, k] = (FA[t-1, k] + FA[t-1, k-1]) * rec_probs_i[t, target_i[k]]
                else:
                    FA[t, k] = FA[t-1, k] * rec_probs_i[t, target_i[k]]
            else:
                # is char
                # pay attention to target_i[k] == target_i[k-1]
                if k == 1 or target_i[k] == target_i[k-2]:  
                    FA[t, k] = (FA[t-1, k-1] + FA[t-1, k]) * rec_probs_i[t, target_i[k]]
                else:
                    FA[t, k] = (FA[t-1, k-1] + FA[t-1, k-2] + FA[t-1, k]) * rec_probs_i[t, target_i[k]]
                    # FA[t-1, k] 写成了 FA[t-1, target_[k]]找了半天错我晕
                    
    return FA[::-1, ::-1] if is_backward else FA
    

def grad_cal_probs(rec_probs_i, target_i, prob_i):
    p = prob_i[0].sum()
    grad = np.zeros_like(rec_probs_i)
    grad_single = prob_i / rec_probs_i[:, target_i]
    for k in range(len(target_i)):
        grad[:, target_i[k]] += grad_single[:, k]
    grad = grad / p     # deal with  log
    return grad


def grad_cal_logits(rec_probs_i, target_i, prob_i):
    p = prob_i[0].sum()
    grad = np.zeros_like(rec_probs_i)
    grad_single = prob_i / rec_probs_i[:, target_i]
    for k in range(len(target_i)):
        grad[:, target_i[k]] += prob_i[:, k] / p
    grad -= rec_probs_i
    return grad


# another grad_cal_logits function
def grad_cal_logits_naive(rec_probs_i, target_i, prob_i):
    y_grad = grad_cal_probs(rec_probs_i, target_i, prob_i)
    sum_y_grad = np.sum(y_grad * rec_probs_i, axis=1, keepdims=True) # sum_y_grad = 1.
    u_grad = rec_probs_i * (y_grad - sum_y_grad) 
    return u_grad


# loss & grad calculation
loss_self = []
grad_prob, grad_logit = [], []
for i in range(B):
    target_ = target[i, :target_length[i]]
    # blank 扩展
    target_i = np.array([0 if i%2==0 else target_[i//2] for i in range(2*len(target_)+1)])  # [0,3,0,2,0,4,0,1,0,1,0,1,0]
    rec_probs_i = rec_probs[i, :, :]
    FA = forward_backward_cal(target_i, rec_probs_i, L)
    FB = forward_backward_cal(target_i, rec_probs_i, L, is_backward=True)
    # loss
    prob_i = FA * FB / rec_probs_i[:, target_i]
    loss_i = -np.log(prob_i[0].sum())      # 对某个t时刻求概率和即可，任意t概率求和是一样的    check -np.log(0)，
    loss_self.append(loss_i)
    # grad_prob
    grad_i = grad_cal_probs(rec_probs_i, target_i, prob_i)
    grad_i = -grad_i
    grad_prob.append(grad_i)   
    # grad_logit
    grad_i = grad_cal_logits(rec_probs_i, target_i, prob_i)
    grad_i = -grad_i
    grad_logit.append(grad_i)



# for check
import torch
from torch import nn
from torch.nn import functional as F
rec_logits = torch.tensor(rec_logits).detach().requires_grad_()
log_probs = torch.log_softmax(rec_logits, dim=2).transpose(0, 1)
targets = torch.tensor(target)
input_lengths = torch.full((1,), 12, dtype=torch.long)
target_lengths = torch.tensor([3])
loss_pytorch = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
loss_pytorch.backward()

print('------')
print('loss_pytorch: ', loss_pytorch.detach())
print('grad_pytorch: ')
print(rec_logits.grad.detach())
print('******')
print('loss_self: ', loss_self)
print('grad_self: ')
print(grad_logit)

    

'''
Attention:
    1. target_i[k] == target_i[k-2]
    2. -log
    3. gradients calculation
'''
    