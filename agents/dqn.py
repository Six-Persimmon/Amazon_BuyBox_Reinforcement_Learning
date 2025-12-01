#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   dqn.py
@Time    :   2025/10/08 13:52:27
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   DQN method designed for N-player pricing competition.
'''
import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F

class ReplayBuffer:
    """Experience Replay Buffer for DQN."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): # add data to buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # sample data from buffer, number = batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # current number of elements in buffer
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    ''' Q-network with a single hidden layer '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        return self.fc2(x)