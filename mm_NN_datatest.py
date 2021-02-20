#!/usr/bin/python

import math
import numpy as np
import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import re

# import autograd as ag
import torch
from torch.autograd import grad, Variable
from torch.utils.data import Sampler, Dataset, DataLoader, ConcatDataset
# from pyDOE2 import lhs

import timeit

cwd = os.getcwd()
pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)

df = pd.read_csv('./stiff_results/static_LHS_SG_SR_ng.csv')

feature_cols = ['tau', 'tau_F', 'tau_SG', 'tau_SR', 'm0', 'n', 'ptime', 'mem_stiff', 'prime_stiff', 'prime_time']
target_cols = ['mem_time']

df['tm_over_tp'] = df['mem_time'] / df['prime_time']

print(df.sort_values(by='tm_over_tp', ascending=False))

print(df[target_cols].min())
print(df[target_cols].max())

# df = df[feature_cols + target_cols]

df['tm_over_tp'].hist(bins=100)

plt.show()
# fig, ax = plt.subplots(1,2, figsize=(8,6))

# ax[0].plot(np.arange(len(train_loss)), train_loss)