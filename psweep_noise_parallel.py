#!/usr/bin/python

import math
import cmath
import numpy as np
import scipy
import sympy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from pylab import cm
import json

import os
import sys
import re

import matplotlib.patches as patches
from matplotlib import colors as m2colors

# import plotly
# import plotly.graph_objects as go
import multiprocessing as mp
from itertools import product
import timeit

from MMfxns import *
from pyDOE2 import lhs

cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)

mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

saveall = True

# print(plotly.__version__)
print(matplotlib.__version__)
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)

def unpack_and_run(pgrid):
    
    outgrid = pgrid.copy(deep=True)
    outgrid = outgrid.reset_index(drop=True)
    print(len(pgrid))
    for pi in np.arange(len(outgrid)):

        params = outgrid.iloc[pi].to_dict()
        resultsDF = pd.DataFrame(columns=['m_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])
        resultsDF, params, priming_times, memory_times, stiffP, stiffA = run_profile(integrate_profile, params['input_m'], params, resultsDF)
        # print(type(memory_times))
        # print(type(priming_times))
        outgrid.at[pi,'mem_time'] = memory_times
        # print(pgrid)
        outgrid.at[pi,'prime_time'] = priming_times
        outgrid.at[pi,'mem_stiff'] = stiffA
        outgrid.at[pi,'prime_stiff'] = stiffP

        outgrid.at[pi,'x_c'] = params['x_c']
        outgrid.at[pi,'a_c'] = params['a_c']
        outgrid.at[pi,'m_c'] = params['m_c']
        # outgrid.at[pi,'t1max'] = params['t1max']
        # outgrid.at[pi,'t_prime'] = params['t_prime']
        outgrid.at[pi,'dt'] = params['dt']

    # outgrid.index = pgrid.index

    return outgrid

def run_sim(ins, params, trials):
    
    memout = []
#     params = deepcopy(params)
    print(len(trials))
    # sys.exit()
    for pi in np.arange(len(trials)):

        resultsDF = pd.DataFrame(columns=['m_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])
        resultsDF, params, priming_times, memory_times, stiffP, stiffA = run_profile(integrate_profile, ins, params, resultsDF)
        memout.append(memory_times)
        params = set_params_rates()
        
    return memout

def set_params_rates(file=None):
    params = {}
    if file:
        with open(file, 'r') as f:
            params = json.load(f)
    else:
#         params['tau'] = .98
        params['tau_F'] = 24. # params['tau'] * 2
        # params['tau_SG'] = 260. #params['tau'] * 150
        # params['tau_SR'] = params['tau_SG']
        
        params['kc'] = np.array([0.8, 2.]) # 1. # '-stiff' # 0.98 # 'linear' # 0.5 #'soft' 1.
        params['km'] = 'stiff'      
        params['x0'] = np.array([1., 2.5]); 
        params['a0'] = 1.; params['xtt'] = 0.
        
        params['m0'] = np.array([4., 8.]) # 6.
        params['a_max'] = 50
        params['n'] = np.array([3., 7.])
        params['resolution'] = 1.
#         params['type'] = 'stiff'
        params['color'] = None
        params['input_m'] = []
        
        # params['tau_R0'] = 200 # params['tau_SG'] #* 2
        # params['TV0SR'] = 0.05
        # params['TV0SG'] = 1.8

        params['tau_R0'] = np.array([48., 480.])
        params['TV0SR'] = np.array([0.003, 1.00])
        params['TV0SG'] = np.array([0.003, 2.50])
        
        params['dynamics'] = 'exp_dynamicTS'
        params['eps'] = (0., 1., 0.008) # mean, std, magnitude
        
        params['res'] = 100
    
    return params

params = set_params_rates()

loop_over = ['m0','tau_R0','TV0SG','TV0SR', 'kc', 'x0', 'n']
derivative = ['tau_F', 'tau_SG', 'pstiff', 'mstiff']
exclude = ['color', 'res', 'resolution', 'dynamics', 'input_m', 'eps'] # 'color', 'type'

samples = lhs(len(loop_over), int(100))

outputDF = pd.DataFrame()
si = 0
for ki, key in enumerate(params.keys()):
    if key in loop_over:
        # print(key)
        # print(params[key])
        outputDF[key] = samples[:,si] * np.diff(params[key]) + np.amin(params[key])
        si += 1
    elif key not in exclude:
        if isinstance(params[key], str):
            outputDF[key] = params[key]
        else:
            print(key)
            # print(params[key])
            outputDF[key] = np.ones(len(samples[:,0])) * params[key]

## derivative columns
# outputDF['tau_F'] = outputDF['tau'] #* 2.
outputDF['tau_SG'] = outputDF['tau_R0']
outputDF['tau_SR'] = outputDF['tau_R0']

outputDF['input_m'] = ins
acrit = alpha_crit(outputDF['n'], outputDF['tau'])
outputDF['a0'] = rand3 * acrit

outputDF = outputDF.reindex(columns = outputDF.columns.tolist() + ['mem_stiff', 'prime_stiff', 'mem_time', 'prime_time'])
print('number of trials:',end=" "); print(len(outputDF))
print(outputDF)