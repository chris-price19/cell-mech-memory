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
from copy import deepcopy

# import plotly
# import plotly.graph_objects as go
import multiprocessing as mp
from itertools import product
import timeit

from MMfxns import *
from pyDOE2 import lhs

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
        outgrid.at[pi,'result_prime_time'] = priming_times
        outgrid.at[pi,'mem_stiff'] = stiffA
        outgrid.at[pi,'prime_stiff'] = stiffP
        outgrid.at[pi,'a_max'] = params['a_max']

        outgrid.at[pi,'x_c'] = params['x_c']
        outgrid.at[pi,'m_c'] = params['m_c']
        
        # outgrid.at[pi,'t1max'] = params['t1max']
        # outgrid.at[pi,'t_prime'] = params['t_prime']
        # outgrid.at[pi,'dt'] = params['dt']

    # outgrid.index = pgrid.index

    return outgrid


cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)
pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)

mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

saveall = True

# print(plotly.__version__)
# print(matplotlib.__version__)
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def set_params_rates(file=None):
    params = {}
    if file:
        with open(file, 'r') as f:
            params = json.load(f)
    else:
      # params['tau'] = .98
        params['tau_F'] = 12. # params['tau'] * 2
        params['tau_SG'] = 200. #params['tau'] * 150
        params['tau_SR'] = params['tau_SG']
        
        params['kc'] = np.array([0.8, 2.]) # 1.5
        params['km'] = 'stiff'      
        params['x0'] = np.array([0.9, 2.5]) # 1.9
        params['a0'] = np.array([0.2, 1.2]) # 1.; 
        params['xtt'] = 0.
        params['g'] = np.array([10, 50]) # 35
        params['n'] = np.array([3, 8]) # 6        
        params['m0'] = np.array([4.5, 8.]) # 6.5
        params['a_max'] = 15

        params['time_resolution'] = 1.
        params['color'] = None
        params['input_m'] = []
        
        params['tau_R0'] = np.array([48, 240])  # 130 
        params['TV0SR'] = 1 # params['x0']
        params['TV0SG'] = 1 # params['x0']
        
        params['dynamics'] = 'updated_exp'
        params['eps'] = (0., 1., 0.05) # mean, std, magnitude        
        params['grid_resolution'] = 150
        params['input_primetime'] = np.array([48, 320])
    
    return params

params = set_params_rates()

if params['dynamics'] == 'updated_exp_staticTS':
    params['tau_SG'] = np.array([48, 240])
    params['tau_SR'] = np.array([120, 480])
    params['tau_R0'] = 1
    loop_over = ['a0', 'm0', 'x0', 'tau_SG', 'tau_SR', 'kc', 'n', 'g', 'input_primetime']
    fdir = './updated_exp_staticTS/'    
    # derivative = ['tau_F', V0SR', 'TV0SG']
    # exclude = ['color', 'input_m', 'eps', 'tau_R0']
else:
    loop_over = ['a0', 'm0', 'x0', 'tau_R0', 'kc', 'n', 'g', 'input_primetime']
    fdir = './update_exp/'
    # derivative = ['tau_F', 'tau_SG', 'TV0SR', 'TV0SG']

exclude = ['color', 'input_m', 'eps']

samples = lhs(len(loop_over), int(40000))

outputDF = pd.DataFrame()
si = 0
for ki, key in enumerate(params.keys()):
    if key in loop_over:
        print(key)
        print(params[key])
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
if params['dynamics'] != 'updated_exp_staticTS':
    outputDF['tau_SG'] = outputDF['tau_R0']
    outputDF['tau_SR'] = outputDF['tau_R0']
    outputDF['TV0SG'] = outputDF['x0']
    outputDF['TV0SR'] = outputDF['x0']

longeps = [
    (0., 0.,  0.)
        for pi in np.arange(len(outputDF))
    ]

outputDF['eps'] = longeps
## sample stiffnesses

rand1 = np.random.uniform(0., 1., size=len(outputDF))
rand2 = np.random.uniform(0., 1., size=len(outputDF))
rand3 = np.random.uniform(0.5, 0.99, size=len(outputDF))

ins = [
    np.array(
    [
        [24, -rand1[pi]],
        [outputDF['input_primetime'].iloc[pi], -rand2[pi]],
        [np.amax(params['input_primetime'])*10, -rand1[pi]],
    ]
        ) for pi in np.arange(len(outputDF))
    ]

# params['input_m'] = ins #[ins[i].tolist() for i in ins]
outputDF['input_m'] = ins
# acrit = alpha_crit(outputDF['n'], outputDF['tau'])
# outputDF['a0'] = rand3 * acrit

outputDF = outputDF.reindex(columns = outputDF.columns.tolist() + ['mem_stiff', 'prime_stiff', 'mem_time', 'result_prime_time'])
print('number of trials:',end=" "); print(len(outputDF))
print(outputDF)

# output = unpack_and_run(outputDF)
start = timeit.default_timer()
# n_proc = 16
n_proc = mp.cpu_count() # // 2
chunksize = len(outputDF) // n_proc
proc_chunks = []
for i_proc in range(n_proc):
    chunkstart = i_proc * chunksize
    # make sure to include the division remainder for the last process
    chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

    proc_chunks.append(outputDF.iloc[slice(chunkstart, chunkend)])

assert sum(map(len, proc_chunks)) == len(outputDF)

with mp.Pool(processes=n_proc) as pool:
    # starts the sub-processes without blocking
    # pass the chunk to each worker process
    proc_results = [pool.apply_async(unpack_and_run, args=(chunk,)) for chunk in proc_chunks]

    # blocks until all results are fetched
    result_chunks = [r.get() for r in proc_results]

results = pd.concat(result_chunks)

print(results)

# make sure we got a result for each coordinate pair:
assert len(results) == len(outputDF)

end = timeit.default_timer()

randlabel = np.random.randint(0,1000)

fname = fdir + 'psweep_plain_trial_' + str(randlabel)
results.loc[:, results.columns != 'input_m'].to_csv(fname+'.csv')
np.save(fname+'_inputs.npy', results['input_m'].values)
# test = np.load('inputs.npy', allow_pickle=True)

print((end - start)/3600, end = ' '); print(' hours')