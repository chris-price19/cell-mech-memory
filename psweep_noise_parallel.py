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
from copy import deepcopy

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

def run_sim(params, trials):
    
    memout = []; primeout = []; pstiffout = []; mstiffout = []; amaxout = [];

    noiseDF = pd.DataFrame()
    base_params = deepcopy(params)

    for pi in np.arange(trials):
        start = timeit.default_timer()

        resultsDF = pd.DataFrame(columns=['trial_id', 'm_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])
        
        resultsDF, params, priming_times, memory_times, stiffP, stiffA = run_profile(integrate_profile, params['input_m'], params, resultsDF)
        resultsDF['trial_id'] = pi

        noiseDF = pd.concat((noiseDF, resultsDF), axis=0)
        memout.append(memory_times)
        primeout.append(priming_times)
        pstiffout.append(stiffP)
        mstiffout.append(stiffA)
        amaxout.append(params['a_max'])

        params = deepcopy(base_params)

        end = timeit.default_timer()

        print(end-start)

    # print(noiseDF)
        
    return noiseDF, params, memout, primeout, pstiffout, mstiffout, amaxout

def unpack_and_run(pgrid):
    
    ingrid = pgrid.copy(deep=True)
    ingrid = ingrid.reset_index(drop=True)
    outgrid = pd.DataFrame()
    print(len(pgrid))
    trials = 128
    for pi in np.arange(len(ingrid)):

        params = ingrid.iloc[pi].to_dict()
        
        # print(params)

        resultsDF, params, memout, primeout, pstiffout, mstiffout, amaxout = run_sim(params, trials)

        # print(params[])

        intergrid = pd.DataFrame(np.repeat(ingrid.iloc[[pi]].values, trials, axis=0), columns=ingrid.columns)

        intergrid['trial_id'] = pd.unique(resultsDF['trial_id'])
        intergrid['mem_time'] = memout
        # print(pgrid)
        intergrid['result_prime_time'] = primeout
        intergrid['mem_stiff'] = mstiffout
        intergrid['prime_stiff'] = pstiffout
        intergrid['a_max'] = amaxout

        print(params)

        outgrid = pd.concat((outgrid, intergrid), axis=0)
        # outgrid.at[pi,'x_c'] = params['x_c']
        # outgrid.at[pi,'a_c'] = params['a_c']
        # outgrid.at[pi,'m_c'] = params['m_c']
        # outgrid.at[pi,'dt'] = params['dt']
        print(outgrid)

    return outgrid

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
    
    return params

params = set_params_rates()

loop_over = ['a0', 'm0', 'x0', 'tau_R0', 'kc', 'n', 'g']
derivative = ['tau_F', 'tau_SG', 'TV0SR', 'TV0SG']
exclude = ['color', 'input_m', 'eps']

samples = lhs(len(loop_over), int(400))

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
outputDF['tau_SG'] = outputDF['tau_R0']
outputDF['tau_SR'] = outputDF['tau_R0']
outputDF['TV0SG'] = outputDF['x0']
outputDF['TV0SR'] = outputDF['x0']

## noise

longeps = [
    (0., np.random.uniform(0.5, 1.5),  np.random.uniform(0.001, 0.01))
        for pi in np.arange(len(outputDF))
    ]

outputDF['eps'] = longeps

concatframe = pd.DataFrame()
primetimes = [72, 168, 240]
# primetimes = [180]

for pi, pp in enumerate(primetimes):

    outputDF['input_primetime'] = pp
    outputDF['global_id'] = np.arange(len(outputDF))

    concatframe = pd.concat((concatframe, outputDF), axis=0)

outputDF = concatframe

# print(outputDF.loc[outputDF['global_id'] == 42])

ins = [
    np.array(
    [
        [24, 2.],
        [outputDF['input_primetime'].iloc[pi], 10.],
        [outputDF['input_primetime'].iloc[pi]*3, 2.],
    ]
        ) for pi in np.arange(len(outputDF))
    ]

outputDF['input_m'] = ins
outputDF = outputDF.reindex(columns = outputDF.columns.tolist() + ['mem_stiff', 'prime_stiff', 'mem_time', 'result_prime_time'])
print('number of trials:',end=" "); print(len(outputDF))
print(outputDF)

# output = unpack_and_run(outputDF)
# print(output)

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
print(len(results))
print(results)


# make sure we got a result for each coordinate pair:
# assert len(results) == len(outputDF)

end = timeit.default_timer()

fname = './noise_results_mswitch/psweep_noise_trial'
results.loc[:, results.columns != 'input_m'].to_csv(fname+'.csv')
np.save(fname+'_inputs.npy', results['input_m'].values)
# test = np.load('inputs.npy', allow_pickle=True)

print((end - start)/3600, end = ' '); print(' hours')