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
from copy import deepcopy

import matplotlib.patches as patches
from matplotlib import colors as m2colors

# import plotly
# import plotly.graph_objects as go

from MMfxns import *
# from MMplotting import *
from pyDOE2 import lhs

import multiprocessing as mp
import timeit

cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)

mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

saveall = True

# print(plotly.__version__)
# print(matplotlib.__version__)
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)

def run_sim(params):
    
    memout = []; primeout = []; pstiffout = []; mstiffout = []; amaxout = [];

    noiseDF = pd.DataFrame()
    base_params = deepcopy(params)

    # for pi in np.arange(trials):
    start = timeit.default_timer()

    resultsDF = pd.DataFrame(columns=['trial_id', 'm_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])
    
    resultsDF, params, priming_times, memory_times, stiffP, stiffA = run_profile(integrate_profile, params['input_m'], params, resultsDF)
    resultsDF['trial_id'] = 0

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
    for pi in np.arange(len(ingrid)):

        params = ingrid.iloc[pi].to_dict()
        
        # print(params)

        resultsDF, params, memout, primeout, pstiffout, mstiffout, amaxout = run_sim(params)

        # print(params[])

        intergrid = pd.DataFrame(ingrid.iloc[[pi]].values, columns=ingrid.columns)

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

def set_params(file=None):
    params = {}
    if file:
        with open(file, 'r') as f:
            params = json.load(f)
    else:
        params['tau'] = 1.05
        params['tau_F'] = params['tau'] * 5 # params['tau'] * 2
        params['tau_SG'] = 360 #params['tau'] * 150
        params['tau_SR'] = params['tau_SG']

        params['m0'] = 6.
        params['x0'] = 0.3
        params['a0'] = .8
        params['a_max'] = 50
        params['n'] = 4.
        params['resolution'] = 1.
        params['type'] = 'stiff'
        params['color'] = None
        params['input_m'] = []

        params['tau_R0'] = 200 # params['tau_SG'] #* 2
        params['TV0SR'] = 0.1
        params['TV0SG'] = 3.
        
        params['dynamics'] = 'exp_dynamicTS'
#         params['eps'] = noise(0., 1., 0.02)
    
    return params

def set_params_rates(file=None):
    params = {}
    if file:
        with open(file, 'r') as f:
            params = json.load(f)
    else:
#         params['tau'] = .98
        params['tau_F'] = 12. # params['tau'] * 2
        params['tau_SG'] = 200. #params['tau'] * 150
        params['tau_SR'] = params['tau_SG']
        
        params['kc'] = 1.05 # '-stiff' # 0.98 # 'linear' # 0.5 #'soft' 1.
        params['km'] = 'stiff'      
        params['x0'] = 1.16; 
        params['a0'] = 0.82; params['xtt'] = 0.
        params['g'] = 38.
        params['n'] = 4.86
        params['m0'] = 6.05
        params['a_max'] = 15

        params['time_resolution'] = 1.
        params['color'] = None
        params['input_m'] = []
        
        params['tau_R0'] = 160.2 # params['tau_SG'] #* 2
        params['TV0SR'] = params['x0']
        params['TV0SG'] = params['x0']
        
        params['dynamics'] = 'updated_exp'
        params['eps'] = (0., 1., 0.011) # mean, std, magnitude        
        params['grid_resolution'] = 150
    
    return params


ptime = 240 # 240 # 120

ins = np.array(
    [
        [24, 2.],
        [ptime, 10.], #320
        [480, 2.],
    ]
        )

numtrials = np.arange(1024)

params = set_params_rates()

exclude = ['color', 'input_m', 'eps']

outputDF = pd.DataFrame()
for ki, key in enumerate(params.keys()):
    # if key in loop_over:
    # print(key)
    # outputDF[key] = params[key]
    if key not in exclude:
        if isinstance(params[key], str):
            outputDF[key] = params[key]
        else:
            print(key)
            # print(params[key])
            outputDF[key] = np.ones(len(numtrials)) * params[key]

longeps = [
    (0., np.random.uniform(0.5, 1.5),  np.random.uniform(0.001, 0.01))
        for pi in np.arange(len(outputDF))
    ]

outputDF['eps'] = longeps
outputDF['input_m'] = [
    ins for pi in np.arange(len(outputDF))
    ]

outputDF['global_id'] = np.zeros(len(outputDF))

outputDF = outputDF.reindex(columns = outputDF.columns.tolist() + ['mem_stiff', 'prime_stiff', 'mem_time', 'result_prime_time'])
print('number of trials:',end=" "); print(len(outputDF))
print(outputDF)
# sys.exit()

if params['dynamics'] == 'updated_exp':
    fdir = './noise_results_mswitch/'
elif params['dynamics'] == 'updated_exp_staticTS':
    fdir = './linear_noise_results_mswitch/'

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
    proc_results = [pool.apply_async(unpack_and_run, args=(chunk, )) for chunk in proc_chunks]

    # blocks until all results are fetched
    result_chunks = [r.get() for r in proc_results]

results = pd.concat(result_chunks)
print(len(results))
print(results)

# np.save('../figures_v2/figure6_exp_fits/'+str(ptime)+'.N'+str(np.amax(numtrials))+'.npy', results)
# with open('../figures_v2/figure6_exp_fits/'+str(ptime)+'.N'+str(np.amax(numtrials))+'.json', 'w') as f:
#     f.write(json.dumps(params))

randlabel = np.random.randint(0,1000)
fname = fdir + 'single_noise_' + str(randlabel)
results.loc[:, results.columns != 'input_m'].to_csv(fname+'.csv')
np.save(fname+'_inputs.npy', results['input_m'].values)

end = timeit.default_timer()
print((end - start)/3600, end = ' '); print(' hours') 
