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

def unpack_and_run(pgrid):
    ## insert the equation solving here
    ### extract parameter dict from dataframe
    ### solve equations
    ### add results to dataframe column

    outgrid = pgrid.copy(deep=True)
    outgrid = outgrid.reset_index(drop=True)
    print(len(pgrid))
    for pi in np.arange(len(outgrid)):

        params = outgrid.iloc[pi].to_dict()
        resultsDF = pd.DataFrame(columns=['m_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])
        resultsDF, params, priming_times, memory_times, stiffP, stiffA = run_profile(integrate_profile_Edependent, params['input_m'], params, resultsDF)
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


cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)
pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)

#### Energy dependent alpha increase and decrease
params = {}
resultsDF = pd.DataFrame(columns=['m_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])

## set constants, and upper / lower bounds of variables to test (the np.arrays)

params['tau'] = np.array([0.6, 1.2]) #, 5.)
params['tau_F'] = 1.
params['tau_SG'] = 300

params['m0'] = np.array([2., 10.])
params['x0'] = 0.2
params['a0'] = 0.05
params['a_max'] = 50
params['n'] = np.array([2.,5.])
params['resolution'] = .5     # hours per timestep
params['type'] = 'stiff'

params['tau_R0'] = np.array([50., 800.])
params['TV0SR'] = np.array([0.01, 1.00])
params['TV0SG'] = np.array([0.01, 2.00])

params['t_prime'] = 0.
params['t1max'] = 0.

params['x_c'] = 0.; params['a_c'] = 0.; params['m_c'] = 0.; params['dt'] = 0.

params['ptime'] = np.array([48, 480])

loop_over = ['m0','tau_R0','TV0SG','TV0SR','tau','ptime', 'n']
derivative = ['tau_F', 'tau_SG', 'pstiff', 'mstiff']

## latin hypercube sampling. set the number of trials here
samples = lhs(len(loop_over), int(7e4))

## setup data frame of experiments
outputDF = pd.DataFrame()
si = 0
for ki, key in enumerate(params.keys()):
    if key in loop_over:
        # print(key)
        # print(params[key])
        outputDF[key] = samples[:,si] * np.diff(params[key]) + np.amin(params[key])
        si += 1
    else:
        if isinstance(params[key], str):
            outputDF[key] = params[key]
        else:
            outputDF[key] = np.ones(len(samples[:,0])) * params[key]

## derivative columns (constrain values here)
outputDF['tau_F'] = outputDF['tau'] * 2.
outputDF['tau_SG'] = outputDF['tau_R0']

## sample stiffnesses

rand1 = np.random.uniform(0., 1., size=len(outputDF))
rand2 = np.random.uniform(0., 1., size=len(outputDF))

ins = [
    np.array(
    [
        [24, -rand1[pi]],
        [outputDF['ptime'].iloc[pi], -rand2[pi]],
        [np.amax(params['ptime'])*10, -rand1[pi]],
    ]
        ) for pi in np.arange(len(outputDF))
    ]

outputDF['input_m'] = ins

outputDF = outputDF.reindex(columns = outputDF.columns.tolist() + ['mem_stiff', 'prime_stiff', 'mem_time', 'prime_time'])
print('number of trials:',end=" "); print(len(outputDF))
print(outputDF)


### serial run
# output = unpack_and_run(outputDF)

#### parallelization
start = timeit.default_timer()
# n_proc = 16
## set # of processors
n_proc = mp.cpu_count() // 2
chunksize = len(outputDF) // n_proc
proc_chunks = []
## give each process a roughly equal chunk of the dataframe
for i_proc in range(n_proc):
    chunkstart = i_proc * chunksize
    # make sure to include the division remainder for the last process
    chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

    proc_chunks.append(outputDF.iloc[slice(chunkstart, chunkend)])

assert sum(map(len, proc_chunks)) == len(outputDF)

## run and collect results
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

fname = 'energy_dependent_LHS_ng'
results.loc[:, results.columns != 'input_m'].to_csv(fname+'.csv')
np.save(fname+'_inputs.npy', results['input_m'].values)
# test = np.load('inputs.npy', allow_pickle=True)

print((end - start)/3600, end = ' '); print(' hours')