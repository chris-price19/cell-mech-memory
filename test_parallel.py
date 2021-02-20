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

from MMfxns import *

def unpack_and_run(pgrid):
    
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

mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

saveall = True

# print(plotly.__version__)
# print(matplotlib.__version__)
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#### Energy dependent alpha increase and decrease
params = {}
resultsDF = pd.DataFrame(columns=['m_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])

params['tau'] = 0.5
params['tau_F'] = params['tau'] * 4
params['tau_SG'] = params['tau'] * 500
# params['tau_SG'] = 
# params['tau_SR'] = params['tau_SG']

# params['m0'] = 2.5
params['m0'] = np.linspace(2.,3., 5)
params['x0'] = 0.2
params['a0'] = 0.2
params['a_max'] = 50
params['n'] = np.linspace(3,4.5,3)
params['resolution'] = .5     # hours per timestep
params['type'] = 'stiff'
params['color'] = None
params['input_m'] = []

params['tau_R0'] = np.linspace(params['tau'] * 200, params['tau'] * 800, 5)
params['TV0SR'] = 0.1
params['TV0SG'] = np.linspace(0.3, 0.7, 3)

params['t_prime'] = 0.
params['t1max'] = 0.

params['x_c'] = 0.; params['a_c'] = 0.; params['m_c'] = 0.; params['dt'] = 0.

ins = [np.array(
    [
        [24, 2],
        [240, 10],
        [1000, 2],
    ]
        ),
    np.array(
    [
        [24, 2],
        [120, 10],
        [1000, 2],
    ]
        ),
    np.array(
    [
        [24, 2],
        [48, 10],
        [1000, 2],
    ]
        ),
    np.array(
    [
        [24, 2],
        [360, 10],
        [1000, 2],
    ]
        )
    ]


params['input_m'] += ins #[ins[i].tolist() for i in ins]

loop_over = ['m0','tau_R0','TV0SG','n','input_m'] # input_m has to be in here, janky but sorry! there is a better way, can't figure it out now
subset = {key: params[key] for key in loop_over}
outputDF = pd.DataFrame([dict(zip(subset.keys(),v)) for v in product(*subset.values())])

for key in params.keys():
    if key not in loop_over:
        # print(key)
        # print(params[key])
        if isinstance(params[key], list):
            outputDF[key] = [params[key] for i in np.arange(len(test))]
        else:
            outputDF[key] = params[key]

outputDF = outputDF.reindex(columns = outputDF.columns.tolist() + ['mem_stiff', 'prime_stiff', 'mem_time', 'prime_time'])
print('number of trials:',end=" "); print(len(outputDF))
print(outputDF)
# print(outputDF[['input_m']].iloc[0].values[0][0].shape)
# print(type(outputDF[['input_m']].values))
# print(outputDF['input_m'].values)

# output = unpack_and_run(outputDF)

n_proc = 4
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

fname = 'energy_dependent_paramtest_SGonly'
results.loc[:, results.columns != 'input_m'].to_csv(fname+'.csv')
np.save(fname+'_inputs.npy', results['input_m'].values)
# test = np.load('inputs.npy', allow_pickle=True)