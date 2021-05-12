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
        
        params['kc'] = 1.5 # '-stiff' # 0.98 # 'linear' # 0.5 #'soft' 1.
        params['km'] = 'stiff'      
        params['x0'] = 1.9; 
        params['a0'] = 1.; params['xtt'] = 0.
        params['g'] = 35.
        params['n'] = 7.        
        params['m0'] = 6.5
        params['a_max'] = 15

        params['time_resolution'] = 1.
        params['color'] = None
        params['input_m'] = []
        
        params['tau_R0'] = 130 # params['tau_SG'] #* 2
        params['TV0SR'] = params['x0']
        params['TV0SG'] = params['x0']
        
        params['dynamics'] = 'updated_exp_staticTS'
        params['eps'] = (0., 1., 0.005) # mean, std, magnitude        
        params['grid_resolution'] = 150
    
    return params

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


ptime = 240 # 240 # 120

ins = np.array(
    [
        [24, 2.],
        [ptime, 10.], #320
        [480, 2.],
    ]
        )

params = set_params_rates()
# params['input_m'].append(ins.tolist())

if params['dynamics'] == 'updated_exp':
    fdir = './noise_results_mswitch/'
elif params['dynamics'] == 'updated_exp_staticTS':
    fdir = './linear_noise_results_mswitch/'

start = timeit.default_timer()

numtrials = np.arange(8)

memappend = [];

# for ni, nn in enumerate(numtrials):
#     params = set_params()
#     out = run_sim(ins, params)
#     print(out)
#     memappend.append(out)

print(memappend)
n_proc = mp.cpu_count() # // 2
chunksize = len(numtrials) // n_proc
proc_chunks = []
for i_proc in range(n_proc):
    chunkstart = i_proc * chunksize
    # make sure to include the division remainder for the last process
    chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

    proc_chunks.append(numtrials[chunkstart:chunkend])

# assert sum(map(len, proc_chunks)) == len(outputDF)
print(proc_chunks)
with mp.Pool(processes=n_proc) as pool:

    proc_results = [pool.apply_async(run_sim, args=(ins, params, chunk,)) for chunk in proc_chunks]

    # blocks until all results are fetched
    result_chunks = [r.get() for r in proc_results]

results = pd.concat(result_chunks)
# print(result_chunks)
# results = [item for sublist in result_chunks for item in sublist]
# print(results)
# results = np.array([i[0] for i in results])

print(results)
print(len(results))

# np.save('../figures_v2/figure6_exp_fits/'+str(ptime)+'.N'+str(np.amax(numtrials))+'.npy', results)
# with open('../figures_v2/figure6_exp_fits/'+str(ptime)+'.N'+str(np.amax(numtrials))+'.json', 'w') as f:
#     f.write(json.dumps(params))

label = np.random.randint(0,1000)

np.save(fdir + str(label) + '.P'+str(ptime)+'.N'+str(np.amax(numtrials))+'.npy', results)
with open(fdir + str(label) + '.P'+str(ptime)+'.N'+str(np.amax(numtrials))+'.json', 'w') as f:
    f.write(json.dumps(params))

end = timeit.default_timer()
print((end - start)/3600, end = ' '); print(' hours') 
