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
import timeit

# import plotly
# import plotly.graph_objects as go

from MMfxns import *

cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)

mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

saveall = True

# print(plotly.__version__)
# print(matplotlib.__version__)
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#### Energy dependent alpha increase and decrease
params = {}
resultsDF = pd.DataFrame(columns=['m_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])
resultsDF2 = pd.DataFrame(columns=['m_profile','t_space','x_prof','alpha_prof','active_region','deltaV'])

params['tau'] = 1.
params['tau_F'] = params['tau'] * 2
params['tau_SG'] = params['tau'] * 150
params['tau_SR'] = params['tau_SG']

params['m0'] = 4.
params['x0'] = 0.2
params['a0'] = 0.2
params['a_max'] = 50
params['n'] = 3
params['resolution'] = .3     # hours per timestep
params['type'] = 'stiff'
params['color'] = None
params['input_m'] = []

params['tau_R0'] = params['tau_SG']
params['TV0'] = 0.05
params['TV02'] = 1.

# 480 hours = 20 days
# time in hours | # stiffness
# ins = np.array(
#     [
#         [24, np.log(2)],
#         [240, np.log(10)],
#         [200, np.log(2)],
#     ]
#         )

ins = np.array(
    [
        [24, (2 / params['m0'])],
        [240, (10 / params['m0'])],
        [200, (2 / params['m0'])],
    ]
        )

start = timeit.default_timer()

resultsDF, params, priming_times, memory_times, mech_ratio = run_profile(integrate_profile_Edependent, ins, params, resultsDF)

end = timeit.default_timer()
print(end - start, end = ' '); print('seconds')

fig, ax = plt.subplots(2, 2, figsize=(14,9), gridspec_kw={'height_ratios': [1, .75]})
fig.suptitle('Energy Dependent \u03b1')

params, fig, fig2, ax2 = plot_profile(fig, ax[:,0], resultsDF, params, colors = 
                                      [mcolors['darkorange'], mcolors['red'], mcolors['navy'], 
                                           mcolors['navy'], mcolors['springgreen'], mcolors['springgreen']])

params['input_m'].append(ins.tolist())

# print(resultsDF['tSG'])

plt.tight_layout()


fig3, ax3 = plt.subplots(1,1)

ax3.plot(np.arange(len(resultsDF))*params['resolution'], resultsDF['tSG'])

plt.show()