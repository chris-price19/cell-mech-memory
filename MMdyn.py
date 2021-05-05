#!/usr/bin/python

import math
import numpy as np
import scipy
from scipy.signal import argrelextrema, find_peaks
import scipy.ndimage as nd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import cm

import os
import sys
import re
import sympy

import cmath
import matplotlib.patches as patches
from matplotlib import colors as m2colors

cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)
mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

def static_alphaG(p,t):
    return p['a0'] * np.exp(t/p['tau_SG'])

def static_alphaR(p,t):
    return p['a0'] + (p['a_enter']-p['a0']) * np.exp(-t/p['tau_SR'])

def dyn_alphaG(p, t, dv):
    tau_SGR = p['tau_SGR'] * np.exp(dv/p['TV0SG'])
    return p['a0'] * np.exp(t/tau_SGR)
    
def dyn_alphaR(p, t, dv):
    tau_SGR = p['tau_SGR'] * np.exp(dv/p['TV0SR'])
    return p['a0'] + (p['a_enter']-p['a0']) * np.exp(-t/tau_SGR)

def plot_sample_dynamics(ax, t, a1, a2, params):

    dt = t[1] - t[0]        
    
    ax.plot(t, a1, linewidth=3., color=mcolors['lime'], label='priming')
    ax.plot(t + np.amax(t), a2, linewidth=3.,color=mcolors['lime'],label='memory')

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    w1 = t[a1 > params['a_c']][-1] - t[a1 > params['a_c']][0]
    w2 = t[a2 > params['a_c']][-1] - t[a2 > params['a_c']][0]

    # print(xlims)

    # ax[1].legend(loc=1)
    ax.plot([t[-1], t[-1]], [ylims[0], ylims[1]],
          color = mcolors['black'], linewidth=3., linestyle='-.')

    rect0 = patches.Rectangle((xlims[0], ylims[0]), 
                         t[-1]-w1 + t[0]-xlims[0], ylims[1]-ylims[0], 
                         color = '#1e88e5ff', #color=mcolors['darkorange'], 
                         # alpha=0.2
                         )

    rect1 = patches.Rectangle((t[-1] - w1, ylims[0]), 
                             w1, ylims[1]-ylims[0], 
                             color = '#1e88e5ff', #color=mcolors['red'], 
                             # alpha=0.2
                             )
    
    rect2 = patches.Rectangle((t[-1], ylims[0]), 
                             w2,ylims[1]-ylims[0], 
                             color= '#0f2080ff' # mcolors['springgreen'], 
                             # alpha=0.2
                             )
    
    rect3 = patches.Rectangle((w2 + t[-1], ylims[0]), 
                             t[-1] - w2 + xlims[1] - t[-1]*2, ylims[1]-ylims[0], 
                             color = '#f5793aff', #color=mcolors['darkviolet'], 
                         # alpha=0.2
                         )
    
    ax.add_patch(rect0); ax.add_patch(rect1); ax.add_patch(rect2); ax.add_patch(rect3)

    # ax.plot(xlims,[params['a_c'], params['a_c']], color = mcolors['blue'], linestyle='-.', linewidth=2., label='\u03b1$_{c}$')

    ax.set_xlim(xlims); ax.set_ylim(ylims)

    return ax

def plotwrap(params, res, figdim):
    
    t_space_long = np.linspace(0,240, res)
    t_space_short = np.linspace(0, 120, res//2)
    dv_space_short = np.linspace(0.001, 0.1, res//2)
    dv_space_long = np.linspace(0.001, 0.4, res)
    dt = t_space_long[1] - t_space_long[0]

    alpha_static_g = static_alphaG(params, t_space_long)
    params['a_enter'] = np.amax(alpha_static_g)
    alpha_static_r = static_alphaR(params, t_space_long)

    fig, ax = plt.subplots(2, 1, figsize=figdim)
    ax[0].set_xlim([np.amin(t_space_long), np.amax(t_space_long)*2])
    ax[1].set_xlim([np.amin(t_space_long), np.amax(t_space_long)*2])

    ax[1] = plot_sample_dynamics(ax[1], t_space_long, alpha_static_g, alpha_static_r, params)
    ax[0].set_xticks([])
    ax[1].set_xlabel('t') # (hours)
    ax[0].set_ylim(ax[1].get_ylim())

    alpha_static_g = static_alphaG(params, t_space_short)
    params['a_enter'] = np.amax(alpha_static_g)
    alpha_static_r = static_alphaR(params, t_space_short)
    tdiff = np.amax(t_space_long)*2 - np.amax(t_space_short) * 2
    t_diff_space = np.arange(0.0, tdiff, t_space_short[1] - t_space_short[0])
    params['a_enter'] = np.amin(alpha_static_r)
    xtra = static_alphaR(params, t_diff_space)

    ax[0] = plot_sample_dynamics(ax[0], t_space_short, alpha_static_g, alpha_static_r, params)
    ax[0].plot(t_diff_space + np.amax(t_space_short)*2, xtra, linewidth=3.,color=mcolors['darkviolet'],linestyle = '--')

    ## dynamic
    params['a_c'] = 1.5
    alpha_dyn_g = dyn_alphaG(params, t_space_long, dv_space_long)
    params['a_enter'] = np.amax(alpha_dyn_g)
    alpha_dyn_r = dyn_alphaR(params, t_space_long, np.flip(dv_space_long))

    fig2, ax2 = plt.subplots(2, 1, figsize=figdim)
    ax2[0].set_xlim([np.amin(t_space_long), np.amax(t_space_long)*2])
    ax2[1].set_xlim([np.amin(t_space_long), np.amax(t_space_long)*2])
    
    ax2[1] = plot_sample_dynamics(ax2[1], t_space_long, alpha_dyn_g, alpha_dyn_r, params)
    ax2[0].set_xticks([])
    ax2[1].set_xlabel('t') # (hours)
    ax2[0].set_ylim(ax2[1].get_ylim())
    
    params['TV0SR'] = 0.06
    alpha_dyn_g = alpha_dyn_g[:len(alpha_dyn_g)//2] # dyn_alphaG(params, t_space, dv_space)
    params['a_enter'] = np.amax(alpha_dyn_g)
    alpha_dyn_r = dyn_alphaR(params, t_space_short, np.flip(dv_space_short))

    tdiff = np.amax(t_space_long)*2 - np.amax(t_space_short) * 2
    t_diff_space = np.arange(0.0, tdiff, t_space_short[1] - t_space_short[0])
    params['a_enter'] = np.amin(alpha_dyn_r)
    xtra = dyn_alphaR(params, t_diff_space, dv_space_short[0])

    ax2[0] = plot_sample_dynamics(ax2[0], t_space_short, alpha_dyn_g, alpha_dyn_r, params)
    ax2[0].plot(t_diff_space + np.amax(t_space_short)*2, xtra, linewidth=3.,color=mcolors['darkviolet'],linestyle = '--')
    
    def format_tick_labels(x, pos):
        return '{0:.1f}'.format(x)

    ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(format_tick_labels))
    ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(format_tick_labels))

    ax2[0].yaxis.set_major_formatter(ticker.FuncFormatter(format_tick_labels))
    ax2[1].yaxis.set_major_formatter(ticker.FuncFormatter(format_tick_labels))

    fig.tight_layout()
    fig2.tight_layout()

    return fig, fig2, ax, ax2