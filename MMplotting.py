#!/usr/bin/python

import math
import numpy as np
import scipy
from scipy.signal import argrelextrema
import scipy.ndimage as nd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import cm

import plotly
import plotly.graph_objects as go

import os
import sys
import re
import sympy

import cmath
import matplotlib.patches as patches
from matplotlib import colors as m2colors

from MMfxns import *

cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)
mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

def plot_blank_PD(params):

    fig2, ax2 = plt.subplots(1,1,figsize=(7,5.5))

    # a_space, mtst = calc_PD(params)
    # ax2.plot(a_space[a_space>=params['a_c']], np.real(mtst[a_space>=params['a_c'],0]), color = mcolors['darkviolet'], linewidth=3., label='m1')
    # ax2.plot(a_space[a_space>=params['a_c']], np.real(mtst[a_space>=params['a_c'],1]), color = mcolors['dodgerblue'], linewidth=3., label='m2')
    ax2.set_xlabel('\u03b1')
    ax2.set_ylabel('m')

    ax2.plot([params['a_c'], params['a_c']],[0, params['m_c']+1], mcolors['green'], linewidth=3.5, linestyle='-.', label='\u03b1$_{c}$')
    ax2.plot([0, params['a_max']+0.5], [params['m_c'], params['m_c']], mcolors['deeppink'], linewidth=3.5, linestyle='-.', label='m$_{c}$')
    ## a_max
    ax2.plot([params['a_max'], params['a_max']],[0, params['m_c']+1], mcolors['firebrick'], linewidth=3., linestyle='-.')

    # ax2.legend(loc='upper center', bbox_to_anchor=(0.75, 1.15), ncol=2)

    # ax2.set_xlim([0,params['a_max']+0.5])
    ax2.set_xlim([0,2*params['a_c']])
    ax2.set_ylim([0,2*params['m_c']])
    
    xlims = ax2.get_xlim()
    ylims = ax2.get_ylim()
    
    # ax2.plot(a,m,color=mcolors['black'])
    # mred = np.concatenate((m[np.abs(np.diff(m, prepend=0))>0], m[np.abs(np.diff(m, append=0))>0]))
    # ared = np.concatenate((a[np.abs(np.diff(m, prepend=0))>0], a[np.abs(np.diff(m, append=0))>0]))
    # tred = np.concatenate((t[np.abs(np.diff(m, prepend=0))>0], t[np.abs(np.diff(m, append=0))>0]))

    # ax[1].scatter(tred, mred, color='r', s=120)
    # ax2.scatter(ared, mred, color='r', s=120)

    plt.tight_layout()

    return fig2

def subplot_PD(ax2, params, colors = [mcolors['darkorange'], mcolors['red'], mcolors['navy'], mcolors['darkviolet'], mcolors['deepskyblue'], mcolors['springgreen']]):
    
    a_space, mtst = calc_PD(params)

    a_term = a_space[np.where(mtst < 0)[0]][0]
    # print('aterm', end=" "); print(a_term)
    # for mm in np.arange(len(mtst)):
    #     print(mtst[mm,:])
    # removing negative roots, essentially for ease of plotting purposes. is there a deeper meaning to the singularity for soft priming?
    ##### **** ####
    if params['type'] == 'soft':
        mtst[mtst < 0] = params['m_c']*2 + 1
    ##### **** ####

    ax2.plot(a_space[a_space>=params['a_c']], np.real(mtst[a_space>=params['a_c'],0]), color = mcolors['darkviolet'], linewidth=3., label='memory loss')
    # ax2[0].plot(a_space[a_space>=params['a_c']], np.real(mtst[a_space>=params['a_c'],1]), color = mcolors['dodgerblue'], linewidth=3., label='m2')
    # ax2.set_xlabel('\u03b1')
    # ax2.set_ylabel('m/m0')

    ### turn on for fig 3, off otherwise
    # ax2.set_xlim([0,params['a_max']+0.5])
    # ax2.set_xlim([0,params['a_c'] * 2])
    # ax2.set_ylim([0,params['m_c'] * 2])        

    ax2.plot([params['a_c'], params['a_c']],[0, np.amax(ax2.get_ylim())], mcolors['green'], linewidth=4., linestyle='-.', label='\u03b1$_{c}$')
    ax2.plot([0, np.amax(ax2.get_xlim())], [params['m_c'], params['m_c']], mcolors['deeppink'], linewidth=4., linestyle='-.',  label='m$_{c}$')
    ## a_max
    ax2.plot([params['a_max'], params['a_max']],[0, params['m_c']+1], mcolors['firebrick'], linewidth=3., linestyle='-.')
   
    xlims = ax2.get_xlim()
    ylims = ax2.get_ylim()
    
    if params['type'] == 'stiff':
        
        ### turn on for fig 3, off otherwise
        # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        for ni, nn in enumerate((np.arange(6)+1)):
            if nn == 1:
                rect = patches.Rectangle((xlims[0],params['m_c']),params['a_c']-xlims[0],ylims[1]-params['m_c'], color=colors[nn-1], alpha=0.2)
                ax2.add_patch(rect)
            elif nn == 2:            
                rect = patches.Rectangle((params['a_c'],params['m_c']),xlims[1]-params['a_c'], ylims[1]-params['m_c'], color=colors[nn-1], alpha=0.2)
                ax2.add_patch(rect)
            elif nn == 3:
                rect = patches.Rectangle((xlims[0],ylims[0]), params['a_c']-xlims[0], params['m_c']-ylims[0], color=colors[nn-1], alpha=0.2)
                ax2.add_patch(rect)
            elif nn == 4:
                zmask = np.zeros(len(a_space[a_space>=params['a_c']]))
                ax2.fill_between(a_space[a_space>=params['a_c']], zmask, np.real(mtst[a_space>=params['a_c'],0]), color=colors[nn-1], alpha=0.2)
            elif nn == 5 or nn ==6:
                m_cmask = np.ones(len(a_space[a_space>=params['a_c']])) * params['m_c']
                ax2.fill_between(a_space[a_space>=params['a_c']], np.real(mtst[a_space>=params['a_c'],0]), m_cmask, color=colors[nn-1], alpha=0.2) # np.real(mtst[a_space>=params['a_c'],1]), color=colors[nn-1], alpha=0.2)
            # elif nn == 6:
            #     m_cmask = np.ones(len(a_space[a_space>=params['a_c']])) * params['m_c']
            #     ax2[0].fill_between(a_space[a_space>=params['a_c']], m_cmask, np.real(mtst[a_space>=params['a_c'],1]), color=colors[nn-1], alpha=0.2)
            else:
                break
    elif params['type'] == 'soft':
        ax2[0].legend(loc='lower right')
        for ni, nn in enumerate((np.arange(6)+1)):
            if nn == 1:
                rect = patches.Rectangle((xlims[0],ylims[0]), params['a_c']-xlims[0], params['m_c']-ylims[0], color=colors[nn-1], alpha=0.2)
                ax2[0].add_patch(rect)
            elif nn == 2:            
                rect = patches.Rectangle((params['a_c'],xlims[0]),xlims[1]-params['a_c'], params['m_c'], color=colors[nn-1], alpha=0.2)
                ax2[0].add_patch(rect)
            elif nn == 3:                
                rect = patches.Rectangle((xlims[0],params['m_c']),params['a_c']-xlims[0],ylims[1]-params['m_c'], color=colors[nn-1], alpha=0.2)
                ax2[0].add_patch(rect)
            elif nn == 4:
                m_cmask = np.ones(len(a_space[a_space>=params['a_c']])) * ylims[1]
                ax2[0].fill_between(a_space[a_space>=params['a_c']], m_cmask, np.real(mtst[a_space>=params['a_c'],0]), color=colors[nn-1], alpha=0.2)
            elif nn == 5:
                ax2[0].fill_between(a_space[a_space>=params['a_c']], np.real(mtst[a_space>=params['a_c'],1]), np.real(mtst[a_space>=params['a_c'],0]), color=colors[nn-1], alpha=0.2)
            elif nn == 6:
                zmask = np.ones(len(a_space[a_space>=params['a_c']])) * params['m_c']
                ax2[0].fill_between(a_space[a_space>=params['a_c']], zmask, np.real(mtst[a_space>=params['a_c'],1]), color=colors[nn-1], alpha=0.2)               
            else:
                break

        return ax2

def plot_full_PD(params, resultsDF, colors = [mcolors['darkorange'], mcolors['red'], mcolors['navy'], mcolors['darkviolet'], mcolors['deepskyblue'], mcolors['springgreen']]):

    # fig2, ax2 = plt.subplots(1,2,figsize=(12, 7), gridspec_kw={'width_ratios': [1, .35]})

    if np.amax(resultsDF['m_profile']) / params['m0'] * 0.85 > params['m_c'] * 2 and np.amax(resultsDF['alpha_prof']) * 0.85 > params['a_c'] * 2:

        ## double break
        fig = plt.figure(constrained_layout=False, figsize=(10,8))
        # gs_kw = dict(width_ratios=4, height_ratios=4)
        gs = fig.add_gridspec(nrows=4,ncols=4) #, left=)
        axBL = fig.add_subplot(gs[1:,:-1]) #, gridspec_kw=gs_kw)
        axTL = fig.add_subplot(gs[0,:-1])
        axTR = fig.add_subplot(gs[0,-1])
        axBR = fig.add_subplot(gs[1:,-1])

        axTL.spines['bottom'].set_visible(False)
        axTL.spines['right'].set_visible(False)
        axTL.xaxis.tick_top()
        axTL.tick_params(labeltop=False)
        axTL.get_xaxis().set_ticks([])

        axBL.spines['top'].set_visible(False)
        axBL.spines['right'].set_visible(False)
        axBL.xaxis.tick_bottom()

        axBR.spines['left'].set_visible(False)
        axBR.spines['top'].set_visible(False)
        axBR.yaxis.tick_left()
        axBR.get_yaxis().set_ticks([])

        axTR.spines['bottom'].set_visible(False)
        axTR.spines['left'].set_visible(False)
        axTR.xaxis.tick_top()
        axTR.tick_params(labeltop=False)
        axTR.get_xaxis().set_ticks([])
        axTR.get_yaxis().set_ticks([])

        axBL.set_xlim(0., params['a_c'] * 2 * 0.85)
        axBL.set_ylim(0., params['m_c'] * 2 * 0.85)

        axTL.set_xlim(axBL.get_xlim())
        axTL.set_ylim(0.85 * resultsDF['m_profile'].max()/params['m0'], 1.05 * resultsDF['m_profile'].max()/params['m0'])

        axBR.set_xlim(0.85 * resultsDF['alpha_prof'].max(), 1.05 * resultsDF['alpha_prof'].max())
        axBR.set_ylim(axBL.get_ylim())
        axBR.set_xticks([np.round(resultsDF['alpha_prof'].max(),1)])

        axTR.set_xlim(axBR.get_xlim())
        axTR.set_ylim(axTL.get_ylim())

        axlist = [axBL, axTL, axBR, axTR]

        for ai, aa in enumerate(axlist):

            aa = subplot_PD(aa, params, colors)
            axlist[ai] = aa

        axTL.legend(loc='upper center', bbox_to_anchor=(0.7, 1.5), ncol=3)
        # axBL.set_xlabel('\u03b1')
        # axBL.set_ylabel('m/m0')

        ### y-axis break
        dX = .03
        dY = .03

        kwargs = dict(transform=axTL.transData, color='k', clip_on=False)
        axTL.plot((np.amin(axTL.get_xlim())-dX, np.amin(axTL.get_xlim())+dX), (np.amin(axTL.get_ylim())-dY, np.amin(axTL.get_ylim())+dY), **kwargs)        # top-left diagonal

        kwargs = dict(transform=axBL.transData, color='k', clip_on=False)
        axBL.plot((np.amin(axBL.get_xlim())-dX, np.amin(axBL.get_xlim())+dX), (np.amax(axBL.get_ylim())-dY, np.amax(axBL.get_ylim())+dY), **kwargs)

        kwargs = dict(transform=axBR.transData, color='k', clip_on=False)
        axBR.plot((np.amax(axBR.get_xlim())-dX, np.amax(axBR.get_xlim())+dX), (np.amax(axBR.get_ylim())-dY, np.amax(axBR.get_ylim())+dY), **kwargs)

        kwargs = dict(transform=axTR.transData, color='k', clip_on=False)
        axTR.plot((np.amax(axTR.get_xlim())-dX, np.amax(axTR.get_xlim())+dX), (np.amin(axTR.get_ylim())-dY, np.amin(axTR.get_ylim())+dY), **kwargs)

        
        ### x axis break
        dX = .03
        dY = .03

        kwargs = dict(transform=axTL.transData, color='k', clip_on=False)
        axTL.plot((np.amax(axTL.get_xlim())-dX, np.amax(axTL.get_xlim())+dX), (np.amax(axTL.get_ylim())-dY, np.amax(axTL.get_ylim())+dY), **kwargs)        # top-left diagonal

        kwargs = dict(transform=axBL.transData, color='k', clip_on=False)
        axBL.plot((np.amax(axBL.get_xlim())-dX, np.amax(axBL.get_xlim())+dX), (np.amin(axBL.get_ylim())-dY, np.amin(axBL.get_ylim())+dY), **kwargs)

        kwargs = dict(transform=axBR.transData, color='k', clip_on=False)
        axBR.plot((np.amin(axBR.get_xlim())-dX, np.amin(axBR.get_xlim())+dX), (np.amin(axBR.get_ylim())-dY, np.amin(axBR.get_ylim())+dY), **kwargs)

        kwargs = dict(transform=axTR.transData, color='k', clip_on=False)
        axTR.plot((np.amin(axTR.get_xlim())-dX, np.amin(axTR.get_xlim())+dX), (np.amax(axTR.get_ylim())-dY, np.amax(axTR.get_ylim())+dY), **kwargs)

    elif np.amax(resultsDF['m_profile']) / params['m0'] * 0.85 < params['m_c'] * 2 and np.amax(resultsDF['alpha_prof']) * 0.85 > params['a_c'] * 2:
        ## horizontal break only
        fig = plt.figure(constrained_layout=False, figsize=(10,8))
        # gs_kw = dict(width_ratios=4, height_ratios=4)
        gs = fig.add_gridspec(nrows=4,ncols=4) #, left=)
        axBL = fig.add_subplot(gs[:,:-1]) #, gridspec_kw=gs_kw)
        # axTL = fig.add_subplot(gs[0,:-1])
        # axTR = fig.add_subplot(gs[0,-1])
        axBR = fig.add_subplot(gs[:,-1])

        axBL.spines['right'].set_visible(False)
        axBL.xaxis.tick_bottom()

        axBR.spines['left'].set_visible(False)
        axBR.get_yaxis().set_ticks([])

        axBL.set_xlim(0., params['a_c'] * 2 * 0.85)
        axBL.set_ylim(0., params['m_c'] * 2 * 0.85)

        axBR.set_xlim(0.85 * resultsDF['alpha_prof'].max(), 1.05 * resultsDF['alpha_prof'].max())
        axBR.set_ylim(axBL.get_ylim())
        axBR.set_xticks([np.round(resultsDF['alpha_prof'].max(),1)])

        ### x axis break
        dX = .03
        dY = .03

        kwargs = dict(transform=axBL.transData, color='k', clip_on=False)
        axBL.plot((np.amax(axBL.get_xlim())-dX, np.amax(axBL.get_xlim())+dX), (np.amax(axBL.get_ylim())-dY, np.amax(axBL.get_ylim())+dY), **kwargs)        # top-left diagonal

        kwargs = dict(transform=axBL.transData, color='k', clip_on=False)
        axBL.plot((np.amax(axBL.get_xlim())-dX, np.amax(axBL.get_xlim())+dX), (np.amin(axBL.get_ylim())-dY, np.amin(axBL.get_ylim())+dY), **kwargs)

        kwargs = dict(transform=axBR.transData, color='k', clip_on=False)
        axBR.plot((np.amin(axBR.get_xlim())-dX, np.amin(axBR.get_xlim())+dX), (np.amin(axBR.get_ylim())-dY, np.amin(axBR.get_ylim())+dY), **kwargs)

        kwargs = dict(transform=axBR.transData, color='k', clip_on=False)
        axBR.plot((np.amin(axBR.get_xlim())-dX, np.amin(axBR.get_xlim())+dX), (np.amax(axBR.get_ylim())-dY, np.amax(axBR.get_ylim())+dY), **kwargs)

        axlist = [axBL, axBR]

        for ai, aa in enumerate(axlist):

            aa = subplot_PD(aa, params, colors)
            axlist[ai] = aa

        axBL.legend(loc='upper center', bbox_to_anchor=(0.7, 1.5), ncol=3)

    elif np.amax(resultsDF['m_profile']) / params['m0'] * 0.85 > params['m_c'] * 2 and np.amax(resultsDF['alpha_prof']) * 0.85 < params['a_c'] * 2:
        ## vertical break only
        fig = plt.figure(constrained_layout=False, figsize=(10,8))
        # gs_kw = dict(width_ratios=4, height_ratios=4)
        gs = fig.add_gridspec(nrows=4,ncols=4) #, left=)
        axBL = fig.add_subplot(gs[1:,:]) #, gridspec_kw=gs_kw)
        axTL = fig.add_subplot(gs[0,:])

        axTL.spines['bottom'].set_visible(False)
        axTL.xaxis.tick_top()
        axTL.tick_params(labeltop=False)
        axTL.get_xaxis().set_ticks([])

        axBL.spines['top'].set_visible(False)
        axBL.xaxis.tick_bottom()
        axBL.set_xlim(0., params['a_c'] * 2 * 0.85)
        axBL.set_ylim(0., params['m_c'] * 2 * 0.85)

        axTL.set_xlim(axBL.get_xlim())
        axTL.set_ylim(0.85 * resultsDF['m_profile'].max()/params['m0'], 1.05 * resultsDF['m_profile'].max()/params['m0'])

        ### y-axis break
        dT = .03 * np.diff(axBL.get_ylim())
        dB = .03 * np.diff(axBL.get_ylim())

        kwargs = dict(transform=axTL.transData, color='k', clip_on=False)
        axTL.plot((np.amin(axTL.get_xlim())-dT, np.amin(axTL.get_xlim())+dT), (np.amin(axTL.get_ylim())-dT, np.amin(axTL.get_ylim())+dT), **kwargs)        # top-left diagonal

        kwargs = dict(transform=axBL.transData, color='k', clip_on=False)
        axBL.plot((np.amin(axBL.get_xlim())-dB, np.amin(axBL.get_xlim())+dB), (np.amax(axBL.get_ylim())-dB, np.amax(axBL.get_ylim())+dB), **kwargs)

        kwargs = dict(transform=axBL.transData, color='k', clip_on=False)
        axBL.plot((np.amax(axBL.get_xlim())-dB, np.amax(axBL.get_xlim())+dB), (np.amax(axBL.get_ylim())-dB, np.amax(axBL.get_ylim())+dB), **kwargs)

        kwargs = dict(transform=axTL.transData, color='k', clip_on=False)
        axTL.plot((np.amax(axTL.get_xlim())-dT, np.amax(axTL.get_xlim())+dT), (np.amin(axTL.get_ylim())-dT, np.amin(axTL.get_ylim())+dT), **kwargs)

        axlist = [axBL, axTL]

        for ai, aa in enumerate(axlist):

            aa = subplot_PD(aa, params, colors)
            axlist[ai] = aa

        axTL.legend(loc='upper center', bbox_to_anchor=(0.7, 1.5), ncol=3)

    else:
        ## no break
        fig, ax = plt.subplots(1,1,figsize=(8, 8)) #, gridspec_kw={'width_ratios': [1, .35]})
        # axlist = [ax]
        ax = subplot_PD(ax, params, colors)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3)
        ax.set_xlabel('\u03b1')
        ax.set_ylabel('m/m0')

        axlist = [ax]


    

    # ax2[0].plot(a,m / params['m0'],color=mcolors['black'])
    # mred = np.concatenate((m[np.abs(np.diff(m, prepend=0))>0], m[np.abs(np.diff(m, append=0))>0]))
    # ared = np.concatenate((a[np.abs(np.diff(m, prepend=0))>0], a[np.abs(np.diff(m, append=0))>0]))
    # tred = np.concatenate((t[np.abs(np.diff(m, prepend=0))>0], t[np.abs(np.diff(m, append=0))>0]))

    # ax[1].scatter(tred, mred / params['m0'], color='r', s=120, zorder=50)
    # ax2[0].scatter(ared, mred / params['m0'], color='r', s=120, zorder=50)

    # priming_times, memory_times, stiffP, stiffA = summary_stats(a, x, m, t, params)
    # mech_stats = np.abs(stiffP - stiffA) / params['m0']

    # labels = ['${t_{prime}}/{t_{memory}}$', '$m_{diff}$']
    # barx = np.arange(len(labels))
    # barlist = ax2[1].bar(barx, [np.mean(priming_times / memory_times), mech_stats])
    # barlist[0].set_color('g')
    # barlist[1].set_color('k')
    # ax2[1].set_xticks(barx)
    # ax2[1].set_xticklabels(labels)

    # plt.tight_layout()

    return fig, axlist

def add_m_traj(ax, a, m, t, params):
        
        if params['color']:
            lcolor = params['color']
        else:
            lcolor = 'k'
        
        mred = np.concatenate((m[np.abs(np.diff(m, prepend=0))>0], m[np.abs(np.diff(m, append=0))>0]))
        # print(a[-10:])
        # print(mred)
        ared = np.concatenate((a[np.abs(np.diff(m, prepend=0))>0], a[np.abs(np.diff(m, append=0))>0]))
        ared[0] = params['a0']
        # ared[-1] = a[-1]
        tred = np.concatenate((t[np.abs(np.diff(m, prepend=0))>0], t[np.abs(np.diff(m, append=0))>0]))
        if ax is not None:
            # print('a_red')
            # print(ared)
            # print('m_red')
            # print(mred/params['m0'])
            ax.plot(a, m / params['m0'], color=lcolor, linewidth = 3.)
            ax.scatter(ared, mred / params['m0'], color=lcolor, s=120, zorder=50)
        
        return ax, ared, mred, tred

def plot_profile(fig, ax, resultsDF, params, colors=[mcolors['darkorange'], 
              mcolors['red'],
              mcolors['navy'],
              mcolors['darkviolet'],
              mcolors['deepskyblue'], 
              mcolors['springgreen']], plot_PD = True):

    def hours_to_days(time):
        return time/24.

    def update_days_axis(ax_hrs):
        x1, x2 = ax_hrs.get_xlim()
        ax_dys.set_xlim(hours_to_days(x1), hours_to_days(x2))
        ax_dys.figure.canvas.draw()
    
    t = resultsDF['t_space'].values
    m = resultsDF['m_profile'].values
    a = resultsDF['alpha_prof'].values
    x = resultsDF['x_prof'].values
    active_region = resultsDF['active_region'].values
    # print(active_region)
    
    dt = t[1] - t[0]
        
    count, cumsum, ids = rle(active_region)    
    
    ax_dys = ax[0].twiny()
    ax[0].callbacks.connect("xlim_changed", update_days_axis)
    ax_dys.set_xlabel('time (days)')
    ### x and alpha plot
    atwin = ax[0].twinx() #atwin for alpha
    # lc1 = ax[0].plot(t, x, linewidth=4., color='k', label='x', zorder=10);
    lc1 = ax[0].plot(t, x, linewidth=4., color=params['color'], label='x', zorder=10);
    lc2 = atwin.plot(t, a, linewidth=4., color = mcolors['green'], label='\u03b1', zorder=5);
    atwin.set_ylabel('\u03b1')
    atwin.yaxis.label.set_color(mcolors['green'])
    atwin.tick_params(axis='y', colors=mcolors['green'])

    ax[0].set_ylabel('x')
    ax[0].set_ylim([0, np.amax(x)*1.05])

    ax[0].yaxis.label.set_color(params['color'])
    ax[0].tick_params(axis='y', colors=params['color'])

    ax[0].set_xticks([])

    ## a_max
    # lc4 = ax[0].plot([t[0],t[-1]],[params['a_max'],params['a_max']],color = mcolors['firebrick'], linestyle='-.', linewidth=2., label='\u03b1$_{max}$', zorder=4)

    ## a_c
    lc3 = atwin.plot([t[0],t[-1]],[params['a_c'],params['a_c']],color = mcolors['green'], linestyle='-.', linewidth=2., label='\u03b1$_{c}$', zorder=4)
    ## x_c
    # ax[0].plot([t[0],t[-1]],[params['x_c'],params['x_c']],color = mcolors['black'], linestyle='-.', linewidth=2., label='x$_{c}$', zorder=4)

    ## legend
    lns = lc1+lc2+lc3 #+ lc4
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc=0)
    # ax[0].set_xlabel('time (hours)')
    # ax[0].yaxis.label.set_color('k')
    # ax[0].tick_params(axis='y', colors='k')
    
    for ci, cc in enumerate(cumsum):
        # print(ids)
        # print(cumsum)
        # print(dt)
        ax[0].plot([cc*dt, cc*dt],ax[0].get_ylim(),color=colors[ids[ci]-1], linestyle='-.', alpha = 0.8, zorder=0)
    
    ### m plot
    ax[1].set_ylim([np.amin(m / params['m0'])*0.8,np.amax(m / params['m0'])*1.1])
    # ax[1].plot(t, m/params['m0'], color = mcolors['black'], linewidth = 3., label='m')
    ax[1].plot(t, m/params['m0'], color = params['color'], linewidth = 3., label='m')
    ax[1].plot([t[0],t[-1]],[params['m_c'], params['m_c']], color = 'k', linestyle='-.', label='$m_{c}$')
    
    # /params['m0']
    # ax[1].plot([params['tau_F']*4,params['tau_F']*4],ax[1].get_ylim(), color = mcolors['red'], linestyle = '-.', linewidth = 2)

    ## adjusted for initial burn-in.
    # ax[1].plot([params['t_prime'][0]+24, params['t_prime'][0]+24],ax[1].get_ylim(), color = mcolors['red'], linestyle = '-.', linewidth = 2)
    ## this above is plotting on the top axis now instead of the bottom
    # ax[1].plot([params['t1max'][0], params['t1max'][0]],ax[1].get_ylim(), color = mcolors['red'], linestyle = '-.', linewidth = 2)

    ax[1].set_xlabel('time (hours)')
    ax[1].set_ylabel('m/m0')
    ax[1].legend(loc=1)
    
    print('timesteps per region:', end=" "); print(count); 
    print('regions:', end=" "); print(ids)
    params['time_to_AC'] = ((count[ids==1]) * params['dt'])[0]
    print(params['time_to_AC'])
    
    for ni, nn in enumerate(ids):
        rect = patches.Rectangle((cumsum[ni]*dt,ax[1].get_ylim()[0]), count[ni]*dt, np.diff(ax[1].get_ylim()), color=colors[nn-1], alpha=0.2)
        #### ragged list warnings happen below ####
        ### plt error.. 
        ax[1].add_patch(rect)
    
    ### phase diagram plot
    fig.tight_layout()

    if plot_PD:
        # print('here')
        fig2, axs = plot_full_PD(params, resultsDF, colors=[mcolors['darkorange'], mcolors['red'], mcolors['navy'], mcolors['navy'], mcolors['springgreen'], mcolors['springgreen']])
        for ai, aa in enumerate(axs):

            aa, ared, mred, tred = add_m_traj(aa, a, m, t, params)
            priming_times, memory_times, stiffP, stiffA = summary_stats(resultsDF, params, False)
    else:
        # print('there')
        _, ared, mred, tred = add_m_traj(None, a, m, t, params)
        fig2 = None
        axs = None

    ax[1].scatter(tred, mred / params['m0'], color=params['color'], s=120, zorder=50)    

    # plt.tight_layout()
    
    return params, fig, fig2, axs


def plot_memcorr(memDF, subcols, prettylabel, ax):
        
        def autolabel(rects):
        # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.0025,
                        '%.2f' % height,
                        ha='center', va='bottom')
 
        subDF = memDF[subcols]    
        mem_corr = subDF.corrwith(memDF['tm_over_tp'], method='spearman') # /memDF['prime_time']
        print(mem_corr)
        x_labels = [prettylabel[i] for i in mem_corr.index.values]
        barz = ax.bar(x_labels, np.abs(mem_corr.values), color=[mcolors['blue'] if i > 0 else mcolors['red'] for i in np.sign(mem_corr.values)])
        autolabel(barz)

        ax.set_ylim([0., np.amax(np.abs(mem_corr.values)*1.1)])
        
        plt.tight_layout()
        
        return mem_corr

def plot_groupedDF_heatmap(allDF, heat_cols, prettylabel, limits=[None, None]):

    # print(allDF.loc[allDF['prime_time'] > 52.316].loc[allDF['prime_time'] < 56.618].loc[allDF['tau_SG'] > 569].loc[allDF['tau_SG'] < 576])

    heatDF = allDF.groupby([pd.cut(allDF[heat_cols[0]], 20), pd.cut(allDF[heat_cols[1]], 20)]).agg('mean')
    ## nan values after this correspond to bins where there weren't any values in the original data.
    # heatDF['mem_time'] = heatDF['mem_time'] / heatDF['prime_time']

    heatDF['tm_over_tp'].loc[heatDF['tm_over_tp'] > 5.] = 5.

    heatDF = heatDF[heatDF.columns.difference(heat_cols)].dropna().reset_index()
    # drop empty bins and put the indexes into columns.
    
    # print(heatDF.loc[heatDF['mem_time'].isna()])
    heatDF['x_avg'] = heatDF[heat_cols[0]].apply(lambda x: x.mid)
    heatDF['y_avg'] = heatDF[heat_cols[1]].apply(lambda x: x.mid)
    # heatDF['mem_time'] = heatDF['mem_time'].values / heatDF['prime_time'].values

    # subheatDF = heatDF['mem_time'].unstack()

    if limits[0]:
        xrang = limits[0]
    else:
        xrang = [np.amin(heatDF['x_avg']), np.amax(heatDF['x_avg'])]

    if limits[1]:
        yrang = limits[1]
    else:
        yrang = [np.amin(heatDF['y_avg']), np.amax(heatDF['y_avg'])]

    fig = go.Figure(data=[go.Contour(z=heatDF['tm_over_tp'], x=heatDF['x_avg'], y=heatDF['y_avg'], colorscale='haline',
                                line_smoothing=0.85, 
                                contours=dict(
                                        start=0.,
                                        end=2.,
                                        size=0.5,),
                                contours_coloring='heatmap'
                                #cmin=np.amin(mem_z)*0.9, cmax=np.amax(mem_z)/6
                                )],
               )

    fig.update_layout(autosize=True, 
                    xaxis = dict(
                        title=dict(text=prettylabel[heat_cols[0]], standoff = 3),
                        titlefont=dict(family='Cambria', size=22),
                        range=xrang),
                    yaxis = dict(
                        title=dict(text=prettylabel[heat_cols[1]], standoff = 3),
                        titlefont=dict(family='Cambria', size=22),
                        range=yrang), 
                  width=500, height=500,
                  margin=dict(l=70, r=70, b=70, t=70),
                  font=dict(family='Cambria', size=16, color='#000000'),

                  )
    return fig

def plot_sample_dynamics(ax, t, a1, a2, params):

    dt = t[1] - t[0]        
    
    ax.plot(t, a1, linewidth=3., color=mcolors['green'], label='priming')
    ax.plot(t + np.amax(t), a2, linewidth=3.,color=mcolors['green'],label='memory')

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
                         color=mcolors['darkorange'], alpha=0.2)

    rect1 = patches.Rectangle((t[-1] - w1, ylims[0]), 
                             w1, ylims[1]-ylims[0], 
                             color=mcolors['red'], alpha=0.2)
    
    rect2 = patches.Rectangle((t[-1], ylims[0]), 
                             w2,ylims[1]-ylims[0], 
                             color=mcolors['springgreen'], alpha=0.2)
    
    rect3 = patches.Rectangle((w2 + t[-1], ylims[0]), 
                             t[-1] - w2 + xlims[1] - t[-1]*2, ylims[1]-ylims[0], 
                             color=mcolors['darkviolet'], alpha=0.2)
    
    ax.add_patch(rect0); ax.add_patch(rect1); ax.add_patch(rect2); ax.add_patch(rect3)

    ax.plot(xlims,[params['a_c'], params['a_c']], color = mcolors['green'], linestyle='-.', linewidth=2., label='\u03b1$_{c}$')

    ax.set_xlim(xlims); ax.set_ylim(ylims)

    return ax

if __name__ == '__main__':
    
    pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)

    direc = './stiff_results/'
    # fname = 'static_LHS_SG_SR_ng'
    fname = 'energy_dependent_LHS_n3'
    # fname2 = 'static_LHS_SG_SR_n4'

    allDF = pd.read_csv(direc + fname +'.csv')
    # allDF2 = pd.read_csv(direc + fname2 +'.csv')
    input_prof = np.load(direc + fname + '_inputs.npy', allow_pickle=True)
    # input_prof2 = np.load(direc + fname2 + '_inputs.npy', allow_pickle=True)

    # allDF = pd.concat((allDF, allDF2))

    allDF = allDF.reset_index(drop=True)
    print(allDF.head(1))
    print(len(allDF))
    # print(input_prof.shape)
    # print(input_prof)
    prettylabel = {
                        'prime_time':'$t_{prime}$', 
                        'tau_R0':'${\\tau}_{SGR}$', 
                        'TV0SG':'${V}_{SG}$',
                        'TV0SR':'${V}_{SR}$',
                        # 'm0':'${m}_{0}$',
                        # 'tau':'${\\tau}$',
                        'delta_prime':'${\Delta}_{prime}$',
                        'delta_mem':'${\Delta}_{mem}$', 
                        'delta_tot':'${\Delta}_{tot}$',
                        'n':'n'
                        }



    allDF['delta_prime'] = (allDF['prime_stiff'] - allDF['m_c'] * allDF['m0']) / allDF['m0']
    allDF['delta_mem'] = (allDF['m_c'] * allDF['m0'] - allDF['mem_stiff']) / allDF['m0']
    allDF['delta_tot'] = allDF['delta_prime'] + allDF['delta_mem']

    scaledDF = (allDF.select_dtypes(include=np.number) - allDF.select_dtypes(include=np.number).mean(axis=0)) / allDF.select_dtypes(include=np.number).std(axis=0)

    print(scaledDF.head())

    allDF['mem_time'].hist(bins=100)
    manual_regress = allDF.loc[np.abs(allDF['mem_time']/allDF['prime_time']-1) < 0.1]
    print(manual_regress)
    plt.show()
    sys.exit()

    memDF = allDF.loc[allDF['mem_time'] > 0.].loc[allDF['mem_time'].notna()]

    subcols = ['prime_time','tau_R0','TV0SG', 'TV0SR', 'delta_prime', 'delta_mem', 'delta_tot'] # 'm0','tau',
    fig, ax = plt.subplots(1,1,figsize=(6, 6))

    mem_corr = plot_memcorr(memDF, subcols, prettylabel, ax)
    mem_corr = mem_corr.iloc[mem_corr.abs().argsort()]

    print(mem_corr)

    heat_cols = mem_corr.index.values[-2:]
    print(heat_cols)
    heat_cols = ['tau_R0', 'delta_tot']

    # allDF = allDF.loc[allDF['mem_time'] > 0.].loc[allDF['mem_time'].notna()]

    

    plot_groupedDF_heatmap(allDF[subcols+['mem_time']], heat_cols, prettylabel)

    fig, axes = plt.subplots(1, len(manual_regress[subcols].columns), figsize=(32, 8))   

    fig, axes = plt.subplots(1, len(manual_regress[subcols].columns), figsize=(32, 8))
    for col, axis in zip(manual_regress[subcols].columns, axes.flatten()):
        print(col)
        manual_regress[subcols].hist(column = col, bins = 100, ax=axis)

    fig.tight_layout()

    print(manual_regress)

    

    plt.show()


