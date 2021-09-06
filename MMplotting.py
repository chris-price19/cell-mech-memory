#!/usr/bin/python

"""

Plotting functions for the main figures

"""

import math
import numpy as np
import scipy
from scipy.signal import argrelextrema
# from scipy.interpolate import griddata
# import scipy.ndimage as nd

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

# from MMfxns import x_equil, rle

cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)
mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

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

def plot_profile(fig, ax, resultsDF, params, x_solve, colors=[mcolors['darkorange'], 
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
    m_space = params['m_space']
    active_region = resultsDF['active_region'].values
    # print(active_region)

    mc_ind = np.where(np.abs(m_space/ params['m0'] - params['m_c']) == np.amin(np.abs(m_space/ params['m0'] - params['m_c'])))[0][0]
    
    x_baseline = np.zeros(len(x))
    for mi, mm in enumerate(m):
        if mi > 0:
            x_baseline[mi] = scipy.optimize.fsolve(x_solve, x_baseline[mi-1], args=(mm, params['a0'], params), xtol=1e-10)[0]
        else:
            x_baseline[mi] = scipy.optimize.fsolve(x_solve, 1., args=(mm, params['a0'], params), xtol=1e-10)[0]

    dt = t[1] - t[0]
        
    count, cumsum, ids = rle(active_region)    
    
    ax_dys = ax[0].twiny()
    ax[0].callbacks.connect("xlim_changed", update_days_axis)
    ax_dys.set_xlabel('time (days)')
    ### x and alpha plot
    atwin = ax[1].twinx() #atwin for alpha
    # lc1 = ax[1].plot(t, x, linewidth=4., color='k', label='x', zorder=10);
    lc1 = ax[1].plot(t, x, linewidth=4., color=params['color'], label='x', zorder=10);
    lc4 = ax[1].plot(t, x_baseline, linewidth=2., color=params['color'], linestyle = '-.', label='$x_{ref}$', zorder=10);
    lc2 = atwin.plot(t, a, linewidth=4., color = mcolors['green'], label='\u03b1', zorder=5);
    atwin.set_ylabel('\u03b1')
    atwin.yaxis.label.set_color(mcolors['green'])
    atwin.tick_params(axis='y', colors=mcolors['green'])
    atwin.set_ylim([0,np.amax(a)*1.05])

    ax[1].set_ylabel('x')
    ax[1].set_ylim([0, np.amax(x)*1.05])

    ax[1].yaxis.label.set_color(params['color'])
    ax[1].tick_params(axis='y', colors=params['color'])

    ax[0].set_xticks([])

    ## a_max
    # lc4 = ax[1].plot([t[0],t[-1]],[params['a_max'],params['a_max']],color = mcolors['firebrick'], linestyle='-.', linewidth=2., label='\u03b1$_{max}$', zorder=4)

    ## a_c
    # lc3 = atwin.plot([t[0],t[-1]],[params['a_c'][mc_ind],params['a_c'][mc_ind]],color = mcolors['green'], linestyle='-.', linewidth=2., label='\u03b1$_{c}$', zorder=4)
    ## x_c
    # lc3 = ax[1].plot([t[0],100],[params['x_c'],params['x_c']],color = mcolors['lime'], linestyle='--', linewidth=2., label='x$_{c}$', zorder=4)


    ## legend
    lns = lc1+lc2 +lc4 #+ lc4
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc='upper right')
    # ax[1].set_xlabel('time (hours)')
    # ax[1].yaxis.label.set_color('k')
    # ax[1].tick_params(axis='y', colors='k')
    
    for ci, cc in enumerate(cumsum):
        # print(ids)
        # print(cumsum)
        # print(dt)
        if ci == 0:
            continue
        ax[1].plot([cc*dt, cc*dt],ax[1].get_ylim(),color=colors[ids[ci]-1], linestyle='--', linewidth=1.5, alpha = 0.6, zorder=0)
    
    ### m plot
    ax[0].set_ylim([np.amin(m / params['m0'])*0.8,np.amax(m / params['m0'])*1.1])
    # ax[0].plot(t, m/params['m0'], color = mcolors['black'], linewidth = 3., label='m')
    ax[0].plot(t, m/params['m0'], color = params['color'], linewidth = 3., label='m')
    # ax[0].plot([t[0],t[-1]],[params['m_c'], params['m_c']], color = 'k', linestyle='-.', label='$m_{c}$')
    
    # /params['m0']
    # ax[0].plot([params['tau_F']*4,params['tau_F']*4],ax[0].get_ylim(), color = mcolors['red'], linestyle = '-.', linewidth = 2)

    ## adjusted for initial burn-in.
    # ax[0].plot([params['t_prime'][0]+24, params['t_prime'][0]+24],ax[0].get_ylim(), color = mcolors['red'], linestyle = '-.', linewidth = 2)
    ## this above is plotting on the top axis now instead of the bottom
    # ax[0].plot([params['t1max'][0], params['t1max'][0]],ax[0].get_ylim(), color = mcolors['red'], linestyle = '-.', linewidth = 2)

    ax[1].set_xlabel('time (hours)')
    ax[0].set_ylabel('m/m$_{0}$')
    ax[0].legend(loc=1)


    
    print('timesteps per region:', end=" "); print(count); 
    print('regions:', end=" "); print(ids)
    params['time_to_AC'] = ((count[ids==1]) * params['dt'])[0]
    print(params['time_to_AC'])
    print(colors)
    for ni, nn in enumerate(ids):
        rect = patches.Rectangle((cumsum[ni]*dt,ax[0].get_ylim()[0]), count[ni]*dt, np.diff(ax[0].get_ylim()), color=colors[nn-1], alpha=.8)
        #### ragged list warnings happen below ####
        ### plt error.. 
        ax[0].add_patch(rect)
    
    # ### phase diagram plot
    # fig.tight_layout()

    # if plot_PD:
    #     # print('here')
    #     fig2, axs = plot_full_PD(params, resultsDF, colors=[mcolors['darkorange'], mcolors['red'], mcolors['navy'], mcolors['navy'], mcolors['springgreen'], mcolors['springgreen']])
    #     for ai, aa in enumerate(axs):

    #         aa, ared, mred, tred = add_m_traj(aa, a, m, t, params)
    #         priming_times, memory_times, stiffP, stiffA = summary_stats(resultsDF, params, False)
    # else:
    #     # print('there')
    #     _, ared, mred, tred = add_m_traj(None, a, m, t, params)
    #     fig2 = None
    #     axs = None

    # ax[0].scatter(tred, mred / params['m0'], color=params['color'], s=120, zorder=50)    

    # plt.tight_layout()
    
    return params, fig, ax # fig2,
    

# def plot_PD_rates(capture2minima, capmax, capture_mvals, x_cvals, m_space, a_space, params):

#     a_c = np.array(params['a_c'])
#     m_c = params['m_c']
#     mc_ind = np.where(np.abs(m_space/ params['m0'] - m_c) == np.amin(np.abs(m_space/ params['m0']-m_c)))[0][0]

#     gfig = go.Figure().update_layout(
#             template="simple_white",
#             width=600, height=600,
#             xaxis = dict(range=[0., 4.], mirror=True, showline=True),
#             yaxis = dict(range=[np.amin(capture_mvals), 1.25], mirror=False, showline=True),
#             # yaxis = dict(range=[0., 1.5], mirror=True, showline=True),
#             font=dict(
#                 # family="Courier New, monospace",
#                 size=18,
#                 # color="RebeccaPurple"
#                 ),
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(0,0,0,0)',
#             showlegend=False
#             )

#     xlims = gfig.layout.xaxis.range; ylims = gfig.layout.yaxis.range
#     capture_mvals = np.array(capture_mvals)


#     #### fills

#     ## bottom
#     xx = np.concatenate((capture2minima, capmax, )) # np.array([xlims[1], np.amin(capture2minima)])
#     yy = np.concatenate((capture_mvals, capture_mvals,)) / params['m0'] #  np.array([ylims[0], ylims[0]])

#     gfig.add_trace(go.Scatter(x=np.append(capture2minima, [a_c[mc_ind]]), 
#                               y=np.append(np.array(capture_mvals)/params['m0'], [m_c]),
#                         mode='none',  fill='tozerox', fillcolor='rgba(245, 121, 58, 1)'
#                         )
#                   )
#     gfig.add_trace(go.Scatter(x=capmax, 
#                              y=capture_mvals / params['m0'],
#                         mode='none', fill='tonextx', fillcolor='rgba(15, 32, 128, 1)'
#                              )
#                   )
#     ## top
#     # xx = [np.amax(a_c), xlims[1], xlims[1], a_c[mc_ind]]
#     # xx = np.concatenate((
#     #     a_c[m_space/params['m0']>m_c], [xlims[1], xlims[1]], capmax
#     # ))

#     # yy = [m_space[-1]/params['m0'], m_space[-1]/params['m0'], np.amin(capmax)/params['m0'], m_c]
#     # yy = np.concatenate((
#     #     m_space/params['m0'], [m_space[-1]/params['m0'], np.amin(capture_mvals)/params['m0']], capture_mvals
#     # ))
#     # gfig.add_trace(go.Scatter(
#     #                     x=xx,
#     #                     y=yy,
#     #                     mode='none', fill='toself', fillcolor='rgba(255, 0, 0, 0.3)'
#     #                 )
#     # #                                      line=dict(color='green', width=4))
#     #               )

#     # gfig.add_trace(go.Scatter(
#     #                     x=np.concatenate((a_c[m_space/params['m0']>m_c], np.array([a_space[0]]))),
#     #                     y=np.concatenate((m_space[m_space/params['m0']>m_c], np.array([m_space[-1]]))) / params['m0'],
#     #                     mode='none', fill='tozerox', fillcolor='rgba(148, 0,211, 0.3)'
#     #                 )
#     # #                                      line=dict(color='green', width=4))
#     #               )


#     #### lines
#     # gfig.add_trace(go.Scatter(
#     #                     x=a_c[a_c >= a_c[mc_ind]],
#     #                     y=m_space[a_c >= a_c[mc_ind]] / params['m0'],
#     #                     mode='lines', line=dict(color='green', width=4)
#     #                     )              
#     #               )

#     gfig.add_trace(go.Scatter(
#                         x=x_cvals[x_cvals[:,0] < np.amin(capture2minima),0],
#                         y=x_cvals[x_cvals[:,0] < np.amin(capture2minima),1],
#                         mode='lines', line=dict(color='rgba(78, 247, 75, 1)', width=6), fill='tozerox', fillcolor='rgba(245, 121, 58, 1)'

#                         )              
#                   )
#     # gfig.add_trace(go.Scatter(
#     #                     x=a_c[a_c <= a_c[mc_ind]],
#     #                     y=m_space[a_c <= a_c[mc_ind]] / params['m0'],
#     #                     mode='lines', line=dict(color='green', width=4, dash='dot')
#     #                     )              
#     #               )

#     gfig.add_trace(go.Scatter(x=np.append(capture2minima, [a_c[mc_ind]]),
#                               y=np.append(np.array(capture_mvals), [m_space[mc_ind]]) / params['m0'],
#                         mode='lines',line=dict(color='rgba(78, 247, 75, 1)', width=6),
#                              )
#                   )

#     # gfig.add_trace(go.Scatter(x=[np.amin(a_space), np.amax(a_space)], y=[m_c, m_c], # a_c[mc_ind]
#     #                     mode='lines', line=dict(color='red', width=4, dash='dot'),
#     #                          )
#     #               )

#     gfig.add_trace(go.Scatter(x=np.append(capmax, a_c[mc_ind]), 
#                              y=np.append(np.array(capture_mvals), m_space[mc_ind]) / params['m0'],
#                         mode='lines', line=dict(color='rgba(78, 247, 75, 1)', width=6),
#                              )
#                   )

#     return gfig


def plot_PD_rates_v2(lowlines, highlines, m_space, a_space, params):

    a_c = np.array(params['a_c'])
    m_c = params['m_c']
    # mc_ind = np.where(np.abs(m_space/ params['m0'] - m_c) == np.amin(np.abs(m_space/ params['m0']-m_c)))[0][0]

    gfig = go.Figure().update_layout(
            template="simple_white",
            width=600, height=600,
            xaxis = dict(range=[0., 4.], mirror=True, showline=True),
            yaxis = dict(range=[np.amin(lowlines[:,1]), 1.25], mirror=False, showline=True),
            # yaxis = dict(range=[0., 1.5], mirror=True, showline=True),
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
                ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
            )

    xlims = gfig.layout.xaxis.range; ylims = gfig.layout.yaxis.range
    # capture_mvals = np.array(capture_mvals)


    #### fills

    ## bottom
    # xx = np.concatenate((capture2minima, capmax, )) # np.array([xlims[1], np.amin(capture2minima)])
    # yy = np.concatenate((capture_mvals, capture_mvals,)) / params['m0'] #  np.array([ylims[0], ylims[0]])

    gfig.add_trace(go.Scatter(x=lowlines[:,0], 
                              y=lowlines[:,1],
                        mode='none',  fill='tozerox', fillcolor='rgba(245, 121, 58, 1)'
                        )
                  )
    gfig.add_trace(go.Scatter(x=highlines[:,0], 
                             y=highlines[:,1],
                        mode='none', fill='tonextx', fillcolor='rgba(15, 32, 128, 1)'
                             )
                  )


    # gfig.add_trace(go.Scatter(
    #                     x=x_cvals[x_cvals[:,0] < np.amin(capture2minima),0],
    #                     y=x_cvals[x_cvals[:,0] < np.amin(capture2minima),1],
    #                     mode='lines', line=dict(color='rgba(78, 247, 75, 1)', width=6), fill='tozerox', fillcolor='rgba(245, 121, 58, 1)'

    #                     )              
    #               )

    gfig.add_trace(go.Scatter(x=lowlines[:,0],
                              y=lowlines[:,1],
                        mode='lines',line=dict(color='rgba(78, 247, 75, 1)', width=6),
                             )
                  )


    gfig.add_trace(go.Scatter(x=highlines[:,0], 
                             y=highlines[:,1],
                        mode='lines', line=dict(color='rgba(78, 247, 75, 1)', width=6),
                             )
                  )

    return gfig


def plot_PD_rates_soft(lowlines, highlines, m_space, a_space, params):

    a_c = np.array(params['a_c'])
    m_c = params['m_c']
    # mc_ind = np.where(np.abs(m_space/ params['m0'] - m_c) == np.amin(np.abs(m_space/ params['m0']-m_c)))[0][0]

    gfig = go.Figure().update_layout(
            template="simple_white",
            width=600, height=600,
            xaxis = dict(range=[0., 4.], mirror=True, showline=True),
            yaxis = dict(range=[0., 6.], mirror=False, showline=True),
            # yaxis = dict(range=[0., 1.5], mirror=True, showline=True),
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
                ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
            )

    xlims = gfig.layout.xaxis.range; ylims = gfig.layout.yaxis.range
    # capture_mvals = np.array(capture_mvals)


    #### fills

    ## bottom
    # xx = np.concatenate((capture2minima, capmax, )) # np.array([xlims[1], np.amin(capture2minima)])
    # yy = np.concatenate((capture_mvals, capture_mvals,)) / params['m0'] #  np.array([ylims[0], ylims[0]])

    gfig.add_trace(go.Scatter(x=lowlines[:,0], 
                              y=lowlines[:,1],
                        mode='none',  fill='tozerox', fillcolor='rgba(245, 121, 58, 1)'
                        )
                  )
    gfig.add_trace(go.Scatter(x=highlines[:,0], 
                             y=highlines[:,1],
                        mode='none', fill='tonextx', fillcolor='rgba(15, 32, 128, 1)'
                             )
                  )


    # gfig.add_trace(go.Scatter(
    #                     x=x_cvals[x_cvals[:,0] < np.amin(capture2minima),0],
    #                     y=x_cvals[x_cvals[:,0] < np.amin(capture2minima),1],
    #                     mode='lines', line=dict(color='rgba(78, 247, 75, 1)', width=6), fill='tozerox', fillcolor='rgba(245, 121, 58, 1)'

    #                     )              
    #               )

    gfig.add_trace(go.Scatter(x=lowlines[:,0],
                              y=lowlines[:,1],
                        mode='lines',line=dict(color='rgba(78, 247, 75, 1)', width=6),
                             )
                  )


    gfig.add_trace(go.Scatter(x=highlines[:,0], 
                             y=highlines[:,1],
                        mode='lines', line=dict(color='rgba(78, 247, 75, 1)', width=6),
                             )
                  )

    return gfig


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


