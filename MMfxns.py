#!/usr/bin/python

import math
import scipy
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import griddata
import scipy.ndimage as nd

import matplotlib.pyplot as plt
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
import plotly.graph_objects as go

from numpy.random import Generator, PCG64

from MMplotting import plot_PD_rates_v2 # plot_PD_rates,

# import warnings
# ## importing and doing this to ignore warning in plt add_patch
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)
mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

class noise:
    def __init__(self, mean=0., std = 1., mag = 0.01):
        self.mean = mean
        self.std = std
        self.mag = mag
        self.rg = Generator(PCG64())
    def draw(self):
        return self.mag * self.rg.normal(self.mean, self.std)

def f_m(m, params):
    if isinstance(params['type'], str):
        if params['type'] == 'stiff':
            return 1 - np.exp(-m/params['m0'])  + 0.4
        if params['type'] == 'soft':
            return 1 - np.exp(-params['m0']/m) + 0.3
        if params['type'] ==  'basic':
            return m/params['m0']
        if params['type'] == 'linear':
            return 0.3 * m/params['m0'] + .2
        if params['type'] == '-stiff':
            inv_og_tau = 1.
            return -(1 - np.exp(-m/params['m0'])) + inv_og_tau
    else:
        if isinstance(m, np.ndarray):
            return params['type'] * np.ones(len(m))
        else:
            return params['type']

def x_total(a, params):
    # fix x_total to some value x0.
    return params['xtt'] * (a - params['a0']) + params['x0']

def U(fm, m, x, a, params):
    km = fm(m, {'type':params['km'], 'm0':params['m0'], 'g':params['g']})
    kc = fm(m, {'type':params['kc'], 'm0':params['m0'], 'g':params['g']})
    if params['km'] == 'soft':
        mscale = params['m0']/m
    else:
        mscale = m/params['m0']
    return (
        (-km * x_total(a, params) - (a + mscale**params['g'] / (mscale**params['g']+1))) * x + x**2/2 * (km + kc) + (a + mscale**params['g'] / (mscale**params['g']+1)) * x * scipy.special.hyp2f1(1, 1/params['n'], 1+1/params['n'], -x**params['n']) 
    )

# def U(fm, m, x, a, params):
#   without x_ref

#     km = fm(m, {'type':params['km'], 'm0':params['m0'], 'g':params['g']})
#     kc = fm(m, {'type':params['kc'], 'm0':params['m0'], 'g':params['g']})
#     mscale = m/params['m0']
#     return (
#         (-km * x - (a + mscale**params['g'] / (mscale**params['g']+1))) * x + x**2/2 * (kc) + (a + mscale**params['g'] / (mscale**params['g']+1)) * x * scipy.special.hyp2f1(1, 1/params['n'], 1+1/params['n'], -x**params['n']) 
#         # + m * x * scipy.special.hyp2f1(1, 1/params['n'], 1+1/params['n'], -x**params['n'])
#     )

def U_old(fm, m, x, a, params):
    km = fm(m, {'type':params['km'], 'm0':params['m0'], 'g':params['g']})
    kc = fm(m, {'type':params['kc'], 'm0':params['m0'], 'g':params['g']})
    return (
        (-km * x_total(a, params) - a) * x + x**2/2 * (km + kc) + a * x * scipy.special.hyp2f1(1, 1/params['n'], 1+1/params['n'], -x**params['n']) 
        # + m * x * scipy.special.hyp2f1(1, 1/params['n'], 1+1/params['n'], -x**params['n'])
    )

def x_crit(n):
    return ((n-1)/(n+1))**(1/n)

def alpha_crit(m, params):
    n = params['n']
    if params['km'] == 'soft':
        mscale = params['m0']/m
    else:
        mscale = m/params['m0']
    km = f_m(m, {'type':params['km'], 'm0':params['m0'], 'g':params['g']})
    kc = f_m(m, {'type':params['kc'], 'm0':params['m0'], 'g':params['g']})
    return ( -(mscale**params['g']/(mscale**params['g'] +1)) + 4 * n * (km + kc))/((n-1)**((n-1)/n)*(n+1)**((n+1)/n) ) 

def m_crit_general(mc, params):
    # general to f(m)
    # mscale = m/params['m0']
    km = f_m(mc, {'type':params['km'], 'm0':params['m0'], 'g':params['g']})
    kc = f_m(mc, {'type':params['kc'], 'm0':params['m0'], 'g':params['g']})
    return km * x_total(alpha_crit(np.abs(mc), params), params) - params['x_c'] * (km + kc) + (alpha_crit(np.abs(mc), params) + (np.abs(mc)/params['m0'])**params['g']/((np.abs(mc)/params['m0'])**params['g'] +1)) * params['x_c']**params['n']/(params['x_c']**params['n']+1)
    # return (f_m(mc, params) - params['x_c']/params['tau'] + params['a_c'] * params['x_c']**params['n']/(params['x_c']**params['n']+1))

def x_equil(x, m, alpha, params): 
#     x = vs   
    # dynamic nuclear exit rate
    km = f_m(m, {'type':params['km'], 'm0':params['m0'], 'g':params['g']})
    kc = f_m(m, {'type':params['kc'], 'm0':params['m0'], 'g':params['g']})
    if params['km'] == 'soft':
        mscale = params['m0']/m
    else:
        mscale = m/params['m0']
    return km * x_total(alpha, params)  - x * (km + kc) + (alpha + mscale**params['g']/(mscale**params['g']+1)) * x**params['n']/(x**params['n']+1)

def collect_minima(U, m_space, x_space, a_space, params):
    
    U_data = np.zeros((len(m_space), len(x_space), len(a_space)))
    U_mins = np.zeros((U_data.shape[0], U_data.shape[-1]))
    barrier_heights = np.zeros((U_data.shape[0], U_data.shape[-1]))
    x_arr_max = np.zeros((U_data.shape[0], U_data.shape[-1]))
    
    xc_ind = np.where(np.abs(x_space - params['x_c']) == np.amin(np.abs(x_space-params['x_c'])))[0][0]

    gmin_overm = []; b1_overm = []; b2_overm = []; inf_overm = [];
    capture2minima = []; capture_mvals = []; capmax = [];
    for mi, mm in enumerate(m_space):
        gmin_coords = []; bi1_coords = []; bi2_coords = []; inf_coords = []
        for ai, aa in enumerate(a_space):
            for xi, xx in enumerate(x_space):
                U_data[mi, xi, ai] = U(f_m, mm, xx, aa, params)
                
            xargs = scipy.signal.argrelextrema(U_data[mi,:,ai], np.less)[0]      
            x_mins = x_space[xargs]

            barrier_heights[mi, ai] = U_data[mi, xc_ind, ai] - np.amin(U_data[mi, :, ai])

            if len(x_mins) > 1:
                bi1_coords.append([np.amin(x_mins), aa, U_data[mi, np.amin(xargs), ai]])
                bi2_coords.append([np.amax(x_mins), aa, U_data[mi, np.amax(xargs), ai]])
                U_mins[mi, ai] = len(x_mins)
                x_arr_max[mi, ai] = x_mins[np.argmin(U_data[mi,xargs,ai])]
            elif len(x_mins) > 0:
                gmin_coords.append([x_mins[U_data[mi, xargs, ai] == np.amin(U_data[mi, xargs, ai])][0], aa, np.amin(U_data[mi, xargs, ai])])
                inf_coords.append([x_mins[U_data[mi, xargs, ai] == np.amax(U_data[mi, xargs, ai])][0], aa, np.amax(U_data[mi, xargs, ai])])
                U_mins[mi, ai] = len(x_mins)
                x_arr_max[mi, ai] = x_mins[np.argmin(U_data[mi,xargs,ai])]
            else:
                x_mins = x_space[np.argmin(U_data[mi, :, ai])]
                gmin_coords.append([np.amin(x_mins), aa, U_data[mi, np.argmin(U_data[mi,:,ai]), ai]])
                U_mins[mi, ai] = 0.
                x_arr_max[mi, ai] = x_space[np.argmin(U_data[mi,:,ai])]
#                 gmin_coords.append([np.amin(x_mins), aa, U_data[mi, np.amin(xargs), ai]])
            

        gmin_overm.append(np.array(gmin_coords))
        b1_overm.append(np.array(bi1_coords))
        b2_overm.append(np.array(bi2_coords))
        inf_overm.append(np.array(inf_coords))
        
        if len(b1_overm[-1]) > 0:
            # get the minimum alpha where two minima appear for a given m (B1 and B2 are populated, second column)
            # verified that using b1 or b2 gives the same results as it should
            capmax.append(np.amax(b1_overm[-1][:,1]))
            capture2minima.append(np.amin(b1_overm[-1][:,1]))
            capture_mvals.append(mm)

#     print(capture2minima)
#     print(capture2test)
    return U_data, U_mins, x_arr_max, gmin_overm, b1_overm, b2_overm, inf_overm, capture2minima, capture_mvals, capmax, barrier_heights

def build_mprof(inarr, res):

    """
    takes in an input array with first column number of time units, second column stiffness, and constructs the m profile. res is dt
    (should it be m/m0?)
    """
    m = []
    # inarr[:,0] /= res
    for ri in np.arange(len(inarr)):

        m = m + [inarr[ri,1] for ss in np.arange(math.floor(inarr[ri,0]) / res)]

    # print(m)

    return np.array(m)

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

def calc_PD_rates(params, display = False):

    def smooth(y, box_pts):
        # print(y.shape)
        box = np.ones(box_pts)/box_pts
        # y_smooth = np.hstack((np.convolve(y[:,0], box, mode='same'), np.convolve(y[:,1], box, mode='same'))).squeeze()
        # print(y_smooth.shape)

        cumsum_vec = np.cumsum(np.insert(y, 0, 0, axis=0), axis=0)
        ma_vec = (cumsum_vec[box_pts:,:] - cumsum_vec[:-box_pts,:]) / box_pts
        # print(ma_vec.shape)
        # ma_vec = np.unique(ma_vec, axis=0)
        # print(ma_vec.shape)
        return ma_vec

    if 'x_max' not in params.keys():
        params['x_max'] = params['x0']

    a_space = np.linspace(0.01, params['a_max']+1, int(params['grid_resolution']))
    params['a_space'] = a_space
    x_space = np.linspace(0.01, np.amax([params['x_max']*1.5, 8.]), int(params['grid_resolution']))
    params['x_space'] = x_space
    
    U_data, U_mins, x_arr_max, gmin_overm, b1_overm, b2_overm, inf_overm, capture2minima, capture_mvals, capmax, barrier_heights = collect_minima(U, params['m_space'], x_space, a_space, params)

    if len(capture2minima) == 0 or len(capture_mvals) == 0 or len(capmax) == 0:
        print('m_c exit?', end = ''); print(params['m_c'])
        print(params)
        params['earlyexit'] = 1
        if display:
            choose_m = params['m0']
            choose_m_ind = np.where(np.abs(params['m_space'] - choose_m) == np.amin(np.abs(params['m_space']-choose_m)))[0][0]
            Uslice = U_data[choose_m_ind, :, :].squeeze()
            xplot = x_space
            aplot = a_space
            uplot = Uslice # -U2_data

            fig = go.Figure(data=[go.Surface(z=uplot.T, x=xplot, y=aplot, colorscale='blackbody',
                                            cmin=np.amin(uplot)*0.8, cmax=np.amax(uplot)/6
                                            )],
                                 layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
                           )

            # fig.update_traces(contours_z=dict(show=True, usecolormap=False,
            #                                   project_z=True, start=np.amin(uplot) * 0.9, 
            #                                   end=np.amax(uplot), color='black', size=0.09))
            fig.show()
        return [], [], [], []

    cs = plt.contour(a_space, params['m_space'] / params['m0'], x_arr_max, levels=[params['x_c']])
    # plt.show()
    
    x_cvals = np.vstack(cs.allsegs[0])
    # print(x_cvals)
    x_cvals = x_cvals[x_cvals[:,0] < np.amin(capture2minima),:]

    # gfig = plot_PD_rates(capture2minima, capmax, capture_mvals, x_cvals, params['m_space'], a_space, params)

    # first col is m values, second col is low minima (blue), third col is high minima (red),
    mtst = np.concatenate((np.array(capture_mvals)[:,None] / params['m0'], np.array(capture2minima)[:,None], np.array(capmax)[:,None]), axis=1)

    lowlines = np.concatenate((x_cvals,mtst[:,[1,0]]), axis=0)
    lowlines = lowlines[lowlines[:,0].argsort()]
    lowlines = smooth(lowlines, 5)
    # print(lowlines)

    boostgrid = np.linspace(np.amin(lowlines[:,0]),np.amax(lowlines[:,0]),int(params['grid_resolution']*3))
    boostrez = griddata(lowlines[:,0].squeeze(),lowlines[:,1].squeeze(),boostgrid,method='linear')
    newlowlines = np.concatenate((boostgrid[:,None], boostrez[:,None]), axis=1)
    # print(newlowlines)

    highlines = np.concatenate((x_cvals,mtst[:,[2,0]]), axis=0)
    highlines = highlines[highlines[:,0].argsort()]
    highlines = smooth(highlines, 5)

    boostgrid = np.linspace(np.amin(highlines[:,0]),np.amax(highlines[:,0]),int(params['grid_resolution']*3))
    boostrez = griddata(highlines[:,0].squeeze(),highlines[:,1].squeeze(),boostgrid,method='linear')
    newhighlines = np.concatenate((boostgrid[:,None], boostrez[:,None]), axis=1)

    gfig = plot_PD_rates_v2(newlowlines, newhighlines, params['m_space'], a_space, params)

    if display:
        gfig.show()

    # x_cvals is the dividing line between region 1 and region 2
    return mtst, x_cvals, newlowlines, newhighlines

def update_alpha(t_region, alpha, params):

    # mscale = m/params['m0']

    if params['dynamics'] == 'constant':
        # print(params['dynamics'])
        if t_region[0] in [2]:
            dalpha = 1/params['tau_SG'] * params['dt']

        elif t_region[0] in [1]:
            dalpha = -1/params['tau_F'] * params['dt']

        elif t_region[0] in [3]:
            dalpha = -1/params['tau_SR'] * params['dt']

    elif params['dynamics'] == 'exp_staticTS':
        # print(params['dynamics'])
        if t_region[0] in [2]:
            dalpha = alpha/params['tau_SG'] * params['dt']

        elif t_region[0] in [1]:
            dalpha = -alpha/params['tau_F'] * params['dt']

        elif t_region[0] in [3]:
            dalpha = -alpha/params['tau_SR'] * params['dt']

    elif params['dynamics'] == 'exp_dynamicTS':
        # print(params['dynamics'])
        if t_region[0] in [2]:
            dalpha = alpha/params['tau_SG'] * params['dt']

        elif t_region[0] in [1]:
            dalpha = -alpha/params['tau_F'] * params['dt']

        elif t_region[0] in [3]:
            dalpha = -alpha/params['tau_SR'] * params['dt']
            
    elif params['dynamics'] == 'updated_exp':
        if t_region[0] in [2]:
            dalpha = alpha/params['tau_SG'] * params['dt']

        elif t_region[0] in [1]:
            dalpha = -(alpha - params['a0'])/params['tau_F'] * params['dt']

        elif t_region[0] in [3]:
            dalpha = -alpha/params['tau_SR'] * params['dt']

    elif params['dynamics'] == 'updated_exp_staticTS':
        if t_region[0] in [2]:
            dalpha = alpha/params['tau_SG'] * params['dt']

        elif t_region[0] in [1]:
            dalpha = -(alpha - params['a0'])/params['tau_F'] * params['dt']

        elif t_region[0] in [3]:
            dalpha = -alpha/params['tau_SR'] * params['dt']     
    else:
        dalpha = 0.    

    return dalpha + params['eps'].draw()

def update_tau(params, x_prof, m_prof):

    if params['km'] == 'soft':
        mscale = m_prof / params['m0']
    else:
        mscale = params['m0'] / m_prof

    params['tau_SR'] = params['tau_R0'] * mscale * np.exp(x_prof/params['TV0SR'])
    params['tau_SG'] = params['tau_R0'] * mscale * np.exp(x_prof/params['TV0SG'])

    return params

def integrate_profile(m_profile, t_space, params, resultsDF):

    params['earlyexit'] = 0
    errorflag = False; maxflag = False;

    if 'eps' not in params.keys():
        print('adding noise')
        params['eps'] = (0., 1., 0.)
    params['eps'] = noise(params['eps'][0], params['eps'][1], params['eps'][2])

    kmdict = {'type':params['km'], 'm0':params['m0']}
    kcdict = {'type':params['kc'], 'm0':params['m0']}

    dt = params['dt']
    x_prof = np.zeros(len(t_space))
    alpha_prof = np.zeros(len(t_space))
    active_region = []; t_primelist = []; tsr_list = []; deltaVlist = []; tsg_list = []
    t_region = [0,0] # (region, t)
    
    mtst2, x_cvals, newlowlines, newhighlines = calc_PD_rates(params, display = False)

    if np.isnan(newlowlines).any() or np.isnan(newhighlines).any() or params['earlyexit'] == 1:
        print('see ya')
        params['earlyexit'] = 1
        return resultsDF, params

    if np.amax(m_profile / params['m0']) < np.amax(newhighlines[:,1]) or np.amin(m_profile / params['m0']) > np.amax(newlowlines[:,1]):
        print('bye')
        params['earlyexit'] = 1
        return resultsDF, params

    # plt.plot(newlowlines[:,0], newlowlines[:,1])
    # plt.plot(newhighlines[:,0], newhighlines[:,1])
    # plt.show()
    params['a_max'] = np.amax(newhighlines[:,0])
    # print('amax2')
    # print(params['a_max'])
    a_term = np.amax(mtst2[:,1])

    for ti, tt in enumerate(t_space):

        if ti == 0:
            alpha_prof[ti] = params['a0']
            axc_low = newlowlines[np.abs(newlowlines[:,0] - alpha_prof[ti]) == np.amin(np.abs(newlowlines[:,0] - alpha_prof[ti]))].squeeze()
            axc_high = newhighlines[np.abs(newhighlines[:,0] - alpha_prof[ti]) == np.amin(np.abs(newhighlines[:,0] - alpha_prof[ti]))].squeeze()
            # ensure that we start in region 1
            if hasattr(axc_low[1], "__len__"):
                print('bye')
                params['earlyexit'] = 1
                print(axc_low)
                print('----')
                print(axc_high)
                print('----')
                print(params)
                return resultsDF, params

            if m_profile[ti] / params['m0'] >= axc_low[1]:
                alpha_prof[ti] = axc_low[0] / 2
                params['a0'] = alpha_prof[ti]

            x_prof[ti] = scipy.optimize.fsolve(x_equil, 1., args=(m_profile[ti], params['a0'], params), xtol=1e-10)[0]
            continue;

        params['x_max'] = np.amax(x_prof)
        xUd = np.linspace(0, params['x_max']*2.5, int(1e3))

        axc_low = newlowlines[np.abs(newlowlines[:,0] - alpha_prof[ti-1]) == np.amin(np.abs(newlowlines[:,0] - alpha_prof[ti-1]))].squeeze()
        axc_high = newhighlines[np.abs(newhighlines[:,0] - alpha_prof[ti-1]) == np.amin(np.abs(newhighlines[:,0] - alpha_prof[ti-1]))].squeeze()

        try:
            if m_profile[ti] / params['m0'] > axc_high[1]:
                pass
        except:

            if not errorflag:
                print('pd error')
                print(params)            
            
            errorflag = True
            axc_high = np.unique(axc_high, axis=0).squeeze()

        if params['m_c']*params['m0'] > 50 or params['m_c'] < 0:
            if not errorflag:
                print('m_c error')
                print(params) 
            errorflag = True

        if alpha_prof[ti-1] >= params['a_max']:
            if not maxflag:
                print('max')
                print(params)
            maxflag = True
            # permanent memory;
            alpha_prof[ti] = params['a_max']
            x_prof[ti] = scipy.optimize.fsolve(x_equil, x_prof[ti-1], args=(m_profile[ti], alpha_prof[ti], params), xtol=1e-10)[0]
            # if m_profile[ti] / params['m0'] > axc_high[1]:
            active_region.append(2)
            # else:
            #     active_region.append(3)

        else:
           
            if m_profile[ti] / params['m0'] < axc_low[1]:
            # region 1
                active_region.append(1)

                if t_region[0] != 1:
                    t_region[0] = 1
                    t_region[1] = dt
                else:
                    t_region[1] += dt
                
                dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                alpha_prof[ti] = alpha_prof[ti-1] + dalph

            elif m_profile[ti] / params['m0'] > axc_high[1]:
            # region 2
                active_region.append(2)
                
                if t_region[0] != 2:
                    if t_region[0] not in [3]:
                        t_region[0] = 2
                        t_region[1] = dt
                        a_enter = alpha_prof[ti-1]
                    else:
                        t_region[0] = 2
                        t_region[1] += dt
                        a_enter = alpha_prof[ti-1]
                else:
                    t_region[1] += dt
                    
                dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                alpha_prof[ti] = alpha_prof[ti-1] + dalph

            else:
             # region 3
                active_region.append(3)
                
                if t_region[0] != 3:
                    if t_region[0] not in [3]:
                        # print('enter 6')
                        # print('tau_SG start %e' % (params['tau_SG']))
                        if params['dynamics'] == 'updated_exp':
                            params['tau_SR'] = params['tau_SG']
                        # print('tau_SR start %e' % (params['tau_SR']))
                        t_region[0] = 3
                        t_region[1] = dt
                        a_enter = alpha_prof[ti-1]
                    else:
                        t_region[0] = 3
                        t_region[1] += dt
                
                dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                alpha_prof[ti] = alpha_prof[ti-1] + dalph

            x_prof[ti] = scipy.optimize.fsolve(x_equil, x_prof[ti-1], args=(m_profile[ti], alpha_prof[ti], params), xtol=1e-10)[0]          

        if t_region[0] in [2,3] and params['dynamics'] in ['updated_exp', 'exp_dynamicTS']:
            # print('tau')
            params = update_tau(params, x_prof[ti], m_profile[ti])

        if t_region[0] == 3:

            # # finding the barrier height.
            # # energy slice over x at alpha, m
            U_data = U(f_m, m_profile[ti], xUd, alpha_prof[ti], params)
            x_args = find_peaks(-np.abs(np.diff(U_data)))[0]
            deltaV = np.amax(U_data[x_args]) - np.amin(U_data[x_args])
            if deltaV > 0:
                deltaVlist.append(deltaV)
            else:
                deltaVlist.append(np.NaN)

            tsg_list.append(params['tau_SG'])
            tsr_list.append(params['tau_SR'])

        else:

            deltaVlist.append(np.NaN)
            tsg_list.append(params['tau_SG'])
            tsr_list.append(params['tau_SR'])

    # print(x_prof)

    active_region.insert(0, active_region[0])
    deltaVlist.insert(0, deltaVlist[0])
    tsg_list.insert(0, tsg_list[0])
    tsr_list.insert(0, tsr_list[0])

    resultsDF['m_profile'] = m_profile
    resultsDF['t_space'] = t_space
    resultsDF['x_prof'] = x_prof
    resultsDF['alpha_prof'] = alpha_prof
    resultsDF['active_region'] = active_region
    resultsDF['deltaV'] = deltaVlist
    resultsDF['tSG'] = tsg_list
    resultsDF['tSR'] = tsr_list

    return resultsDF, params

def summary_stats_v2(resultsDF, params, verbose=True):

    t = resultsDF['t_space'].values
    m = resultsDF['m_profile'].values
    alpha = resultsDF['alpha_prof'].values
    x = resultsDF['x_prof'].values
    active_region = resultsDF['active_region'].values

    if params['km'] == 'stiff':
        priming_mask = (m / params['m0'] > np.mean(np.unique(m/params['m0']))) & (active_region == 2)
        memory_mask = (m / params['m0'] < np.mean(np.unique(m/params['m0']))) & (active_region == 3)
        perm_mask = (m / params['m0'] < np.mean(np.unique(m/params['m0']))) & (active_region == 2)
    elif params['km'] == 'soft':
        priming_mask = (m / params['m0'] < np.mean(np.unique(m/params['m0']))) & (active_region == 2)
    
    priming_time = np.sum(priming_mask) * params['dt']
    memory_time = np.sum(memory_mask) * params['dt']
    perm_time = np.sum(perm_mask) * params['dt']

    if perm_mask[-1]: # perm_time > 0:
        memory_time = -1.

    pstiff = np.amax(m)
    mstiff = np.amin(m)

    counts, cumsum, ids = rle(active_region) 

    if verbose:
        print('counts', end=" "); print(counts); print(np.sum(counts))
        print('ids', end=" "); print(ids)
        print('priming times', end=" "); print(priming_time)
        print('memory times', end=" "); print(memory_time)        

    return np.array(priming_time), np.array(memory_time), np.array(pstiff), np.array(mstiff)


def run_profile(e_func, inputs, params, resultsDF):


    def early_exit(resultsDF, params, m_profile, t_space):
        print('out early')
        resultsDF['m_profile'] = m_profile
        resultsDF['t_space'] = t_space
        resultsDF['x_prof'] = np.ones(len(t_space))
        resultsDF['alpha_prof'] = np.ones(len(t_space))
        resultsDF['active_region'] = np.ones(len(t_space), dtype=np.int32)
        resultsDF['deltaV'] = np.zeros(len(t_space))
        resultsDF['tSG'] = np.zeros(len(t_space))
        params['t_prime'] = [0.]
        params['t1max'] = [0.]
        return resultsDF, params, np.array([0]), np.array([0]), np.amax(m_profile), np.amin(m_profile)

    x_c = x_crit(params['n']); params['x_c'] = x_c;
    
    m_c = scipy.optimize.fsolve(m_crit_general, 0.5, args=(params), xtol=1e-10)[0] / params['m0']
    params['m_c'] = m_c
    
    if inputs[0,1] < 0:
        ## flag for random number for stiffness. negative number means base the stiffness profiles on m_c
        # pstiff
        inputs[1,1] = m_c * params['m0'] + (-inputs[1,1])*10

        # mstiff
        inputs[0,1] = m_c * params['m0'] - (-inputs[0,1]*m_c*params['m0'] * 0.9) 
        inputs[2,1] = m_c * params['m0'] - (-inputs[2,1]*m_c*params['m0'] * 0.9) 

    m_profile = build_mprof(inputs, params['time_resolution'])

    m_space = np.linspace(np.amin([0.1, np.amin(m_profile)]), np.amax(m_profile)*1., int(params['grid_resolution']))
    choose_mc_ind = np.where(np.abs(m_space - m_c*params['m0']) == np.amin(np.abs(m_space-m_c*params['m0'])))[0][0]
    a_c = alpha_crit(m_space, params)
    params['a_c'] = a_c;
    params['m_space'] = m_space #         print(np.unique(m_profile))    print(len(m_profile))
    # print(len(m_profile))
    # m profile is in absolute units
    # hours
    print('--------')
    print('a_c = %f, x_c = %f, m_c in absolute = %f' % (a_c[choose_mc_ind], x_c, m_c * params['m0']))

    # mtst2, x_cvals = calc_PD_rates(params)

    t_space = np.linspace(0, np.sum(inputs[:,0]), len(m_profile))
    params['dt'] = t_space[1]-t_space[0]

    print(params['dynamics'])
    # print(params['m_c'])
    # print(params['a_c'])

    # if np.amin(m_profile)/params['m0'] >= params['m_c']: #/params['m0']:
    #     early_exit(resultsDF, params, m_profile, t_space)

    resultsDF, params = e_func(m_profile, t_space, params, resultsDF)

    # print(params['a_max'])

    if 'earlyexit' in params.keys():
        if params['earlyexit'] > 0:
            early_exit(resultsDF, params, m_profile, t_space)

    priming_times, memory_times, stiffP, stiffA = summary_stats_v2(resultsDF, params)

    return resultsDF, params, priming_times, memory_times, stiffP, stiffA