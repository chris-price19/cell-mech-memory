#!/usr/bin/python

import math
import numpy as np
import scipy
from scipy.signal import argrelextrema, find_peaks
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

from numpy.random import Generator, PCG64
# import warnings
# ## importing and doing this to ignore warning in plt add_patch
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

cwd = os.getcwd()
sns.set(style="ticks", font_scale=1.5)
mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)

class noise:
    def __init__(self, mean=0., std = 1., mag = 0.1):
        self.mean = mean
        self.std = std
        self.mag = mag
        self.rg = Generator(PCG64())
    def draw(self):
        return self.mag * self.rg.normal(self.mean, self.std)

def f_m(m, params):
    if isinstance(params['type'], str):
        if params['type'] == 'stiff':
            return 1 - np.exp(-m/params['m0'])
        if params['type'] == 'soft':
            return 1 - np.exp(-params['m0']/m)
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
    km = fm(m, {'type':params['km'], 'm0':params['m0']})
    kc = fm(m, {'type':params['kc'], 'm0':params['m0']})
    return (-km * x_total(a, params) - a) * x + x**2/2 * (km + kc) + a * x * scipy.special.hyp2f1(1, 1/params['n'], 1+1/params['n'], -x**params['n'])

def U_old(fm, m, x, a, params):
    return -fm(m, params)*x + x**2/(2*params['tau']) - a/(params['n']+1)*x**(params['n']+1) * scipy.special.hyp2f1(1, (params['n']+1)/params['n'], (2*params['n']+1)/params['n'], -x**params['n'])

def x_crit(n):
    return ((n-1)/(n+1))**(1/n)

def alpha_crit(m, params):
    n = params['n']
    km = f_m(m, {'type':params['km'], 'm0':params['m0']})
    kc = f_m(m, {'type':params['kc'], 'm0':params['m0']})
    return (4 * n * (km + kc))/((n-1)**((n-1)/n)*(n+1)**((n+1)/n))

def m_crit_general(mc, params):
    # general to f(m)
    km = f_m(mc, {'type':params['km'], 'm0':params['m0']})
    kc = f_m(mc, {'type':params['kc'], 'm0':params['m0']})
    return km * x_total(alpha_crit(mc, params), params) - params['x_c'] * (km + kc) + alpha_crit(mc, params) * params['x_c']**params['n']/(params['x_c']**params['n']+1)
    # return (f_m(mc, params) - params['x_c']/params['tau'] + params['a_c'] * params['x_c']**params['n']/(params['x_c']**params['n']+1))

def x_equil(x, m, alpha, params): 
#     x = vs   
    # dynamic nuclear exit rate
    km = f_m(m, {'type':params['km'], 'm0':params['m0']})
    kc = f_m(m, {'type':params['kc'], 'm0':params['m0']})
    return km * x_total(alpha, params)  - x * (km + kc) + alpha * x**params['n']/(x**params['n']+1)

def collect_minima(m_space, x_space, a_space, params):
    
    U_data = np.zeros((len(m_space), len(x_space), len(a_space)))
    
    gmin_overm = []; b1_overm = []; b2_overm = []; inf_overm = [];
    capture2minima = []; capture_mvals = []; capmax = [];
    for mi, mm in enumerate(m_space):
        gmin_coords = []; bi1_coords = []; bi2_coords = []; inf_coords = []
        for ai, aa in enumerate(a_space):
            for xi, xx in enumerate(x_space):
                U_data[mi, xi, ai] = U(f_m, mm, xx, aa, params)
                
            xargs = scipy.signal.argrelextrema(U_data[mi,:,ai], np.less)[0]      
            x_mins = x_space[xargs]
            save_xlen = len(x_mins)    

            if len(x_mins) > 1:
                bi1_coords.append([np.amin(x_mins), aa, U_data[mi, np.amin(xargs), ai]])
                bi2_coords.append([np.amax(x_mins), aa, U_data[mi, np.amax(xargs), ai]])
            elif len(x_mins) > 0:
                gmin_coords.append([x_mins[U_data[mi, xargs, ai] == np.amin(U_data[mi, xargs, ai])][0], aa, np.amin(U_data[mi, xargs, ai])])
                inf_coords.append([x_mins[U_data[mi, xargs, ai] == np.amax(U_data[mi, xargs, ai])][0], aa, np.amax(U_data[mi, xargs, ai])])
            else:
                x_mins = x_space[np.argmin(U_data[mi, :, ai])]
                gmin_coords.append([np.amin(x_mins), aa, U_data[mi, np.argmin(U_data[mi,:,ai]), ai]])
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
    return U_data, gmin_overm, b1_overm, b2_overm, inf_overm, capture2minima, capture_mvals, capmax

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

# def m_solve_real_positives(x_lim, a_current, n_current, tau, stype):

#     # print(a_current)
#     # print(x_lim)
#     x_space = np.linspace(0, np.amax([int(x_lim), 5.]), int(x_lim * 5000))

#     solcands = 1 + x_space**n_current - np.sqrt(n_current * a_current * tau * x_space**(n_current-1))
#     # restricted to real positives, get the two smallest roots.
#     # root_inds = np.argpartition(np.abs(solcands),2)
#     root_inds = np.insert((np.diff(np.sign(solcands)) != 0),0,0)
#     # print(np.sum(root_inds))
#     roots = np.array(sorted(x_space[root_inds]))
#     # print(roots)
#     if stype == 'stiff':
#         m_roots = np.round(-np.log(1 - roots / tau + a_current * roots**n_current / (1+roots**n_current)), 4)
#     elif stype == 'soft':
#         m_roots = np.round(-1 / np.log(1 - roots / tau + a_current * roots**n_current / (1+roots**n_current)), 4)
#         # if m_roots[1] < 0:
#         #     print(x_lim)
#         #     print(a_current)
#         #     print(m_roots)
#         #     print(roots)
#         #     sys.exit()

#     if len(roots) < 2:
#         ## this is where several trials are hanging
#         print('num roots < 2 breaking')
#         print(roots)
#         print(m_roots)
#         print(x_lim)
#         print(a_current)
#         print(tau)

#         return np.NaN, np.NaN

#         # print(root_inds)
#         # print(roots)
#         # print(m_roots)
#         # print(np.array(sorted(x_space[root_inds][0:3])))
#         # plt.plot(x_space, solcands)
#         # plt.plot(plt.xlim(),[0,0])
#         # plt.show()
#         # sys.exit()
#     # print(m_roots)
#     # if len(m_roots) != 2 or np.amax(m_roots) < 0 or roots[0] in [0,len(x_space)] or roots[1] in [0,len(x_space)] or m_roots[0] == m_roots[1]:
#     #     print(m_roots)
        
#     #     print('symbolic solve, check')
#     #     alpha, x, n, tauv = sympy.symbols('alpha x n tauv')
#     #     eqn = sympy.Eq(1 + x**n, sympy.sqrt(alpha * n * tauv * x**(n-1)))

#     #     eqn = eqn.subs(n, n_current)
#     #     eqn = eqn.subs(alpha, a_current)
#     #     eqn = eqn.subs(tauv, tau)

#     #     sol = sympy.solve(eqn, x, force=True)

#     #     roots = np.array([s for s in sol if s.is_real]).astype(np.float64)
#     #     # print(type(roots[0]))
#     #     roots = roots[roots > 0]
#     #     # print(roots)
#     #     # print(type(roots))
#     #     if stype == 'stiff':
#         #     m_roots = np.round(-np.log(1 - roots / tau + a_current * roots**n_current / (1+roots**n_current)), 4)
#         # elif stype == 'soft':
#         #     m_roots = np.round(-1 / np.log(1 - roots / tau + a_current * roots**n_current / (1+roots**n_current)), 4)

#     #     print(m_roots)
#     # returns m1/m0, m2/m0, where m1 is smaller than m2
#     return m_roots[1], m_roots[0]

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

# def calc_PD(params):
    
#     if 'x_max' not in params.keys():
#         params['x_max'] = params['x0']

#     mtst = np.zeros((5000,2)) # + 0j
#     a_space = np.linspace(0, params['a_max']+1, 5000)
#     for ai, aa in enumerate(a_space):
#         # pd in m/m0
#         # mtst[ai,0] = m1n3(aa, n) # * params['m0']
#         # mtst[ai,1] = m2n3(aa, n) # * params['m0']
#         if aa >= params['a_c']:
#             ## for 'soft' this part is problematic, since one of the roots flips to negative at singularity.
#             mtst[ai,:] = m_solve_real_positives(params['x_max']*1.5, aa, params['n'], params['tau'], params['type'])
#     # sum and divide complex conjugate roots under a0
#     # mtst[a_space < params['a_c'],0] = np.sum(mtst[a_space < params['a_c'],:],axis=1)/2
#     # mtst[a_space < params['a_c'],1] = np.sum(mtst[a_space < params['a_c'],:],axis=1)/2

#     return a_space, mtst

def calc_PD_rates(params):

    if 'x_max' not in params.keys():
        params['x_max'] = params['x0']

    a_space = np.linspace(0, params['a_max']+1, params['res'])
    params['a_space'] = a_space
    x_space = np.linspace(0.01, np.amax([params['x_max']*1.5, 5.]), params['res'])
    params['x_space'] = x_space
    # m_space = 

    U_data, gmin_overm, b1_overm, b2_overm, inf_overm, capture2minima, capture_mvals, capmax = collect_minima(params['m_space'], x_space, a_space, params)

    # first col is m values, second col is low minima (blue), third col is high minima (red)
    mtst = np.concatenate((np.array(capture_mvals)[:,None], np.array(capture2minima)[:,None], np.array(capmax)[:,None]), axis=1)   

    return mtst

def update_alpha(t_region, alpha, params):

    if params['dynamics'] == 'constant':
        # print(params['dynamics'])
        if t_region[0] in [1,2]:
            dalpha = 1/params['tau_SG'] * params['dt']

        elif t_region[0] in [3,4]:
            dalpha = -1/params['tau_F'] * params['dt']

        elif t_region[0] in [5,6]:
            dalpha = -1/params['tau_SR'] * params['dt']

    elif params['dynamics'] == 'exp_staticTS':
        # print(params['dynamics'])
        if t_region[0] in [1,2]:
            dalpha = alpha/params['tau_SG'] * params['dt']

        elif t_region[0] in [3,4]:
            dalpha = -alpha/params['tau_F'] * params['dt']

        elif t_region[0] in [5,6]:
            dalpha = -alpha/params['tau_SR'] * params['dt']

    elif params['dynamics'] == 'exp_dynamicTS':
        # print(params['dynamics'])
        if t_region[0] in [1,2]:
            dalpha = alpha/params['tau_SG'] * params['dt']

        elif t_region[0] in [3,4]:
            dalpha = -alpha/params['tau_F'] * params['dt']

        elif t_region[0] in [5,6]:
            dalpha = -alpha/params['tau_SR'] * params['dt']

    else:
        dalpha = 0.    

    return dalpha + params['eps'].draw()


def integrate_profile(m_profile, t_space, params, resultsDF):

    if 'eps' not in params.keys():
        params['eps'] = (0., 1., 0.)
    params['eps'] = noise(params['eps'][0], params['eps'][1], params['eps'][2])

    # print('enter')
    kmdict = {'type':params['km'], 'm0':params['m0']}
    kcdict = {'type':params['kc'], 'm0':params['m0']}

    dt = params['dt']
    x_prof = np.zeros(len(t_space))
    alpha_prof = np.zeros(len(t_space))
    active_region = []; t_primelist = []; t1maxlist = []; deltaVlist = []; tsg_list = []
    t_region = [0,0] # (region, t)
    
    # a_space, mtst = calc_PD(params)
    mtst2 = calc_PD_rates(params)

    if np.isnan(mtst2).any() or np.amin(params['a_space']) > 0:
        params['earlyexit'] = 1
        return resultsDF, params

    a_term = np.amax(mtst2[:,1])

    for ti, tt in enumerate(t_space):

        current_m_ind = np.where(np.abs(params['m_space'] - m_profile[ti]) == np.amin(np.abs(params['m_space'] - m_profile[ti])))[0][0]

        if ti == 0:
            x_prof[ti] = scipy.optimize.fsolve(x_equil, 1., args=(m_profile[ti], params['a0'], params), xtol=1e-10)[0]
            alpha_prof[ti] = params['a0']
            continue;

        params['x_max'] = np.amax(x_prof)
        xUd = np.linspace(0, params['x_max']*2.5, int(1e5))

        if alpha_prof[ti-1] >= params['a_c'][current_m_ind]:
            # m1, m2 = m_solve_real_positives(params['x_max']*1.5, alpha_prof[ti-1], params['n'], params['tau'], params['type'])

            m1ind = np.where(np.abs(mtst2[:,0] - params['m_space'][current_m_ind]) == np.amin(np.abs(mtst2[:,0] - params['m_space'][current_m_ind])))[0][0]

            m1 = mtst2[m1ind, 0] / params['m0']
            a1 = mtst2[m1ind, 1]
            a2 = mtst2[m1ind, 2]

            # if np.isnan(m1) or np.isnan(m2):
            #     print(params)
            #     print(np.amax(m_profile))
            #     print(np.amin(m_profile))
            #     params['earlyexit'] = 1
            #     return resultsDF, params
        # else:
            # m1 = 0; m2 = 0
    
        if alpha_prof[ti-1] >= params['a_max']:
            # permanent memory;
            alpha_prof[ti] = params['a_max']
            x_prof[ti] = scipy.optimize.fsolve(x_equil, x_prof[ti-1], args=(m_profile[ti], alpha_prof[ti], params), xtol=1e-10)[0]
            if m_profile[ti] > params['m_c']:
                active_region.append(2)
            else:
                active_region.append(6)

        else:

            if alpha_prof[ti-1] < params['a_c'][current_m_ind] and m_profile[ti] / params['m0'] > params['m_c']:
            # region 1
                active_region.append(1)

                if t_region[0] != 1:
                    t_region[0] = 1
                    t_region[1] = dt
                else:
                    t_region[1] += dt
                    
                dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                alpha_prof[ti] = alpha_prof[ti-1] + dalph
                # alpha_prof[ti] = a_mstar*np.exp((t_region[1]-4*params['tau_F'])/params['tau_SG'])
            
            elif alpha_prof[ti-1] >= params['a_c'][current_m_ind] and m_profile[ti] / params['m0'] > params['m_c']:
                # region 2
                active_region.append(2)

                if t_region[0] != 2:
                    t_region[0] = 2
                    t_region[1] = dt
                    a_enter = alpha_prof[ti-1]
                else:
                    t_region[1] += dt
                    
                if alpha_prof[ti-1] < params['a_max']:
                    # alpha_prof[ti] = a_enter * np.exp(t_region[1]/params['tau_SG'])
                    # dalph = alpha_prof[ti-1]/params['tau_SG'] * dt
                    dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                    alpha_prof[ti] = alpha_prof[ti-1] + dalph
                else:
                    alpha_prof[ti] = params['a_max']

            elif alpha_prof[ti-1] < params['a_c'][current_m_ind] and m_profile[ti] / params['m0'] < params['m_c']:
                # region 3
                active_region.append(3)
                
                if t_region[0] != 3:
                    if t_region[0] not in [4,5,6]:
                        t_region[0] = 3
                        t_region[1] = dt
                        a_enter = alpha_prof[ti-1]
                    else:
                        t_region[0] = 3
                        t_region[1] += dt
                        a_enter = alpha_prof[ti-1]
                else:
                    t_region[1] += dt
                    
                # alpha_prof[ti] = params['a0'] + (a_enter-params['a0']) * np.exp(-t_region[1]/params['tau_F'])
                if alpha_prof[ti-1] > params['a0']:
                    # dalph = -alpha_prof[ti-1]/params['tau_F'] * dt
                    dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                else:
                    dalph = 0.

                alpha_prof[ti] = alpha_prof[ti-1] + dalph

            elif alpha_prof[ti-1] > params['a_c'][current_m_ind] and alpha_prof[ti-1] < a1 and m_profile[ti] / params['m0'] < params['m_c']:  # and alpha_prof[ti-1] <= a_term:
                # region 4
   
                active_region.append(4)
                
                if t_region[0] != 4:
                    t_region[0] = 4
                    a_enter = alpha_prof[ti-1]
                    # print('here')
                    # print(a_enter)
                    if t_region[0] in [5,6]:
                        t_region[1] += dt
                    else:
                        t_region[1] = dt
                else:
                    t_region[0] = 4
                    t_region[1] += dt
                               
                # alpha_prof[ti] = params['a_c'] + (a_enter - params['a_c']) * np.exp(-t_region[1]/params['tau_F'])

                # alpha_prof[ti] = a_enter * np.exp(-t_region[1]/params['tau_F'])
                # dalph = -alpha_prof[ti-1]/params['tau_F'] * dt
                # alpha_prof[ti] = alpha_prof[ti-1] + dalph
                if alpha_prof[ti-1] > params['a0']:
                    # dalph = -alpha_prof[ti-1]/params['tau_F'] * dt
                    dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                else:
                    dalph = 0.
                # print(dalph)
                alpha_prof[ti] = alpha_prof[ti-1] + dalph

            # elif ((alpha_prof[ti-1] > params['a_c'] and m_profile[ti] / params['m0'] >= m1) or alpha_prof[ti-1] > a_term) and m_profile[ti] / params['m0'] <= m2:
            elif alpha_prof[ti-1] > a1 and alpha_prof[ti-1] < a2 and m_profile[ti] / params['m0'] < params['m_c']:
                # region 5
                active_region.append(5)

                if t_region[0] not in [4,5,6]:
                    # print('enter 5')
                    params['tau_SR'] = params['tau_SG']
                    # print('tau_SR start %e' % (params['tau_SR']))
                    t_region[0] = 5
                    t_region[1] = dt
                    a_enter = alpha_prof[ti-1]
                else:
                    t_region[0] = 5
                    t_region[1] += dt
                    # print(a_enter)
                # alpha_prof[ti] = params['a0'] + (a_enter - params['a0']) * np.exp(-t_region[1]/params['tau_SR'])
                # alpha_prof[ti] = a_enter * np.exp(-t_region[1]/params['tau_SG'])
                # dalph = -alpha_prof[ti-1]/params['tau_SR'] * dt
                dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                alpha_prof[ti] = alpha_prof[ti-1] + dalph

            # elif alpha_prof[ti-1] > params['a_c'] and m_profile[ti] / params['m0'] < params['m_c'] and m_profile[ti] / params['m0'] > m2:
            elif alpha_prof[ti-1] > a2 and m_profile[ti] / params['m0'] < params['m_c']:
                # region 6
                active_region.append(6)
                
                if t_region[0] not in [4,5,6]:
                    # print('enter 6')
                    # print('tau_SG start %e' % (params['tau_SG']))
                    params['tau_SR'] = params['tau_SG']
                    # print('tau_SR start %e' % (params['tau_SR']))
                    t_region[0] = 6
                    t_region[1] = dt
                    a_enter = alpha_prof[ti-1]
                else:
                    t_region[0] = 6
                    t_region[1] += dt
                    
                # alpha_prof[ti] = params['a0'] + (a_enter - params['a0']) * np.exp(-t_region[1]/params['tau_SR'])
                # alpha_prof[ti] = a_enter * np.exp(-t_region[1]/params['tau_SG'])
                # dalph = -alpha_prof[ti-1]/params['tau_SR'] * dt
                dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                alpha_prof[ti] = alpha_prof[ti-1] + dalph

            else:
                print('what')
                sys.exit()

            x_prof[ti] = scipy.optimize.fsolve(x_equil, x_prof[ti-1], args=(m_profile[ti], alpha_prof[ti], params), xtol=1e-10)[0]

        if t_region[0] in [2,5,6] and params['dynamics'] == 'exp_dynamicTS':
            # finding the barrier height.
            # energy slice over x at alpha, m
            U_data = U(f_m, m_profile[ti], xUd, alpha_prof[ti], params)

            x_args = find_peaks(-np.abs(np.diff(U_data)))[0]
            stop = 0
            while len(x_args) < 2 and stop < 20 :
                xUd = np.linspace(0., np.amax(xUd)+0.5, int(3e5))
                U_data = U(f_m, m_profile[ti], xUd, alpha_prof[ti], params)
                x_args = find_peaks(-np.abs(np.diff(U_data)))[0]
                stop += 1

            if stop == 100:
                print('Udata')
                print(U_data)
                np.save('failureU.npy', U_data)
                print(x_args)
                print(x_prof[ti])
                print(np.amin(xUd))
                print(np.amax(xUd))
                # print(x_argmin)
                print(alpha_prof[ti])
                print(params)
                print(m_profile[ti])
                params['earlyexit'] = 1
                return resultsDF, params

            deltaV = np.amax(U_data[x_args]) - np.amin(U_data[x_args])
            # print(deltaV)
            deltaVlist.append(deltaV)
            tsg_list.append(params['tau_SG'])

            if deltaV > 0:
                # print(type(params['tau_SG']))
                params['tau_SR'] = (params['tau_R0'] * np.exp(deltaV/params['TV0SR']))
                 # params['tau_SG'] = params['tau_R0'] * np.exp(-params['TV0']/deltaV)
                params['tau_SG'] = (params['tau_R0'] * np.exp(deltaV/params['TV0SG']))
                # we want t_sg to get larger w/ increasing depth so that it stops increasing               
            else:
                print(deltaV)
                # deltaVlist.append(np.NaN)
                # tsg_list.append(params['tau_SG'])

        elif (t_region[0] in [1,3,4] and params['dynamics'] == 'exp_dynamicTS') or (params['dynamics'] != 'exp_dynamicTS'):
            deltaVlist.append(np.NaN)
            tsg_list.append(params['tau_SG'])

    
    # params['t_prime'] = (np.unique(t_primelist) + 4*params['tau_F']).tolist()
    # params['t1max'] = (np.unique(t1maxlist) + 4*params['tau_F']).tolist()

    active_region.insert(0, active_region[0])
    deltaVlist.insert(0, deltaVlist[0])
    tsg_list.insert(0, tsg_list[0])

    resultsDF['m_profile'] = m_profile
    resultsDF['t_space'] = t_space
    resultsDF['x_prof'] = x_prof
    resultsDF['alpha_prof'] = alpha_prof
    resultsDF['active_region'] = active_region
    resultsDF['deltaV'] = deltaVlist
    resultsDF['tSG'] = tsg_list

    return resultsDF, params

def summary_stats(resultsDF, params, verbose=True):
    
    t = resultsDF['t_space'].values
    m = resultsDF['m_profile'].values
    alpha = resultsDF['alpha_prof'].values
    x = resultsDF['x_prof'].values

    def slice_tuple(slice_):
        return [slice_.start, slice_.stop, slice_.step]

    # if 'type' not in params.keys():
    #     print('assuming stiff')
    #     print('summary')
    #     params['type'] = 'stiff'

    # take the time series arrays and calculate observables.
    dt = t[1] - t[0]

    if params['km'] == 'stiff':
        priming_mask = m / params['m0'] > params['m_c']
    elif params['km'] == 'soft':
        priming_mask = m / params['m0'] < params['m_c']
    else:
        print('no type')
        return

    regions = nd.find_objects(nd.label(priming_mask)[0])
    # print([r for r in regions])
    priming_times = [len(m[r]) * dt for r in regions]

    # average priming stiffness
    stiffP = np.array([np.mean(m[r]) for r in regions])

    if len(stiffP) == 0:
        # print(stiffP)
        if np.amax(m) <= params['m_c'] * params['m0']:
            print('no priming')
            priming_times = np.array([np.NaN])
            memory_times = np.array([np.NaN])
            stiffA = np.array([0])
            stiffP = np.array([0])

            return priming_times, memory_times, stiffP, stiffA

    priming_start_stop = np.array([slice_tuple(r[0]) for r in regions])
    # print(priming_start_stop)
    # print(priming_start_stop[:,1])
    # memory time
    # mem_start_candidates = [np.amax(t[r]) + dt for r in regions]
    # print(mem_start_candidates)
    memory_times = np.zeros(len(regions))

    muniq = np.unique(m)
    x_basic = np.zeros(len(muniq))
    for mi, mm in enumerate(muniq):
        x_basic[mi] = scipy.optimize.fsolve(x_equil, params['x0'], args=(mm, params['a0'], params), xtol=1e-10)[0]

    # print(x_basic)
    for mi, mm in enumerate(priming_start_stop[:,1]):
        if mi == len(priming_start_stop) - 1:
            end_ind = None
            # print('here')
        else:
            end_ind = priming_start_stop[mi+1,0]
            # print('there')

        stiffy = m[mm:end_ind]
        # average relaxation stiffness
        stiffA = np.mean(stiffy)
        x_compare = np.array([x_basic[muniq == ss][0] for ss in stiffy])
        # print(x_compare)
        # print(x[mm:end_ind])

        # count as memory the time where the x expression is outside 5% of the target w/ baseline alpha
        memory_times[mi] = np.sum(np.abs(x[mm:end_ind] - x_compare) > 0.05*x_compare) * dt
        
        # print(np.sum(np.abs(x[mm:end_ind] - x_compare) > 0.05*x_compare))    
        
    mech_stats = np.abs(stiffP - stiffA) / params['m0']

    if verbose:
        print('priming times', end=" "); print(priming_times)
        print('memory times', end=" "); print(memory_times)
        print('mechanical ratios', end=" "); print(mech_stats)

    # print('memory times', end=" "); print(memory_times)

    return np.array(priming_times), memory_times, stiffP, stiffA # mech_stats


def run_profile(e_func, inputs, params, resultsDF):


    def early_exit(resultsDF, params, m_profile, t_space):
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
    
    # a_space, mtst = calc_PD(params)
    # print(a_space)
    # print(mtst)
    # try:
    #     params['a1max'] = a_space[mtst[:,0] == np.nanmin(mtst[:,0])][0]
    #     # params['a1max'] = a_space[mtst[:,0] == np.amin(mtst[:,0])][0]
    # except:
    #     params['a1max'] = np.NaN
    
    m_c = scipy.optimize.fsolve(m_crit_general, 0.5, args=(params), xtol=1e-10)[0] / params['m0']
    params['m_c'] = m_c

    # print(inputs)
    # print(type(inputs))
    
    if inputs[0,1] < 0:
        ## flag for random number for stiffness. negative number means base the stiffness profiles on m_c
        # pstiff
        inputs[1,1] = m_c * params['m0'] + (-inputs[1,1])*10

        # mstiff
        inputs[0,1] = m_c * params['m0'] - (-inputs[0,1]*m_c*params['m0'] * 0.9) 
        inputs[2,1] = m_c * params['m0'] - (-inputs[2,1]*m_c*params['m0'] * 0.9) 

    m_profile = build_mprof(inputs, params['resolution'])

    m_space = np.linspace(np.amin([0.1, np.amin(m_profile)]), np.amax(m_profile)*1.5, params['res'])
    choose_mc_ind = np.where(np.abs(m_space - m_c*params['m0']) == np.amin(np.abs(m_space-m_c*params['m0'])))[0][0]
    a_c = alpha_crit(m_space, params)
    params['a_c'] = a_c;
    params['m_space'] = m_space #         print(np.unique(m_profile))    print(len(m_profile))
    # print(len(m_profile))
    # m profile is in absolute units
    # hours
    print('--------')
    print('a_c = %f, x_c = %f, m_c in absolute = %f' % (a_c[choose_mc_ind], x_c, m_c * params['m0']))

    mtst2 = calc_PD_rates(params)

    t_space = np.linspace(0, np.sum(inputs[:,0]), len(m_profile))
    params['dt'] = t_space[1]-t_space[0]

    print(params['dynamics'])

    if np.amin(m_profile)/params['m0'] >= params['m_c']: #/params['m0']:
        early_exit(resultsDF, params, m_profile, t_space)

    resultsDF, params = e_func(m_profile, t_space, params, resultsDF)

    if 'earlyexit' in params.keys():
        early_exit(resultsDF, params, m_profile, t_space)

    priming_times, memory_times, stiffP, stiffA = summary_stats(resultsDF, params)

    return resultsDF, params, priming_times, memory_times, stiffP, stiffA