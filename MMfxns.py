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

def U(fm, m, x, a, params):
    
    return -fm(m, params)*x + x**2/(2*params['tau']) - a/(params['n']+1)*x**(params['n']+1) * scipy.special.hyp2f1(1, (params['n']+1)/params['n'], (2*params['n']+1)/params['n'], -x**params['n'])

def x_crit(n):
    return ((n-1)/(n+1))**(1/n)

def alpha_crit(n, tau):
    return 4*n/(tau*(n-1)**((n-1)/n)*(n+1)**((n+1)/n))

def m_crit(n, tau, m0):
    return -m0*np.log(1 - (( (n-1)/(n+1))**(1/n) ) * (1/tau-2/(n+1)))

def m_crit_over_m0(n, tau):
    # stiff only
    return -np.log(1 - ( ((n-1)/(n+1))**(1/n) ) * (1/tau-2/(n+1)))

def m_crit_general(mc, params):
    # general to f(m)
    return ( (f_m(mc, params) - params['x_c']/params['tau'] + params['a_c'] * params['x_c']**params['n']/(params['x_c']**params['n']+1)) ) 

def f_m(m, params):
    if params['type'] == 'stiff':
        return 1 - np.exp(-m/params['m0'])
    if params['type'] == 'soft':
        return 1 - np.exp(-params['m0']/m)
    if params['type'] ==  'basic':
        return m/params['m0']

def x_equil(x, m, alpha, params): 
#     x = vs
    return ( f_m(m, params) - x/params['tau'] + alpha * x**params['n']/(x**params['n']+1) )

def m1n3(alpha, n):
    if n != 3:
        return
    x = (
        ((1/2 - 1j*np.sqrt(3)/2)*(3 + cmath.sqrt(9-4*np.sqrt(3)*alpha**(3/2)))**(1/3))/6**(1/3) + 
        ((1/2 + 1j*np.sqrt(3)/2)*(3 - cmath.sqrt(9-4*np.sqrt(3)*alpha**(3/2)))**(1/3))/6**(1/3)
    )
    m = -np.log(1-x + alpha * x**n/(1+x**n))
    return m

def m2n3(alpha, n):
    if n != 3:
        return
    x = (
        -((1/2 - 1j*np.sqrt(3)/2)**2 * (3 + cmath.sqrt(9-4*np.sqrt(3)*alpha**(3/2)))**(1/3))/6**(1/3) -
        ((1/2 + 1j*np.sqrt(3)/2)**2 * (3 - cmath.sqrt(9-4*np.sqrt(3)*alpha**(3/2)))**(1/3))/6**(1/3)
    )
    m = -np.log(1 - x + alpha * x**n / (1+x**n))
    return m

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

def m_solve_real_positives(x_lim, a_current, n_current, tau, stype):

    # print(a_current)
    # print(x_lim)
    x_space = np.linspace(0, np.amax([int(x_lim), 5.]), int(x_lim * 5000))

    solcands = 1 + x_space**n_current - np.sqrt(n_current * a_current * tau * x_space**(n_current-1))
    # restricted to real positives, get the two smallest roots.
    # root_inds = np.argpartition(np.abs(solcands),2)
    root_inds = np.insert((np.diff(np.sign(solcands)) != 0),0,0)
    # print(np.sum(root_inds))
    roots = np.array(sorted(x_space[root_inds]))
    # print(roots)
    if stype == 'stiff':
        m_roots = np.round(-np.log(1 - roots / tau + a_current * roots**n_current / (1+roots**n_current)), 4)
    elif stype == 'soft':
        m_roots = np.round(-1 / np.log(1 - roots / tau + a_current * roots**n_current / (1+roots**n_current)), 4)
        # if m_roots[1] < 0:
        #     print(x_lim)
        #     print(a_current)
        #     print(m_roots)
        #     print(roots)
        #     sys.exit()

    if len(roots) < 2:
        ## this is where several trials are hanging
        print('num roots < 2 breaking')
        print(roots)
        print(m_roots)
        print(x_lim)
        print(a_current)
        print(tau)

        return np.NaN, np.NaN

        # print(root_inds)
        # print(roots)
        # print(m_roots)
        # print(np.array(sorted(x_space[root_inds][0:3])))
        # plt.plot(x_space, solcands)
        # plt.plot(plt.xlim(),[0,0])
        # plt.show()
        # sys.exit()
    # print(m_roots)
    # if len(m_roots) != 2 or np.amax(m_roots) < 0 or roots[0] in [0,len(x_space)] or roots[1] in [0,len(x_space)] or m_roots[0] == m_roots[1]:
    #     print(m_roots)
        
    #     print('symbolic solve, check')
    #     alpha, x, n, tauv = sympy.symbols('alpha x n tauv')
    #     eqn = sympy.Eq(1 + x**n, sympy.sqrt(alpha * n * tauv * x**(n-1)))

    #     eqn = eqn.subs(n, n_current)
    #     eqn = eqn.subs(alpha, a_current)
    #     eqn = eqn.subs(tauv, tau)

    #     sol = sympy.solve(eqn, x, force=True)

    #     roots = np.array([s for s in sol if s.is_real]).astype(np.float64)
    #     # print(type(roots[0]))
    #     roots = roots[roots > 0]
    #     # print(roots)
    #     # print(type(roots))
    #     if stype == 'stiff':
        #     m_roots = np.round(-np.log(1 - roots / tau + a_current * roots**n_current / (1+roots**n_current)), 4)
        # elif stype == 'soft':
        #     m_roots = np.round(-1 / np.log(1 - roots / tau + a_current * roots**n_current / (1+roots**n_current)), 4)

    #     print(m_roots)
    # returns m1/m0, m2/m0, where m1 is smaller than m2
    return m_roots[1], m_roots[0]

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

def calc_PD(params):
    
    if 'x_max' not in params.keys():
        params['x_max'] = params['x0']

    mtst = np.zeros((5000,2)) # + 0j
    a_space = np.linspace(0, params['a_max']+1, 5000)
    for ai, aa in enumerate(a_space):
        # pd in m/m0
        # mtst[ai,0] = m1n3(aa, n) # * params['m0']
        # mtst[ai,1] = m2n3(aa, n) # * params['m0']
        if aa >= params['a_c']:
            ## for 'soft' this part is problematic, since one of the roots flips to negative at singularity.
            mtst[ai,:] = m_solve_real_positives(params['x_max']*1.5, aa, params['n'], params['tau'], params['type'])
    # sum and divide complex conjugate roots under a0
    # mtst[a_space < params['a_c'],0] = np.sum(mtst[a_space < params['a_c'],:],axis=1)/2
    # mtst[a_space < params['a_c'],1] = np.sum(mtst[a_space < params['a_c'],:],axis=1)/2

    return a_space, mtst

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

    # print('enter')
    if 'eps' not in params.keys():
        params['eps'] = noise(0., 1., 0.012)

    dt = t_space[1] - t_space[0]
    x_prof = np.zeros(len(t_space))
    alpha_prof = np.zeros(len(t_space))
    active_region = []; t_primelist = []; t1maxlist = []; deltaVlist = []; tsg_list = []
    t_region = [0,0] # (region, t)
    
    a_space, mtst = calc_PD(params)
    if np.isnan(mtst).any() or np.amin(a_space) > 0:
        params['earlyexit'] = 1
        return resultsDF, params
    a_term = a_space[np.where(mtst < 0)[0]][0]

    for ti, tt in enumerate(t_space):
        
        if ti == 0:
            x_prof[ti] = params['x0']
            alpha_prof[ti] = params['a0']
            continue;

        params['x_max'] = np.amax(x_prof)
        xUd = np.linspace(0, params['x_max']*2.5, int(1e5))

        if alpha_prof[ti-1] >= params['a_c']:
            m1, m2 = m_solve_real_positives(params['x_max']*1.5, alpha_prof[ti-1], params['n'], params['tau'], params['type'])
            if np.isnan(m1) or np.isnan(m2):
                print(params)
                print(np.amax(m_profile))
                print(np.amin(m_profile))
                params['earlyexit'] = 1
                return resultsDF, params
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
#             dx = dt * (1 - np.exp(-m_profile[ti-1]/params['m0']) - x_prof[ti-1]/params['tau'] + alpha_prof[ti-1] * x_prof[ti-1]**n/(x_prof[ti-1]**n + 1))
#             x_prof[ti] = x_prof[ti-1] + dx
        else:
            
            a_ss = (params['a_c'] - params['a0']) * f_m(m_profile[ti], params) # (1 - np.exp(-m_profile[ti] / params['m0']))
            a_mstar = a_ss + (params['a0'] - a_ss)*np.exp(-4)

            tprime = params['tau_SG'] * np.log(params['a_c']/a_mstar)
            t1max = params['tau_SG'] * np.log(params['a1max']/a_mstar)
            t_primelist.append(tprime)
            t1maxlist.append(t1max)

            if alpha_prof[ti-1] < params['a_c'] and m_profile[ti] / params['m0'] > params['m_c']:
#                 # region 1
                active_region.append(1)

                if t_region[0] != 1:
                    t_region[0] = 1
                    t_region[1] = dt
                else:
                    t_region[1] += dt
                    
                # if t_region[1] < 4*params['tau_F']:
                #     # alpha_prof[ti] = a_ss + (params['a0'] - a_ss)*np.exp(-t_region[1]/params['tau_F'])
                #     # dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                #     dalph = alpha_prof[ti-1]/params['tau_F'] * params['dt']
                # else:
                # dalph = alpha_prof[ti-1]/params['tau_SG'] * dt
                dalph = update_alpha(t_region, alpha_prof[ti-1], params)
                alpha_prof[ti] = alpha_prof[ti-1] + dalph
                # alpha_prof[ti] = a_mstar*np.exp((t_region[1]-4*params['tau_F'])/params['tau_SG'])
            
            elif alpha_prof[ti-1] >= params['a_c'] and m_profile[ti] / params['m0'] > params['m_c']:
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

            elif alpha_prof[ti-1] < params['a_c'] and m_profile[ti] / params['m0'] < params['m_c']:
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

            elif alpha_prof[ti-1] > params['a_c'] and m_profile[ti] / params['m0'] < m1 and alpha_prof[ti-1] <= a_term:
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

            elif ((alpha_prof[ti-1] > params['a_c'] and m_profile[ti] / params['m0'] >= m1) or alpha_prof[ti-1] > a_term) and m_profile[ti] / params['m0'] <= m2:
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

            elif alpha_prof[ti-1] > params['a_c'] and m_profile[ti] / params['m0'] < params['m_c'] and m_profile[ti] / params['m0'] > m2:
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
                pass
                # print('dv')
                # print(deltaV)

                # deltaVlist.append(np.NaN)
                # tsg_list.append(params['tau_SG'])

        elif (t_region[0] in [1,3,4] and params['dynamics'] == 'exp_dynamicTS') or (params['dynamics'] != 'exp_dynamicTS'):
            deltaVlist.append(np.NaN)
            tsg_list.append(params['tau_SG'])

    
    params['t_prime'] = (np.unique(t_primelist) + 4*params['tau_F']).tolist()
    params['t1max'] = (np.unique(t1maxlist) + 4*params['tau_F']).tolist()

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
    active_region = resultsDF['active_region'].values

    def slice_tuple(slice_):
        return [slice_.start, slice_.stop, slice_.step]

    if 'type' not in params.keys():
        print('assuming stiff')
        print('summary')
        params['type'] = 'stiff'

    # take the time series arrays and calculate observables.
    dt = t[1] - t[0]

    if params['type'] == 'stiff':
        priming_mask = m / params['m0'] > params['m_c']
    elif params['type'] == 'soft':
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
    # print('&&&')
    # print(params['dt'])
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
    
    counts, cumsum, ids = rle(active_region) 
    memory2 = np.sum(counts[np.isin(ids, [5,6])]) * params['dt']
    priming2 = np.sum(counts[np.isin(ids, [1,2])]) * params['dt']
    if np.abs(memory2 - memory_times[0]) > 24:
        print('oh no')
        print(priming_times)
        print(priming2)
        print(memory_times)
        print(memory2)
        print(counts)
        print(ids)
        print(priming_start_stop)
        # sys.exit()

    mech_stats = np.abs(stiffP - stiffA) / params['m0']

    if verbose:
        print('counts', end=" "); print(ids)
        print('ids', end=" "); print(counts)
        print('priming times', end=" "); print(priming_times)
        print('priming check', end = " "); print(priming2)
        print('memory times', end=" "); print(memory_times)
        print('memory check', end = " "); print(memory2)
        print('mechanical ratios', end=" "); print(mech_stats)

    # print('memory times', end=" "); print(memory_times)

    # return np.array(priming_times), memory_times, stiffP, stiffA # mech_stats
    return np.array([priming2]), np.array([memory2]), stiffP, stiffA


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

    x_c = x_crit(params['n'])
    a_c = alpha_crit(params['n'], params['tau'])
    params['x_c'] = x_c; params['a_c'] = a_c;

    a_space, mtst = calc_PD(params)
    # print(a_space)
    # print(mtst)
    try:
        params['a1max'] = a_space[mtst[:,0] == np.nanmin(mtst[:,0])][0]
        # params['a1max'] = a_space[mtst[:,0] == np.amin(mtst[:,0])][0]
    except:
        params['a1max'] = np.NaN
    
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

    # if inputs[1,1] < m_c * params['m0'] or inputs[2,1] > m_c * params['m0']:
    #     print(inputs)
    #     sys.exit()
    # print(inputs)
    # sys.exit()
    print('--------')
    print('a_c = %f, x_c = %f, m_c in absolute = %f' % (a_c, x_c, m_c * params['m0']))

    m_profile = build_mprof(inputs, params['resolution']) #         print(np.unique(m_profile))    print(len(m_profile))
    # print(len(m_profile))
    # m profile is in absolute units
    # hours
    t_space = np.linspace(0, np.sum(inputs[:,0]), len(m_profile))
    # print('dt = %f hours' % (t_space[1]-t_space[0]))
    params['dt'] = t_space[1]-t_space[0]

    print(params['dynamics'])

    if np.amin(m_profile)/params['m0'] >= params['m_c']: #/params['m0']:
        early_exit(resultsDF, params, m_profile, t_space)

    resultsDF, params = e_func(m_profile, t_space, params, resultsDF)

    if 'earlyexit' in params.keys():
        early_exit(resultsDF, params, m_profile, t_space)

    priming_times, memory_times, stiffP, stiffA = summary_stats(resultsDF, params)

    # print(type(priming_times))
    # print(type(memory_times))
    # print(type(stiffP))
    # print(type(stiffA))

    return resultsDF, params, priming_times, memory_times, stiffP, stiffA