# def summary_stats(resultsDF, params, verbose=True):
    
#     t = resultsDF['t_space'].values
#     m = resultsDF['m_profile'].values
#     alpha = resultsDF['alpha_prof'].values
#     x = resultsDF['x_prof'].values
#     active_region = resultsDF['active_region'].values

#     def slice_tuple(slice_):
#         return [slice_.start, slice_.stop, slice_.step]

#     if 'type' not in params.keys():
#         print('assuming stiff')
#         print('summary')
#         params['type'] = 'stiff'

#     # take the time series arrays and calculate observables.
#     dt = t[1] - t[0]

#     if params['type'] == 'stiff':
#         priming_mask = m / params['m0'] > params['m_c']
#     elif params['type'] == 'soft':
#         priming_mask = m / params['m0'] < params['m_c']
#     else:
#         print('no type')
#         return

#     regions = nd.find_objects(nd.label(priming_mask)[0])
#     # print([r for r in regions])
#     priming_times = [len(m[r]) * dt for r in regions]

#     # average priming stiffness
#     stiffP = np.array([np.mean(m[r]) for r in regions])

#     if len(stiffP) == 0:
#         # print(stiffP)
#         if np.amax(m) <= params['m_c'] * params['m0']:
#             print('no priming')
#             priming_times = np.array([np.NaN])
#             memory_times = np.array([np.NaN])
#             stiffA = np.array([0])
#             stiffP = np.array([0])

#             return priming_times, memory_times, stiffP, stiffA

#     priming_start_stop = np.array([slice_tuple(r[0]) for r in regions])
#     # print('&&&')
#     # print(params['dt'])
#     # print(priming_start_stop)
#     # print(priming_start_stop[:,1])

#     # memory time
#     # mem_start_candidates = [np.amax(t[r]) + dt for r in regions]
#     # print(mem_start_candidates)
#     memory_times = np.zeros(len(regions))

#     muniq = np.unique(m)
#     x_basic = np.zeros(len(muniq))
#     for mi, mm in enumerate(muniq):
#         x_basic[mi] = scipy.optimize.fsolve(x_equil, params['x0'], args=(mm, params['a0'], params), xtol=1e-10)[0]

#     # print(x_basic)
#     for mi, mm in enumerate(priming_start_stop[:,1]):
#         if mi == len(priming_start_stop) - 1:
#             end_ind = None
#             # print('here')
#         else:
#             end_ind = priming_start_stop[mi+1,0]
#             # print('there')

#         stiffy = m[mm:end_ind]
#         # average relaxation stiffness
#         stiffA = np.mean(stiffy)
#         x_compare = np.array([x_basic[muniq == ss][0] for ss in stiffy])
#         # print(x_compare)
#         # print(x[mm:end_ind])

#         # count as memory the time where the x expression is outside 5% of the target w/ baseline alpha
#         memory_times[mi] = np.sum(np.abs(x[mm:end_ind] - x_compare) > 0.05*x_compare) * dt
        
#         # print(np.sum(np.abs(x[mm:end_ind] - x_compare) > 0.05*x_compare))    
    
#     counts, cumsum, ids = rle(active_region) 
#     memory2 = np.sum(counts[np.isin(ids, [3])]) * params['dt']
#     priming2 = np.sum(counts[np.isin(ids, [2])]) * params['dt']

#     if ids[-1] == 2:
#         avgm = np.mean(np.unique(m))
#         priming2 = np.sum(m > avgm) * params['dt']
#         memory2 = -1.  # np.sum(m < avgm) * params['dt']
#     # if np.abs(memory2 - memory_times[0]) > 24:
#     #     print('oh no')
#     #     print(priming_times)
#     #     print(priming2)
#     #     print(memory_times)
#     #     print(memory2)
#     #     print(counts)
#     #     print(ids)
#     #     print(priming_start_stop)
#         # sys.exit()

#     mech_stats = np.abs(stiffP - stiffA) / params['m0']

#     if verbose:
#         print('counts', end=" "); print(counts); print(np.sum(counts))
#         print('ids', end=" "); print(ids)
#         print('priming times', end=" "); print(priming_times)
#         print('priming check', end = " "); print(priming2)
#         print('memory times', end=" "); print(memory_times)
#         print('memory check', end = " "); print(memory2)
#         print('mechanical ratios', end=" "); print(mech_stats)

#     # print('memory times', end=" "); print(memory_times)

#     # return np.array(priming_times), memory_times, stiffP, stiffA # mech_stats
#     return np.array([priming2]), np.array([memory2]), stiffP, stiffA