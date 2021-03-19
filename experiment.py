def collect_minima(m_space, x_space, a_space, params):
    
    U_data = np.zeros((len(m_space), len(x_space), len(a_space)))
    
    gmin_overm = []; b1_overm = []; b2_overm = []; inf_overm = [];
    capture2minima = []; capture_mvals = [];
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
                x_mins = x_space[np.argmin(Uslice[:,ai])]
                gmin_coords.append([np.amin(x_mins), aa, U_data[mi, np.argmin(U_data[mi,:,ai]), ai]])
#                 gmin_coords.append([np.amin(x_mins), aa, U_data[mi, np.amin(xargs), ai]])
        
        gmin_overm.append(np.array(gmin_coords))
        b1_overm.append(np.array(bi1_coords))
        b2_overm.append(np.array(bi2_coords))
        inf_overm.append(np.array(inf_coords))
        
        if len(b1_overm[-1]) > 0:
            # get the minimum alpha where two minima appear (B1 and B2 are populated, second column)
            # verified that using b1 or b2 gives the same results as it should
            capture2minima.append(np.amin(b1_overm[-1][:,1]))
            capture_mvals.append(mm)

#                 if aa > a_c[choose_m_ind]:
#                     x_mins = x_space[xargs]
#             #         print(x_mins)
#                     if len(x_mins) > 1:
#                         m1coords.append([np.amin(x_mins), aa, Uslice[np.amin(xargs), ai]])
#                         m2coords.append([np.amax(x_mins),aa, Uslice[np.amax(xargs), ai]])
#                     else:
#             #             if np.abs(np.amin(x_mins) - m1coords[-1][0]) < np.abs(np.amin(x_mins) - m2coords[-1][0]):
#                         m1coords.append([np.amin(x_mins), aa, Uslice[np.amin(xargs), ai]])
#             #             else:
#             #                 m2coords.append([np.amax(x_mins),aa, Uslice[np.amax(xargs), ai]])
#                 else:
#             #         ax2.plot(x_space, Uslice[:,ai])
#             #         sys.exit()
#                     x_mins = x_space[np.argmin(Uslice[:,ai])]
#                     m0coords.append([np.amin(x_mins), aa, Uslice[np.argmin(Uslice[:,ai]), ai]])

    # m0coords = np.array(m0coords); m1coords = np.array(m1coords);m2coords = np.array(m2coords)
    print(capture2minima)
#     print(capture2test)
    return U_data, gmin_overm, b1_overm, b2_overm, inf_overm, capture2minima, capture_mvals

params = {}
params['kc'] = 1.
params['km'] = 'stiff'
params['n'] = 3
params['m0'] = 3.
params['x_c'] = x_crit(params['n'])
res = 100

m_space = np.linspace(0., 10., res)
x_space = np.linspace(0, 3., res)
a_space = np.linspace(0.01, 3., res)
# ac_ind = np.where(np.abs(a_space - a_c) == np.amin(np.abs(a_space-a_c)))[0]
a_c = alpha_crit(m_space, params)
m_c = scipy.optimize.fsolve(m_crit_general, 0.5, args=(params), xtol=1e-10)[0] / params['m0']

choose_m = 7.
choose_a = 2.6

plt.plot(a_c, m_space/params['m0'])
plt.plot([np.amin(a_space), np.amax(a_space)],[m_c, m_c])
plt.scatter(choose_a, choose_m/params['m0'])


choose_a_ind = np.where(np.abs(a_space - choose_a) == np.amin(np.abs(a_space-choose_a)))[0][0]
choose_m_ind = np.where(np.abs(m_space - choose_m) == np.amin(np.abs(m_space - choose_m)))[0][0]

# U_data, gmin_overm, b1_overm, b2_overm, inf_overm, capture2minima, capture_mvals = collect_minima(m_space, x_space, a_space, params)
U_data, m0coords, m1coords, m2coords, inf_overm, capture2minima, capture_mvals = collect_minima(m_space, x_space, a_space, params)

print(capture_mvals)
plt.plot(capture2minima, capture_mvals)

Uslice = U_data[choose_m_ind, :, :].squeeze()
print(choose_m_ind)
m0coords = m0coords[choose_m_ind]
m1coords = m1coords[choose_m_ind]
m2coords = m2coords[choose_m_ind]
print(m0coords.shape)
print(m1coords.shape)
print(m2coords.shape)
print(Uslice.shape)

# print(np.shape(np.abs(np.diff(U_data, axis=0))))
fig2, ax2 = plt.subplots(1,1, figsize=(5, 4))

if choose_a < a_c[choose_m_ind]:
    ax2.plot(x_space, Uslice[:,choose_a_ind], color=mcolors['red'])
    xargs = scipy.signal.argrelextrema(Uslice[:,choose_a_ind], np.less)[0]      
    x_mins = x_space[xargs]
    ax2.scatter(x_mins, Uslice[xargs, choose_a_ind], s=100, color=mcolors['blueviolet'], zorder=10)    
else:
# m1 is purple
# m2 is blue
    ax2.plot(x_space, Uslice[:,choose_a_ind], color=mcolors['red'])
    xargs = scipy.signal.argrelextrema(Uslice[:,choose_a_ind], np.less)[0]      
    x_mins = x_space[xargs]
    ax2.scatter(x_mins, Uslice[xargs, choose_a_ind], s=100, color=mcolors['blueviolet'], zorder=10)
#     ax2.scatter(m2coords[int(choose_a_ind-len(m0coords)),0], m2coords[int(choose_a_ind-len(m0coords)),2], s=100, color=mcolors['dodgerblue'], zorder=10)




def calc_PD_rates(params):

    a_space = np.linspace(0, params['a_max']+1, 500)
    params['a_space'] = a_space
    x_space = np.linspace(0.01, np.amax([params['x_max']*1.5, 5.]), 500)
    params['x_space'] = x_space
    # m_space = 

    U_data, gmin_overm, b1_overm, b2_overm, inf_overm, capture2minima, capture_mvals, capmax = collect_minima(params['m_space'], x_space, a_space, params)

    # first col is m values, second col is low minima (blue), third col is high minima (red)
    mtst = np.concatenate((np.array(capture_mvals)[:,None], np.array(capture2minima)[:,None], np.array(capmax)[:,None]), axis=1)   

    return mtst