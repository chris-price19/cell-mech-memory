class noise:
    def __init__(self, mean=0., std = 1., mag = 0.1):
        self.mean = mean
        self.std = std
        self.mag = mag
        self.rg = Generator(PCG64())
    def draw(self):
        return self.mag * self.rg.normal(self.mean, self.std)


if 'eps' not in params.keys():
        params['eps'] = noise(0., 1., 0.012)


## figure 2 3D landscapes
params = {}
# stiff genes
res = 100
params = {}
params['kc'] = 1.
params['km'] = 'stiff'
params['n'] = 3
params['m0'] = 3.
params['x0'] = 1.; params['a0'] = 0.5; params['xtt'] = 0.;
params['x_c'] = x_crit(params['n'])

x_space = np.linspace(0, 4., res)
a_space = np.linspace(0.1, 4., res)
m_space = np.linspace(0., 7., res)

a_c = alpha_crit(m_space, params)
params['a_c'] = a_c;

m_c = scipy.optimize.fsolve(m_crit_general, 0.5, args=(params), xtol=1e-10)[0] / params['m0']
params['m_c'] = m_c; print(m_c)

U_data, gmin_overm, b1_overm, b2_overm, inf_overm, capture2minima, capture_mvals, capmax = collect_minima(m_space, x_space, a_space, params)
