import autograd.numpy as np
import scipy
import autograd
solve_ivp = scipy.integrate.solve_ivp

class Dataset:
    def __init__(self, obj, m = None, **kwargs):
        self.m = m
        self.g = 9.8
        self.k = 1
        self.obj = obj
        self.dim = 2
        self.dof = obj * self.dim  # degree of freedom

    def dynamics_hamiltonian_fn(self, t, coords):
        grad_ham = autograd.grad(self.hamiltonian_fn)
        grad = grad_ham(coords)
        dxdt = np.append(grad[self.dof:], -grad[:self.dof])
        return dxdt

    def hamiltonian_kinetic(self, coords):
        T = 0.
        for i in range(self.obj):
            T = T + 0.5 * np.sum(coords[self.dof + 2 * i: self.dof + 2 * i + 2] ** 2, axis = 0)/ self.m[i]
        return T

    def hamiltonian_potential(self, coords):
        k = self.k
        U = 0.
        for i in range(self.obj):
            for j in range(i):
                U = U - k * self.m[i] * self.m[j] / (
                            (coords[2 * i] - coords[2 * j]) ** 2 +
                            (coords[2 * i + 1] - coords[2 * j + 1]) ** 2)**0.5
        return U

    def hamiltonian_fn(self, coords, eng = True):
        assert (len(coords) == self.dof * 2)
        T, U = self.hamiltonian_kinetic(coords), self.hamiltonian_potential(coords)
        # NOT STANDARD
        H = T + U
        return H

    def dynamics_lagrangian_fn(self, t, coords):
        grad_lag = autograd.grad(self.lagrangian_fn)
        jaco_lag = autograd.jacobian(grad_lag)
        grad = grad_lag(coords)
        jaco = jaco_lag(coords)
        size = self.dof
        g = np.linalg.inv(jaco[size:, size:]) @ (grad[:size] - jaco[size:, :size] @ coords[size:])
        return np.append(coords[size:], g)

    def lagrangian_kinetic(self, coords):
        T = 0.
        for i in range(self.obj):
            T = T  + 0.5 * self.m[i] * np.sum(coords[self.dof+2*i: self.dof+2*i+2]**2, axis = 0)
        return T

    def lagrangian_potential(self, coords):
        k = self.k
        U = 0.
        for i in range(self.obj):
            for j in range(i):
                U = U - k * self.m[i] * self.m[j] / (
                        (coords[2 * i] - coords[2 * j]) ** 2 +
                        (coords[2 * i + 1] - coords[2 * j + 1]) ** 2) ** 0.5
        return U

    def lagrangian_fn(self, coords, eng = False):
        assert(len(coords) == self.dof * 2)
        T, U = self.lagrangian_kinetic(coords), self.lagrangian_potential(coords)
        L = T - U if not eng else T + U
        return L

    @staticmethod
    def rotate2d(p, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (R @ p.reshape(2, 1)).squeeze()

    def random_config(self, nu=0.5, min_radius=1, max_radius=5, system="modlanet"):
        # for n objects evenly distributed around the circle,
        # which means angle(obj_i, obj_{i+1}) = 2*pi/n
        # we made the requirement there that m is the same
        # for every obejct to simplify the formula.
        # But it can be improved.
        state = np.zeros(self.dof * 2)

        p0 = 2 * np.random.rand(2) - 1
        r = np.random.rand() * (max_radius - min_radius) + min_radius

        theta = 2 * np.pi / self.obj
        p0 *= r / np.sqrt(np.sum((p0 ** 2)))
        for i in range(self.obj):
            state[2*i: 2*i+2] = self.rotate2d(p0, theta = i * theta)

        # # velocity that yields a circular orbit
        dirction = p0/np.sqrt((p0*p0).sum())
        v0 = self.rotate2d(dirction, theta=np.pi / 2)
        k  = self.k / (2*r)
        for i in range(self.obj):
            v  = v0 * np.sqrt(k * sum([self.m[j % self.obj] / np.sin((j - i) * theta / 2) for j in range(i + 1, self.obj + i)]))
            # make the circular orbits slightly chaotic
            if system == 'modlanet':
                v *= (1 + nu *  (2 * np.random.rand(2) - 1))
            else:
                v *= self.m[i] * (1 + nu *  (2 * np.random.rand(2) - 1) )
            state[self.dof + 2*i: self.dof + 2*i + 2] = self.rotate2d(v, theta= i * theta)

        return state

    def get_trajectory(self, t_span=(0, 5), timescale=None, y0=None, noise_std=0., system="modlanet", **kwargs):
        if timescale is None:
            t_eval = None
        else:
            if isinstance(timescale, float) or isinstance(timescale, int):
                t_eval = np.linspace(t_span[0], t_span[1], int(1+timescale * (t_span[1]-t_span[0])))
            elif isinstance(timescale, np.ndarray):
                t_eval = timescale
            else:
                print('ALARM: The timescale should be float/int for np.linspace(t_span[0], t_span[1], int(timescale))'
                      'OR np.ndarray.')
                t_eval = None

        # get initial state
        self.m = kwargs['m'] if 'm' in kwargs else [1. for i in range(self.obj)]
        self.g = kwargs['g'] if 'g' in kwargs else 9.8

        # Initialization process requires that
        # the number of objects should be less than 10
        if y0 is None:
            y0 = self.random_config(system = system)

        dynamics_fn = self.dynamics_hamiltonian_fn
        energy_fn = self.hamiltonian_fn
        # if system == "hnn":
        #     dynamics_fn = self.dynamics_hamiltonian_fn
        #     energy_fn = self.hamiltonian_fn
        # else:
        #     dynamics_fn = self.dynamics_lagrangian_fn
        #     energy_fn = self.lagrangian_fn
        spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-12, **kwargs)
        dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        E = np.array([energy_fn(y, True) for y in spring_ivp['y'].T])

        if system == "hnn":
            # add noise
            x = spring_ivp['y'] + np.random.randn(*spring_ivp['y'].shape) * noise_std
            return x.T, dydt.T, spring_ivp['t'], E

        elif system == "modlanet":
            # self.dynamics_lagrangian_fn is somewhat slower
            # so we use self.dynamics_hamiltonian_fn to generate data
            # and do the calculate to get velocity and acceleration
            x = spring_ivp['y'][:self.dof]
            v = dydt[:self.dof]
            a = dydt[self.dof:]
            for i in range(self.obj):
                a[self.dim*i:self.dim*(i+1)] /= self.m[i]

            # add noise
            x += np.random.randn(*x.shape) * noise_std
            v += np.random.randn(*v.shape) * noise_std

            x = x.T
            v = v.T
            a = a.T

            return x, v, a, spring_ivp['t'], E

        else:
            raise ValueError('Unsupported coordinate system, choose'
                             ' system = \'hnn\' or \'modlanet\' instead.')

    def get_dataset(self, seed=0, samples=100, test_split=0.9, system='modlanet', timescale = 20, **kwargs):
        data = {'meta': locals()}
        np.random.seed(seed)

        if system == 'hnn':
            # randomly sample inputs
            xs, dxs, ts, Es = [], [], [], []
            for k in range(samples):
                x, dx, t, E = self.get_trajectory(system=system, timescale = timescale,  **kwargs)
                xs.append(x)
                dxs.append(dx)
                ts.append(t)
                Es.append(E)

            data['x'] = np.concatenate(xs)
            data['dx'] = np.concatenate(dxs).squeeze()
            data['t'] = np.expand_dims(np.concatenate(ts), axis=1)
            data['E'] = np.expand_dims(np.concatenate(Es), axis=1)

            # make a train/test split
            split_ix = int(len(data['x']) * test_split)
            split_data = {}
            for k in ['x', 'dx', 't', 'E']:
                split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
            data = split_data

        elif system == 'modlanet':
            # randomly sample inputs
            xs, vs, acs, ts, Es = np.empty((0, self.dof)), np.empty((0, self.dof)), np.empty((0, self.dof)), [], []
            for s in range(samples):
                x, v, ac, t, E = self.get_trajectory(system=system, timescale = timescale, **kwargs)
                xs = np.concatenate([xs, x]); ts = np.concatenate([ts, t])
                vs = np.concatenate([vs, v]); acs = np.concatenate([acs, ac])
                Es = np.concatenate([Es, E])

            data['x'] = xs;   data['t'] = np.expand_dims(ts, axis=1)
            data['ac'] = acs; data['v'] = vs
            data['E'] = np.expand_dims(Es, axis=1)

            # make a train/test split
            split_ix = int((data['x'].shape[0]) * test_split)
            split_data = {}
            for k in ['x', 'v', 'ac', 't', 'E']:
                split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
            data = split_data
            max_a = np.max(acs)


        else:
            raise ValueError('Unsupported system, choose'
                             ' system = \'hnn\' or \'modlanet\' instead.')
        return data