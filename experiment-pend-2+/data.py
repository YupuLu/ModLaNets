import autograd
import autograd.numpy as np

import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp


class Dataset:
    def __init__(self, obj, m = None, l = None, **kwargs):
        self.m = m
        self.g = 9.8
        self.l = l
        self.obj = obj
        self.dim = 1
        self.dof = obj * self.dim # degree of freedom
        self.samples   = 1

    def dynamics_hamiltonian_fn(self, t, coords):
        # From "The double pendulum: Hamiltonian formulation"
        # https://diego.assencio.com/?index=e5ac36fcb129ce95a61f8e8ce0572dbf
        q1, q2, p1, p2 = coords
        l1, l2, m1, m2 = self.l[0], self.l[1], self.m[0], self.m[1]
        g = self.g
        b = l1*l2*(m1+m2*np.sin(q1-q2)**2)
        dq1 = (l2*p1 - l1*p2*np.cos(q1-q2)) / (b*l1)
        dq2 = (-m2*l2*p1*np.cos(q1-q2) + (m1+m2)*l1*p2) / (m2*b*l2)
        h1 = p1*p2*np.sin(q1-q2) / b
        h2 = (m2*l2**2*p1**2 + (m1+m2)*l1**2*p2**2 - 2*m2*l1*l2*p1*p2*np.cos(q1-q2)) / (2*b**2)
        dp1 = -(m1+m2)*g*l1*np.sin(q1) - h1 + h2*np.sin(2*(q1-q2))
        dp2 = -m2*g*l2*np.sin(q2)      + h1 - h2*np.sin(2*(q1-q2))
        return np.stack([dq1, dq2, dp1, dp2])

    def hamiltonian_fn(self, coords, eng = True):
        q1, q2, p1, p2 = np.split(coords, 4)  # q is angle, p is angular momentum.
        l1, l2, m1, m2 = self.l[0], self.l[1], self.m[0], self.m[1]
        H = (m1 + m2) * self.g * l1 * (-np.cos(q1)) + m2 * self.g * l2 * (-np.cos(q2)) \
            + ((m1 + m2) * l1**2 * p2**2 + m2 * l2**2 * p1**2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2)) / \
                    (2 * m2 * (l1 ** 2) * (l2 ** 2) * (m1 + m2 * np.sin(q1 - q2)**2))
        # double pendulum hamiltonian
        return H

    def hamiltonian_kinetic(self, coords):
        assert(len(coords) == self.dof * 2)
        for i in range(self.obj):
            coords[self.dof+i] = coords[self.dof+i] / (self.m[i]*self.l[i]**2)
        T = 0.
        vx, vy = 0., 0.
        for i in range(self.obj):
            vx = vx + self.l[i] * coords[self.dof+i] * np.cos(coords[i])
            vy = vy + self.l[i] * coords[self.dof+i] * np.sin(coords[i])
            T  = T  + 0.5 * self.m[i] * (np.power(vx, 2) + np.power(vy, 2))
        return T

    def hamiltonian_potential(self, coords):
        assert(len(coords) == self.dof * 2)
        g = self.g
        U = 0.
        y = 0.
        for i in range(self.obj):
            y  = y  - self.l[i] * np.cos(coords[i])
            U  = U  + self.m[i] * g * y
        return U

    def dynamics_lagrangian_fn(self, t, coords):
        return self.dynamics_lagrangian_fn2(coords)

    def dynamics_lagrangian_fn2(self, coords):
        grad_lag = autograd.grad(self.lagrangian_fn)
        jaco_lag = autograd.jacobian(grad_lag)
        grad = grad_lag(coords)
        jaco = jaco_lag(coords)
        size = int(len(coords)/2)
        g = np.linalg.inv(jaco[size:, size:]) @ (grad[:size] - jaco[size:, :size] @ coords[size:])
        return np.append(coords[size:], g)
    
    def lagrangian_fn(self, coords, eng = False):
        assert(len(coords) == self.dof * 2)
        g = self.g
        U, T = 0., 0.
        vx, vy = 0., 0.
        y = 0.
        for i in range(self.obj):
            vx = vx + self.l[i] * coords[self.dof+i] * np.cos(coords[i])
            vy = vy + self.l[i] * coords[self.dof+i] * np.sin(coords[i])
            T  = T  + 0.5 * self.m[i] * (np.power(vx, 2) + np.power(vy, 2))
            y  = y  - self.l[i] * np.cos(coords[i])
            U  = U  + self.m[i] * g * y
        L = T - U if not eng else T + U
        return L

    def lagrangian_kinetic(self, coords):
        assert(len(coords) == self.dof * 2)
        T = 0.
        vx, vy = 0., 0.
        for i in range(self.obj):
            vx = vx + self.l[i] * coords[self.dof+i] * np.cos(coords[i])
            vy = vy + self.l[i] * coords[self.dof+i] * np.sin(coords[i])
            T  = T  + 0.5 * self.m[i] * (np.power(vx, 2) + np.power(vy, 2))
        return T

    def lagrangian_potential(self, coords):
        assert(len(coords) == self.dof * 2)
        g = self.g
        U = 0.
        y = 0.
        for i in range(self.obj):
            y  = y  - self.l[i] * np.cos(coords[i])
            U  = U  + self.m[i] * g * y
        return U

    def random_config(self, system='modlanet'):
        max_momentum = 10.
        y0 = np.zeros(self.obj * 2)
        for i in range(self.obj):
            theta = (2 * np.random.rand()) * np.pi
            momentum = (2 * np.random.rand() - 1) * max_momentum
            if system == "modlanet":
                momentum = momentum / (self.m[i] * self.l[i] ** 2)
            y0[i] = theta
            y0[i + self.obj] = momentum
        return y0

    def get_trajectory(self, t_span=(0, 10), timescale=None, y0=None, noise_std=0., system="modlanet", **kwargs):
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
        self.l = kwargs['l'] if 'l' in kwargs else [1. for i in range(self.obj)]
        self.g = kwargs['g'] if 'g' in kwargs else 9.8
        
        if y0 is None:
            y0 = self.random_config(system)

        if system == "hnn":
            dynamics_fn = self.dynamics_hamiltonian_fn
            energy_fn   = self.hamiltonian_fn
        else:
            dynamics_fn = self.dynamics_lagrangian_fn
            energy_fn   = self.lagrangian_fn
        spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-12, **kwargs)
        dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        E = np.array([energy_fn(y, True) for y in spring_ivp['y'].T])

        if system == "hnn":
            # add noise
            x = spring_ivp['y'] + np.random.randn(*spring_ivp['y'].shape) * noise_std
            return x, dydt.T, spring_ivp['t'], E

        elif system == "modlanet":
            x = spring_ivp['y'][:self.dof]
            v = spring_ivp['y'][self.dof:]
            a = dydt[self.dof:]

            # add noise
            x += np.random.randn(*x.shape) * noise_std
            v += np.random.randn(*v.shape) * noise_std

            x = x.T
            v = v.T
            a = a.T

            return x, v, a, spring_ivp['t'], E

        else:
            raise ValueError('Unsupported system system, choose'
                             ' system = \'hnn\' or \'modlanet\' instead.')

    def get_dataset(self, seed=0, samples=100, timescale = 10, test_split=0.9, system='modlanet', **kwargs):
        data = {'meta': locals()}
        self.samples = samples

        if system == 'hnn':
            # randomly sample inputs
            np.random.seed(seed)
            xs, dxs, ts, Es = [], [], [], []
            for s in range(self.samples):
                x, dx, t, E = self.get_trajectory(system=system, timescale= timescale, **kwargs)
                xs.append(x.T)
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
            np.random.seed(seed)
            xs, vs, acs, ts, Es = np.empty((0, self.dof)), np.empty((0, self.dof)), np.empty((0, self.dof)), [], []
            for s in range(self.samples):
                x, v, ac, t, E = self.get_trajectory(system=system, timescale= timescale, **kwargs)
                xs = np.concatenate([xs, x]); ts = np.concatenate([ts, t])
                vs = np.concatenate([vs, v]); acs = np.concatenate([acs, ac])
                Es = np.concatenate([Es, E])

            data['x'] = xs;    data['t'] = np.expand_dims(ts, axis=1)
            data['ac'] = acs;  data['v'] = vs
            data['E'] = np.expand_dims(Es, axis=1)

            # make a train/test split
            split_ix = int((data['x'].shape[0]) * test_split)
            split_data = {}
            for k in ['x', 'v', 'ac', 't', 'E']:
                split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
            data = split_data

        else:
            raise ValueError('Unsupported dynamic system, choose'
                             ' system = \'hnn\' or \'modlanet\' instead.')
        return data

