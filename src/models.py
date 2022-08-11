import torch
import torch.nn as nn
import numpy as np
import layers
import time
from utils import rk4

class ModLaNet(torch.nn.Module):
    r"""
    ModLaNet network to construct dynamics, containing three parts:
    1. Coordinate transformation: Transform each element's state to 
        the global Cartesian coordinate system, the calculation 
        sequence is based on a defined computation tree;
    2. Energy construction: Construct kinetic and potential energies 
        for each element to obtain the system's Lagrangian.
    3. Dynamics derivation: Build dynamical function for the system 
        using the Eular-Lagrange Equation.
    """
    def __init__(self, obj = 2, dim = 2, edof = 1, device = 'cpu', build_computation_tree = None,
            trans_nonlinearity = 'identity', trans_hidden_dim = 16, transform = 'global',
            energy_nonlinearity = 'identity', energy_hidden_dim = 50):
        r"""
        Initialise the ModLaNet.

        Parameters and Variables
        :param device: compute on CPU or GPU. Default to 'cpu'
        :param obj: number of elements
        :param dim: element's input dimention
        :param edof: element's degree of freedom
        :var   sdof: system's degree of freedom
        :var   mass: system's mass matrix
        
        :param trans_hidden_dim: hidden dimension for the transformation networks
        :param trans_nonlinearity: nonlinearity function chosen for the transformation networks 
        :param transform: decide whether the origin coordinate system, 'local' or 'global'.
                          if 'global', then the transformation networks will deprecate to 
                          identity layers. Default to 'local'.
        :param computation_tree: a computation tree defining squence of transformaiton 
                                among elements and local coordinate systems' origins.
        
        :param energy_hidden_dim: hidden dimension for the energy networks
        :param energy_nonlinearity: nonlinearity function chosen for the energy networks 
        """
        super(ModLaNet, self).__init__()
        self.device = device
        self.obj = obj
        self.dim = dim
        self.edof = edof
        self.sdof = obj * edof
        
        self.transform = transform
        self.trans_nonlinearity = trans_nonlinearity
        self.energy_nonlinearity = energy_nonlinearity
        self.build_computation_tree = build_computation_tree
        self.computation_tree = None if self.build_computation_tree is None else self.build_computation_tree(transform=self.transform, obj=self.obj, dim=self.dim, dof=self.edof, device = self.device)
        self.Global4x = layers.GlobalPositionTransform(input_dim=self.edof, hidden_dim=trans_hidden_dim, output_dim=self.dim, nonlinearity = trans_nonlinearity)
        self.Global4v = layers.GlobalVelocityTransform(input_dim=self.edof, hidden_dim=trans_hidden_dim, output_dim=self.dim, nonlinearity = trans_nonlinearity)
        
        self.Potential1 = layers.PotentialEnergyCell(input_dim=self.dim, hidden_dim=energy_hidden_dim, output_dim=1, nonlinearity=energy_nonlinearity)
        self.Potential2 = layers.PotentialEnergyCell(input_dim=self.dim * 2, hidden_dim=energy_hidden_dim, output_dim=1, nonlinearity=energy_nonlinearity)
        self.co1 = torch.nn.Parameter(torch.ones(1, device=device)*0.5)
        self.co2 = torch.nn.Parameter(torch.ones(1, device=device)*0.5)

        self.mass = torch.nn.Linear(1, 1, bias=False)
        for l in [self.mass]:
            torch.nn.init.ones_(l.weight)

    def forward(self, x, v, create_graph=True):
        r"""
        Shapes for some variables:
        x:     (batch, self.sdof), v:   (batch, self.sdof)
        dvL:   (batch, self.sdof), dxL: (batch, self.sdof)
        dvdvL: (batch, self.sdof, self.sdof)
        dxdvL: (batch, self.sdof, self.sdof) in which dx[i]dvL = (batch, i, self.sdof)
        """
        L, T, U = 0., 0., torch.zeros((x.shape[0], 1), dtype = torch.float32, device=self.device)

        # Transformation to global coordinates
        if self.transform == 'local':
            x_global = torch.zeros((x.shape[0], self.dim * (self.obj)), device=self.device)
            x_origin = torch.zeros((x.shape[0], self.dim * (self.obj)), device=self.device)
            
            v_global = torch.zeros((v.shape[0], self.dim * (self.obj)), device=self.device)
            v_origin = torch.zeros((v.shape[0], self.dim * (self.obj)), device=self.device)
            
            for (i, weight_a, weight_b, weight_c) in self.computation_tree:
                for j, weight in weight_a:
                    x_origin[:, (i) * self.dim: (i + 1) * self.dim] += weight * x_global[:, (j) * self.dim: (j + 1) * self.dim]
                    v_origin[:, (i) * self.dim: (i + 1) * self.dim] += weight * v_global[:, (j) * self.dim: (j + 1) * self.dim]
                for j, weight in weight_b:
                    x_origin[:, (i) * self.dim: (i + 1) * self.dim] += weight * x_origin[:, (j) * self.dim: (j + 1) * self.dim]
                    v_origin[:, (i) * self.dim: (i + 1) * self.dim] += weight * v_origin[:, (j) * self.dim: (j + 1) * self.dim]
                x_origin[:, (i) * self.dim: (i + 1) * self.dim] += weight_c
                
                x_global[:, (i) * self.dim: (i + 1) * self.dim] = self.Global4x(x[:, i * self.edof: (i + 1) * self.edof],
                                                                            x_origin[:, i * self.dim: (i + 1) * self.dim])
                v_global[:, (i) * self.dim: (i + 1) * self.dim] = self.Global4v(x[:, i * self.edof: (i + 1) * self.edof],
                                                                            v[:, i * self.edof: (i + 1) * self.edof],
                                                                            v_origin[:, i * self.dim: (i + 1) * self.dim]) 
        else:
            x_global = x
            v_global = v

        # Calculate the potential energy for i-th element
        for i in range(self.obj):
            U  += self.co1 * self.mass(self.Potential1(x_global[:, i * self.dim: (i+1) * self.dim]))
        
        for i in range(self.obj):
            for j in range(i):
                x_ij = torch.cat([x_global[:, i * self.dim: (i + 1) * self.dim], x_global[:, j * self.dim: (j + 1) * self.dim]], dim=1)
                x_ji = torch.cat([x_global[:, j * self.dim: (j + 1) * self.dim], x_global[:, i * self.dim: (i + 1) * self.dim]], dim=1)
                U += self.co2 * (0.5 * self.mass(self.Potential2(x_ij)) + 0.5 * self.mass(self.Potential2(x_ji)))
         
        # Calculate the kinetic energy for i-th element
        for i in range(self.obj):
            T += 0.5 * self.mass(v_global[:, (i) * self.dim: (i + 1) * self.dim].pow(2).sum(axis=1, keepdim=True))
        
        # Construct Lagrangian
        L += (T - U)

        # Calculate derivatives with Lagrangian Equation
        dvL = torch.autograd.grad(L.sum(), v, create_graph=create_graph)[0]
        dxL = torch.autograd.grad(L.sum(), x, create_graph=create_graph)[0]
        dvdvL = torch.zeros((x.shape[0], self.sdof, self.sdof), dtype =torch.float32, device=self.device)
        dxdvL = torch.zeros((x.shape[0], self.sdof, self.sdof), dtype =torch.float32, device=self.device)
        for i in range(self.sdof):
            dvidvL = torch.autograd.grad(dvL[:, i].sum(), v, create_graph=create_graph, allow_unused=True)[0]
            if dvidvL is None:
                break
            else:
                dvdvL[:,i,:] += dvidvL
        for i in range(self.sdof):
            dxidvL = torch.autograd.grad(dvL[:, i].sum(), x, create_graph=create_graph, allow_unused=True)[0]
            if dxidvL is None:
                break
            else:
                dxdvL[:, i, :] += dxidvL

        # in version 1.8.1 you can use torch.linalg.inv() to replace torch.inverse()
        # (batch, self.sdof, self.sdof) * ( (self.sdof, batch) - (self.sdof, self.sdof, batch) * (self.sdof, batch) )
        inv = torch.inverse(dvdvL)
        a = torch.matmul(inv,  ( dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2)) )
        return a.squeeze(2)

    def update(self, obj = None, dim = None, edof = None, mass = None, build_computation_tree = None):
        self.obj = self.obj if obj is None else obj
        self.dim = self.dim if dim is None else dim
        self.edof = self.edof if edof is None else edof
        self.sdof = self.obj * self.edof
        self.build_computation_tree = self.build_computation_tree if build_computation_tree is None else build_computation_tree
        self.computation_tree = None if self.build_computation_tree is None else self.build_computation_tree(transform=self.transform, obj=self.obj, dim=self.dim, dof=self.edof, device = self.device)
    
class LNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='softplus', device='cpu'):
        super(LNN, self).__init__()
        self.device = device
        self.input_dim = int(input_dim / 2)
        self.model = layers.MLP(input_dim, hidden_dim, output_dim, nonlinearity = nonlinearity, init = 'lnn')

    def forward(self, x, v):
        create_graph = True
        y = torch.cat([x, v], dim=1)
        # Calculate Lagrangian
        L = self.model(y)

        # Calculate derivatives with Lagrangian Equation
        dvL = torch.autograd.grad(L.sum(), v, create_graph=create_graph)[0]
        dxL = torch.autograd.grad(L.sum(), x, create_graph=create_graph)[0]
        dvdvL = torch.zeros((x.shape[0], self.input_dim, self.input_dim), dtype=torch.float32, device=self.device)
        dxdvL = torch.zeros((x.shape[0], self.input_dim, self.input_dim), dtype=torch.float32, device=self.device)
        for i in range(self.input_dim):
            dvidvL = torch.autograd.grad(dvL[:, i].sum(), v, create_graph=create_graph, allow_unused=True)[0]
            if dvidvL is None:
                break
            else:
                dvdvL[:, i, :] += dvidvL
        for i in range(self.input_dim):
            dxidvL = torch.autograd.grad(dvL[:, i].sum(), x, create_graph=create_graph, allow_unused=True)[0]
            if dxidvL is None:
                break
            else:
                dxdvL[:, i, :] += dxidvL

        # in version 1.8.1 you can use torch.linalg.inv() to replace torch.inverse()
        inv = torch.inverse(dvdvL)
        a = torch.matmul(inv, (dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2)))
        return a.squeeze(2)


class HNN(torch.nn.Module):
    """Learn arbitrary vector fields that are sums of conservative and solenoidal fields"""
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1, 1)

    def time_derivative(self, x, t=None, separate_fields=False):
        t0 = time.time()
        """NEURAL ODE-STLE VECTOR FIELD"""
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(x) # traditional forward pass
        # print('H Time: {}'.format((time.time() - t0)))
        # t0 = time.time()

        conservative_field = torch.zeros_like(x) # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]
        # print('dH Time: {}'.format((time.time() - t0)))

        return conservative_field + solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1

            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M

    # Saved for further implementation
    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)
