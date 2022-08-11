import torch
import numpy as np
from utils import choose_nonlinearity


class PotentialEnergyCell(torch.nn.Module):
    """Regress the potential energy using a MLP"""
    def __init__(self, input_dim = 2, hidden_dim = 100, output_dim = 1, nonlinearity = 'tanh'):
        super(PotentialEnergyCell, self).__init__()
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim
        self.nonlinearity = nonlinearity
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, nonlinearity=nonlinearity)

    def forward(self, x):
        y = self.mlp(x)
        return y

class GlobalPositionTransform(torch.nn.Module):
    """Doing coordinate transformation using a MLP"""
    def __init__(self, input_dim = 2, hidden_dim = 100, output_dim = 1, nonlinearity = 'tanh'):
        super(GlobalPositionTransform, self).__init__()
        self.output_dim = output_dim
        self.mlp = MLP(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, nonlinearity=nonlinearity)

    def forward(self, x, x_0):
        y = self.mlp(x) + x_0
        return y

class GlobalVelocityTransform(torch.nn.Module):
    """Doing coordinate transformation using a MLP"""
    def __init__(self, input_dim = 2, hidden_dim = 100, output_dim = 1, nonlinearity = 'tanh'):
        super(GlobalVelocityTransform, self).__init__()
        self.output_dim = output_dim
        self.mlp = MLP(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, nonlinearity=nonlinearity)

    def forward(self, x, v, v_0):
        y = self.mlp(x) * v + v_0
        return y

class MLP(torch.nn.Module):
    """Just a salt-of-the-earth MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', init = 'xavier'):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

        # This initialization is important.
        if init == 'xavier':
            for l in [self.linear1, self.linear2, self.linear3]:
                torch.nn.init.xavier_normal_(l.weight)  # use a principled initialization
        elif init == 'lnn':
            for i, l in enumerate([self.linear1, self.linear2, self.linear3]):
                lnn_init(l.weight, i, 3)
        else:
            raise ValueError('Unsupported init function. Please update it by your own.')

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)

class MLPAutoencoder(torch.nn.Module):
    """A salt-of-the-earth MLP Autoencoder + some edgy res connections"""

    def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
        super(MLPAutoencoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

        self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

        for l in [self.linear1, self.linear2, self.linear3, self.linear4, \
                  self.linear5, self.linear6, self.linear7, self.linear8]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def encode(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = h + self.nonlinearity(self.linear2(h))
        h = h + self.nonlinearity(self.linear3(h))
        return self.linear4(h)

    def decode(self, z):
        h = self.nonlinearity(self.linear5(z))
        h = h + self.nonlinearity(self.linear6(h))
        h = h + self.nonlinearity(self.linear7(h))
        return self.linear8(h)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

def lnn_init(tensor: torch.Tensor, current_num_of_layer: int, num_of_layer: int, seed=0) -> torch.Tensor:
    r"""Ref: Lagrangian Neural Networks Appendix C.
    Do an optimized LNN initialization for a simple uniform-width MLP.

    Args:
    tensor: an n-dimensional `torch.Tensor`
    current_num_of_layer: current num of the layer of this MLP
    num_of_layer: all num of the layer of this MLP

    """
    torch.manual_seed(seed)
    assert(tensor.dim() == 2)
    a, b = tensor.shape
    i, n = current_num_of_layer, num_of_layer
    with torch.no_grad():
        if i == 0:
            tensor.normal_(0, 2.2/np.sqrt(b) )
        elif i == (n-1):
            tensor.normal_(0, np.sqrt(a) )
        else:
            tensor.normal_(0, 0.58*(i+1)/np.sqrt((a+b)/2) )
    return tensor
