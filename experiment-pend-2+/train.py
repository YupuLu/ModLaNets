import torch, argparse
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os, sys
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TKAgg')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR+'/src')

from layers import MLP
from models import HNN, ModLaNet, LNN
from data import Dataset
from utils import L2_loss


def get_args():
    parser = argparse.ArgumentParser(description=None)
    # MODEL SETTINGS
    parser.add_argument('--model', default='modlanet', type=str,
                        help='Select model to train, either \'modlanet\', \'lnn\', \'hnn\', or \'baseline\' currently')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--obj', default=2, type=int, help='number of elements')
    parser.add_argument('--dof', default=1, type=int, help='degree of freedom')
    parser.add_argument('--dim', default=2, type=int, help='space dimension, 2D or 3D')
    parser.add_argument('--learn_rate', default=1e-2, type=float, help='learning rate')
    # For LNN/HNN/Baseline
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    # For ModLaNet
    parser.add_argument('--energy_hidden_dim', default=50, type=int, help='hidden dimension of mlp for engergies')
    parser.add_argument('--energy_nonlinearity', default='identity', type=str, help='neural net nonlinearity for engergies')
    parser.add_argument('--trans_hidden_dim', default=16, type=int, help='hidden dimension of mlp for transformations')
    parser.add_argument('--trans_nonlinearity', default='tanh', type=str, help='neural net nonlinearity for transformations')
    
    # TRAINING SETTINGS
    parser.add_argument('--gpu', nargs='?', const=True, default=False, help='try to use gpu?')
    parser.add_argument('--overwrite', nargs='?', const=True, default=False, help='overwrite the saved model.')
    parser.add_argument('--load_obj', default=2,  type=int, help='load saved model for the corresponding system, only works for ModLaNet.')    
    parser.add_argument('--load_epoch', default=0000, type = int, help='load saved model at steps k.')
    parser.add_argument('--end_epoch',  default=10000, type=int, help='end of training epoch')
    parser.add_argument('--use_lr_scheduler', default=True, help='whether to use lr_scheduler.')
    parser.add_argument('--samples', default=100, type=int, help='the number of sampling trajectories')
    parser.add_argument('--noise', default=0., type=float, help='the noise amplitude of the data')
    # GENERAL SETTINGS
    parser.add_argument('--save_dir', default=THIS_DIR+'/data', type=str, help='where to save the trained model')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--plot', default=False, action='store_true', help='plot training and testing loss?')
    parser.add_argument('--verbose', dest='verbose', default=True, action='store_true', help='verbose?')
    parser.add_argument('--print_every', default=100, type=int, help='number of gradient steps between prints')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.set_defaults(feature=True)
    return parser.parse_known_args()


def train_HNN(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.model == 'baseline' else "Training HNN model:")

    input_dim = args.obj * args.dof * 2
    output_dim = input_dim if args.model == 'baseline' else 2
    nn_model = MLP(input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline= (args.model == 'baseline') )
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    scheduler_enc = MultiStepLR(optim, milestones=[3000, 9000, 15000], gamma=0.5)

    def count_parameters(model0):
        return sum(p.numel() for p in model0.parameters() if p.requires_grad)
    print('number of parameters in model: ', count_parameters(model) )

    # load trained models if possible
    # naming example: model-2-pend-hnn-hidden_dim-200-end_epoch-10000-noise-0.0-learn_rate-0.001.tar
    start_epoch = 0
    if args.load_epoch > 0:
        path = '{}/model-{}-{}-{}-hidden_dim-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.obj, args.name,
                                                                           args.model, args.hidden_dim, args.load_epoch,
                                                                           args.noise, args.learn_rate)
        if os.path.exists(path):
            print('load model: {}'.format(path))
            checkpoint = torch.load(path)
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['network_state_dict'])
            scheduler_enc.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = args.load_epoch

    # arrange data
    # naming example: dataset_2_pend_hnn_noise_0.0.npy
    t0 = time.time()
    filename = args.save_dir + '/dataset_' + str(args.obj) + '_' + args.name + '_hnn_noise_' + str(args.noise) + '.npy'
    if os.path.exists(filename):
        print('Start loading dataset.')
        data = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        dataset = Dataset(obj=args.obj, m=[1 for i in range(args.obj)], l = [1 for i in range(args.obj)])
        data = dataset.get_dataset(seed=args.seed, system='hnn', noise_std=args.noise, samples = args.samples)
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        np.save(filename, data)

    x = data['x']
    x[:, :int(x.shape[1]/2)] = x[:, :int(x.shape[1]/2)] % (2 * np.pi)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dx'])

    test_x = data['test_x']
    test_x[:, :int(test_x.shape[1] / 2)] = test_x[:, :int(test_x.shape[1] / 2)] % (2 * np.pi)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32)
    test_dxdt = torch.Tensor(data['test_dx'])
    
    if args.verbose:
        print('DataObtainingTime: {}'.format(time.time() - t0))
        t0 = time.time()

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(start_epoch, args.end_epoch + 1):

        # train step
        dxdt_hat = model.time_derivative(x)
        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)
        if args.use_lr_scheduler:
            scheduler_enc.step()
        optim.zero_grad()

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())

        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt[:, int(test_x.shape[1] / 2):] - train_dxdt_hat[:, int(test_x.shape[1] / 2):]) ** 2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt[:, int(test_x.shape[1] / 2):] - test_dxdt_hat[:, int(test_x.shape[1] / 2):]) ** 2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))
    
    if args.plot:
        fig = plt.figure()
        plt.semilogy((stats['train_loss']), 'b')
        plt.semilogy((stats['test_loss']), 'r')
        plt.show()
        path = '{}/fig-{}-{}-{}-hidden_dim-{}-start_epoch-{}-end_epoch-{}-noise-{}-learn_rate-{}.{}'.format(args.save_dir, args.obj, args.name,
                                                                        args.model, args.hidden_dim, args.load_epoch, args.end_epoch, args.noise,
                                                                        args.learn_rate, 'png')
        fig.savefig(path)

    return model, stats, optim, scheduler_enc



def train_LNN(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    print("Training LNN model:")

    # if gpu is to be used
    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print('device:{}'.format(device))

    # init model and optimizer
    input_dim = args.obj * args.dof * 2
    model = LNN(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=1, nonlinearity='softplus', device=device)

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    lambda1 = lambda epoch: 1 #if epoch < 800 else 0.4 if epoch < 3000 else 0.2 if epoch < 6000 else 0.1 if epoch < 9000 else 0.04 if epoch < 10000 else 0.02 if epoch < 11000 else 0.01
    scheduler_enc = LambdaLR(optim, lr_lambda=lambda1)

    def count_parameters(model0):
        return sum(p.numel() for p in model0.parameters() if p.requires_grad)
    print('number of parameters in model: ', count_parameters(model) )

    # load trained models if possible
    # Example: model-2-pend-lnn-hidden_dim-600-end_epoch-10000-noise-0.0-learn_rate-0.001.tar
    start_epoch = 0
    if args.load_epoch > 0:
        path = '{}/model-{}-{}-epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.name,
                                                                           args.model, args.load_epoch,
                                                                           args.noise, args.learn_rate)
        if os.path.exists(path):
            print('load model: {}'.format(path))
            checkpoint = torch.load(path)
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['network_state_dict'])
            scheduler_enc.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = args.load_epoch
            if torch.cuda.is_available() and args.gpu:
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    # generate data
    # naming example: dataset_2_pend_modlanet_noise_0.0.npy
    t0 = time.time()
    filename = args.save_dir + '/dataset_' + str(args.obj) + '_' + args.name + '_modlanet_noise_' + str(args.noise) + '.npy'
    if os.path.exists(filename):
        print('Start loading dataset.')
        data = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        dataset = Dataset(obj=args.obj, m=[1 for i in range(args.obj)], l = [1 for i in range(args.obj)])
        data = dataset.get_dataset(seed=args.seed, system='modlanet', noise_std=args.noise)#, samples = 2, t_span=(0, 10))
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        np.save(filename, data)

    model.to(device=device)
    
    # arrange data
    x = data['x'] % (2 * np.pi)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32, device=device)
    v = torch.tensor(data['v'], requires_grad=True, dtype=torch.float32, device=device)
    a = torch.tensor(data['ac'], dtype=torch.float32, device=device)

    test_x = data['test_x'] % (2 * np.pi)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32, device=device)
    test_v = torch.tensor(data['test_v'], requires_grad=True, dtype=torch.float32, device=device)
    test_a = torch.tensor(data['test_ac'], device=device)

    if args.verbose:
        print('DataObtainingTime: {}'.format(time.time() - t0))

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(start_epoch, args.end_epoch + 1):
        # train step
        model.train()
        a_hat = model.forward(x, v)
        loss = L2_loss(a_hat, a)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        a_hat = model.forward(test_x, test_v)
        test_loss = L2_loss(test_a, a_hat)
        if args.use_lr_scheduler:
            scheduler_enc.step()
        optim.zero_grad()

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.forward(x, v)
    train_dist = (a - train_dxdt_hat) ** 2
    test_dxdt_hat = model.forward(test_x, test_v)
    test_dist = (test_a - test_dxdt_hat) ** 2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    if args.plot:
        fig = plt.figure()
        plt.semilogy((stats['train_loss']), 'b')
        plt.semilogy((stats['test_loss']), 'r')
        plt.legend(['train_loss', 'test_loss'])
        plt.show()
        path = '{}/fig-{}-{}-{}-hidden_dim-{}-start_epoch-{}-end_epoch-{}-noise-{}-learn_rate-{}.{}'.format(args.save_dir, args.obj, args.name,
                                                                        args.model, args.hidden_dim, args.load_epoch, args.end_epoch, args.noise,
                                                                        args.learn_rate, 'png')
        fig.savefig(path)

    return model, stats, optim, scheduler_enc


def BuildComputationTree(transform = 'local', obj=2, dim=2, dof=1, device = 'cpu'):
    r"""
    User defined relations between elements and origins of local coordinate systems.
    We assume that these connections between origins and elements 
    can be obtained when local coordinate systems are constructed.

    Function: x_{i, origin} =   \sum_j x_{j, element} * weight_j + 
                                \sum_k x_{k, origin}  * weight_k + 
                                weight_c
    Form: [i, 
            [[j1, weight_j1], [j2, weight_j2], ...], 
            [[k1, weight_k1], [k2, weight_k2], ...], 
            weight_c]
    """
    tree = None
    if transform == 'local':
        ################################
        # User defined relations start #
        ################################
        ele  = [0, [], [], torch.zeros((dim), device = device)]
        tree = [ele]
        for i in range(1, obj):
            ele  = [i, [[i-1, 1.]], [], torch.zeros((dim), device = device)]
            tree.append(ele)
        ################################
        # User defined relations end   #
        ################################
    return tree

def train_ModLaNet(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    print("Training ModLaNet model:")

    # if gpu is to be used
    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print('device:{}'.format(device))

    model = ModLaNet(obj=args.obj, dim=args.dim, edof=args.dof, device=device, build_computation_tree=BuildComputationTree, transform = 'local',
                        trans_hidden_dim = args.trans_hidden_dim, trans_nonlinearity = args.trans_nonlinearity, 
                        energy_hidden_dim = args.energy_hidden_dim)

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    lambda1 = lambda epoch: 1 if epoch < 2000 else 0.4 if epoch < 5000 else 0.2 if epoch < 7000 else 0.1 if epoch < 9000 else 0.04 #if epoch < 10000 else 0.02 if epoch < 11000 else 0.01
    scheduler_enc = LambdaLR(optim, lr_lambda=lambda1)

    def count_parameters(model0):
        return sum(p.numel() for p in model0.parameters() if p.requires_grad)
    print('number of parameters in model: ', count_parameters(model) )

    # load trained models if possible
    # Example: model-2-pend-modlanet-hidden_dim-50-end_epoch-3000-noise-0.0-learn_rate-0.001.tar
    start_epoch = 0
    if args.load_epoch > 0:
        path = '{}/model-{}-{}-{}-hidden_dim-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.load_obj, args.name,
                                                                           args.model, args.energy_hidden_dim, args.load_epoch,
                                                                           args.noise, args.learn_rate)
        
        if os.path.exists(path):
            print('load model: {}'.format(path))
            checkpoint = torch.load(path)
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['network_state_dict'])
            scheduler_enc.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = args.load_epoch
            if torch.cuda.is_available() and args.gpu:
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    # arrange data
    t0 = time.time()
    filename = args.save_dir + '/dataset_' + str(args.obj) + '_' + args.name + '_modlanet_noise_' + str(args.noise) + '.npy'
    if os.path.exists(filename):
        print('Start loading dataset.')
        data = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        dataset = Dataset(obj=args.obj, m=[1 for i in range(args.obj)], l = [1 for i in range(args.obj)])
        data = dataset.get_dataset(seed=args.seed, system='modlanet', noise_std=args.noise, samples = args.samples)
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        np.save(filename, data)

    model.to(device=device)
    x = data['x'] % (2 * np.pi)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32, device=device)
    v = torch.tensor(data['v'], requires_grad=True, dtype=torch.float32, device=device)
    a = torch.tensor(data['ac'], dtype=torch.float32, device=device)

    test_x = data['test_x'] % (2 * np.pi)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32, device=device)
    test_v = torch.tensor(data['test_v'], requires_grad=True, dtype=torch.float32, device=device)
    test_a = torch.tensor(data['test_ac'], device=device)

    if args.verbose:
        print('DataObtainingTime: {}'.format(time.time() - t0))
    
    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(start_epoch, args.end_epoch + 1):
        # train step
        model.train()
        a_hat = model.forward(x, v)
        loss = L2_loss(a, a_hat)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        a_hat = model.forward(test_x, test_v)
        test_loss = L2_loss(test_a, a_hat)
        if args.use_lr_scheduler:
            scheduler_enc.step()
        optim.zero_grad()

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.forward(x, v)
    train_dist = (a - train_dxdt_hat) ** 2
    test_dxdt_hat = model.forward(test_x, test_v)
    test_dist = (test_a - test_dxdt_hat) ** 2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    if args.plot:
        fig = plt.figure()
        plt.semilogy((stats['train_loss']), 'b')
        plt.semilogy((stats['test_loss']), 'r')
        plt.show()
        path = '{}/fig-{}-{}-{}-hidden_dim-{}-start_epoch-{}-end_epoch-{}-noise-{}-learn_rate-{}.{}'.format(args.save_dir, args.obj, args.name,
                                                                        args.model, args.energy_hidden_dim, args.load_epoch, args.end_epoch, args.noise,
                                                                        args.learn_rate, 'png')
        fig.savefig(path)                                                  

    return model, stats, optim, scheduler_enc

def main():
    args = get_args()[0]
    model, stats = None, None
    
    # Check whether model exists
    # Example: model-2-pend-modlanet-hidden_dim-50-end_epoch-3000-noise-0.0-learn_rate-0.001.tar
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    args.hidden_dim = args.energy_hidden_dim if args.model == 'modlanet' else args.hidden_dim
    path = '{}/model-{}-{}-{}-hidden_dim-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.obj, args.name,
                                                                           args.model, args.hidden_dim, args.end_epoch,
                                                                           args.noise, args.learn_rate)
    if os.path.exists(path):
        if args.overwrite:
            print('Model already exist, overwrite it.')
        else:
            raise ValueError('Trained model \'{}\' already exists. '
                             'For overwrite, please use --overwrite'.format(path))

    if args.model == 'modlanet':
        model, states, optim, scheduler = train_ModLaNet(args)
    elif args.model == 'lnn':
        model, states, optim, scheduler = train_LNN(args)
    elif args.model == 'hnn':
        model, states, optim, scheduler = train_HNN(args)
    elif args.model == 'baseline':
        model, states, optim, scheduler = train_HNN(args)
    else:
        raise ValueError('Model \'{}\' is not implemented'.format(args.model))

    # save
    torch.save({'network_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, path)
    training_process = '{}/train_process-{}-{}-{}-hidden_dim-{}-start_epoch-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.obj, args.name,
                                                                           args.model, args.hidden_dim, args.load_epoch, args.end_epoch,
                                                                           args.noise, args.learn_rate)
    if os.path.exists(training_process):
        states0 = np.load(training_process, allow_pickle=True).item()
        states0['train_loss'] = np.append(states0['train_loss'], states['train_loss'])
        states0['test_loss'] = np.append(states0['test_loss'], states['test_loss'])
        np.save(training_process, states)
    else:
        np.save(training_process, states)

if __name__ == "__main__":
    main()
