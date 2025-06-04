# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import torch
import math

from argparse import ArgumentParser
from distutils.util import strtobool

from Models import modelpool
from Preprocess import datapool

from funcs import seed_all, eval_ann, eval_snn, train, test, train_ann
from modules import TCL
from utils import regular_set
from utils import (isActivation, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d)
from misc import mkdir_p, save_checkpoint
from logger import Logger

from torch.autograd import Function
from torch.nn import CrossEntropyLoss, Module, Parameter

DEVICE = "cpu"

parser = ArgumentParser(description='PyTorch ANN-SNN Conversion')

# ## ANN or SNN
parser.add_argument(
    '--action',
    default='train',
    type=str,
    help='Action: train, or test/evaluate.',
    choices=['train', 'test', 'evaluate']
)
parser.add_argument(
    '--mode',
    default='ann',
    type=str, help='ANN training/testing, or SNN testing',
    choices=['ann', 'snn']
)
parser.add_argument(
    '--gpus', default=1, type=int, help='GPU number to use.'
)
parser.add_argument('--num_workers', default=4, type=int, help='num of workers to use.')
parser.add_argument('--batch_size', '-bs', default=128, type=int, help='Batchsize')
parser.add_argument('--lr', '-learning_rate', default=0.1, type=float, metavar='LR', help='Initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='Weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum')
parser.add_argument('--optimizer', default='SGD', type=str, help='which optimizer')
# parser.add_argument('--epochs', default=120, type=int, metavar='N', help='Number of total training epochs to run')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='Number of total training epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('-p', '--print-freq', default=20, type=int, metavar='N', help='Print frequency in training and testing (default: 20)')
parser.add_argument('--seed', default=42, type=int, help='Setting the random seed')

# ## Conversion method settings
parser.add_argument('--t', default=256, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--l', default=16, type=int, help='L Quantization steps')
# ## Properties from the new proposed method
parser.add_argument('--a_learnable',
                    type=lambda x: bool(strtobool(x)),
                    nargs='?', const=True, default=False, choices=[False, True],
                    help='Learnable or not, of the slope of proposed SlipReLU activation function')
parser.add_argument('--a', default=0.5, type=float, help='Slope of proposed SlipReLU activation function')
parser.add_argument('--shift1', default=0.0, type=float, help='The Shift of the threshold-ReLU function')
parser.add_argument('--shift2', default=0.5, type=float, help='The Shift of the Step function')

# ## Dataset and model
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--model', default='resnet18', type=str, help='Model architecture',
                    choices=[
                        'vgg16', 'resnet18', 'resnet20',
                        'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed',
                        'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])



# ## Save smodels and results ## Save Checkpoints
parser.add_argument(
    '--checkpoint',
    default='/l/users/haiyan.jiang/res_ann2snn',
    type=str,
    metavar='PATH',
    help='Path to save checkpoint models (default: checkpoint)'
)

parser.add_argument(
    '--resume', default='', type=str, metavar='PATH',
    help='path to latest checkpoint (default: none)'
)
parser.add_argument(
    '--name', default='', type=str, help='Model saved name'
)
parser.add_argument(
    '--result', default='AccRes', type=str, help='Results saved name'
)


args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

# ### Use CUDA on normal GPUs or CPU
use_cuda = False # torch.cuda.is_available()
args.gpus = args.gpus if use_cuda else 0


# ## Print some information
print(f'--a_learnable: {args.a_learnable}')
print(f'--shift1: {args.shift1} --shift2: {args.shift2}')
print(f'--seed: {args.seed}')

# ## These are hyper-parameters
args.checkpoint = args.checkpoint + f'_learn_{args.a_learnable}_shift1_{args.shift1}_shift2_{args.shift2}'
print(f'--checkpoint: {args.checkpoint}')

args.name = f'{args.dataset}_{args.model}_L_{args.l}_a_{args.a}_seed_{args.seed}'

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class MySlipReLU_(Module):
    def __init__(self, up=3., N=4, a=0., shift1=0., shift2=0., a_learnable=False):
        super().__init__()
        self.slip_relu = SlipReLU_.apply
        self.up = Parameter(torch.tensor(up), requires_grad=True)
        self.N = torch.tensor(N)
        self.shift1 = torch.tensor(shift1)
        self.shift2 = torch.tensor(shift2)
        self.a_learnable = a_learnable
        if self.a_learnable:
            self.learn = torch.tensor(0.1)
            self.a = Parameter(torch.tensor(a), requires_grad=True)
        else:
            self.a = torch.tensor(a)
            self.learn = None

    def forward(self, x):
        x = self.slip_relu(x, self.up, self.N, self.a, self.shift1, self.shift2, self.learn)
        return x

class MySlipReLU(Module):
    def __init__(self, up=8., N=32, a=0., shift1=0., shift2=0., a_learnable=False):
        """
        Parameters
        ----------
        up : TYPE, optional
            DESCRIPTION. The default is 8..
        N : TYPE, optional
            DESCRIPTION. The default is 32.
        a : TYPE, optional
            DESCRIPTION. The default is 0..
        shift1 : TYPE, optional
            DESCRIPTION. The default is 0..
        shift2 : TYPE, optional
            DESCRIPTION. The default is 0..
        a_learnable : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        For ytemp, if we use the torch.floor(), then it will not give the correct x.grad, up.grad

        """
        super().__init__()
        self.myfloor = GradFloor.apply
        self.up = Parameter(torch.tensor(up), requires_grad=True)
        self.N = N
        self.shift1 = shift1
        self.shift2 = shift2
        self.a_learnable = a_learnable
        if self.a_learnable:
            self.a = Parameter(torch.tensor(a), requires_grad=True)
        else:
            self.a = a

    def __common_forward(self, x):
        # ## Version 2
        x = x / self.up
        temp0 = torch.clamp(x + self.shift1/self.N , 0., 1.)
        # ### If use the torch.floor() wrong x.grad, wrong up.grad (NEVER use torch.floor() here )
        # ### ytemp = self.up * torch.clamp( 1/ self.N * torch.floor(self.N *x / self.up + self.shift2) , 0., 1.)  # [-shift2* up/N, up -shift2*up/N]
        ztemp = self.up * temp0
        temp1 = self.myfloor(self.N * x + self.shift2) / self.N  # [-shift2* up/N, up +shift1*up/N]
        temp2 = torch.clamp(temp1, 0., 1.)
        ytemp = self.up *temp2
        w = self.a * ztemp + (1-self.a) * ytemp
        return w

    def forward(self, x):
        assert 0 <= self.a <= 1.0
        if self.a_learnable:
            ### If defined as following,there is grad of a.
            self.a = Parameter(torch.tensor(0.0), requires_grad=True) if self.a < 0.0 else self.a
            self.a = Parameter(torch.tensor(1.0), requires_grad=True) if self.a > 1.0 else self.a
        return self.__common_forward(x)

def adjust_learning_rate(optimizer, epoch, T_max):
    """
    lr = 0.1; epochs = 120; T_max = int(epochs/4)
    y =  [0.5 * lr * (1 + math.cos(epoch/T_max * math.pi)) for epoch in range(epochs)]
    plt.plot(y)
    """
    global state
    state['lr'] = 0.5 * args.lr * (1 + math.cos(epoch/T_max * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']
    return optimizer


# ## opt method 3
def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_list = [50, 100, 140, 240]
    if epoch in lr_list:
        print('change the learning rate')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

def replace_activation_by_slip(net, t, a, shift1, shift2, a_learnable):
    for name, module in net._modules.items():
        key = module.__class__.__name__.lower()
        if hasattr(module, "_modules"):
            net._modules[name] = replace_activation_by_slip(
                module, t, a, shift1, shift2, a_learnable
            )
        if not isActivation(key):
            continue
        tcl = TCL()
        up = 8.0
        if hasattr(module, "up"):
            up = module.up.item()
        slip = MySlipReLU(up, t, a, shift1, shift2, a_learnable)
        if t == 0:
            net._modules[name] = tcl
        else:
            net._modules[name] = slip
    return net

best_acc1 = 0
is_best = 0

def main(args):
    global best_acc1
    global is_best
    # ## Set the seed
    seed_all(args.seed)
    if not os.path.isdir(args.checkpoint + '/' + args.dataset):
        mkdir_p(args.checkpoint + '/' + args.dataset)
    if not os.path.isdir(args.checkpoint + '/' + args.result):
        mkdir_p(args.checkpoint + '/' + args.result)
    # ## Preparing data and model
    l_tr, l_te = datapool(args.dataset, args.batch_size, args.num_workers)
    net = modelpool(args.model, args.dataset)
    net = replace_maxpool2d_by_avgpool2d(net)
    net = replace_activation_by_slip(
        net,
        args.l, args.a,
        args.shift1, args.shift2,
        args.a_learnable
    )
    net = net.to(DEVICE)

    crit = CrossEntropyLoss().to(DEVICE)
    if args.action == 'train':
        # ## Step 1: train the ann model
        best_acc1, net, logger = train_ann(l_tr, l_te, net, crit, args, state)
        if best_acc1 == 0:
            print('================ Failed training with current optimizer ================')
            return
        print(f'Training finished ==> {args.dataset} {args.name}')
        # ## Print the best accuracy
        print(f'Best acc:   {best_acc1} ')
        # ## Plot the loss and accuracy
        fig = logger.plot(['Train Loss', 'Test Loss'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_loss.pdf'))
        fig = logger.plot(['Train Acc1.', 'Test Acc1.'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_acc1.pdf'))
        fig = logger.plot(['Train Acc5.', 'Test Acc5.'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_acc5.pdf'))
        fig = logger.plot(['Learning Rate'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_LearningRate.pdf'))

        fname = os.path.join(args.checkpoint, args.result, args.name + '_loss_acc.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(logger.numbers, f)

        # ## Step 2: test ann
        print('Training Finished, now begin to evaluate the model ')
        print(f'Reloading model ==> {args.dataset} {args.name}')
        # ## MUST load the best model before test/eval
        checkpoint = torch.load(os.path.join(args.checkpoint, args.dataset, args.name + '_best.pth'))
        net.load_state_dict(checkpoint['state_dict'])
        net = net.to(DEVICE)
        # ## Step2: ANN evalue
        ann_acc, _ = eval_ann(l_te, net, crit, DEVICE)
        print('Accuracy of testing ANN: {:.4f}'.format(ann_acc))
        # ## save for every a_learnable, L, a, T, and its snn accuracy
        test_ann_acc = {
            "learn": args.a_learnable,
            "shift1": args.shift1, "shift2": args.shift2,
            "L": args.l,
            "a": args.a, "T": args.t,
            "ann acc": ann_acc}
        with open(os.path.join(args.checkpoint, args.result, 'test_ann_' + args.name + '.pkl'), 'wb') as f:
            pickle.dump(test_ann_acc, f)
        # ## Step3: SNN test
        net = replace_activation_by_neuron(net, shift=args.shift2)
        net = net.to(DEVICE)
        snn_acc = eval_snn(l_te, net, DEVICE, args.t)
        print('Accuracy of testing SNN: ', snn_acc)
        # ## save for every a_learnable, L, a, T, and its snn accuracy
        eval_snn_acc = {
            "learn": args.a_learnable,
            "shift1": args.shift1, "shift2": args.shift2,
            "L": args.l, "a": args.a, "T": args.t,
            "snn acc": snn_acc}
        with open(os.path.join(args.checkpoint, args.result, 'eval_snn_' + args.name + '.pkl'), 'wb') as f:
            pickle.dump(eval_snn_acc, f)
    elif args.action == 'test' or args.action == 'evaluate':
        print(f'Reloading model ==> {args.dataset} {args.name}')
        # MUST load the best model before test/eval
        checkpoint = torch.load(os.path.join(args.checkpoint, args.dataset, args.name + '_best.pth'))
        net.load_state_dict(checkpoint['state_dict'])
        net = net.to(DEVICE)
        if args.mode == 'snn':
            net = replace_activation_by_neuron(net, shift=args.shift2)
            net = net.to(DEVICE)
            snn_acc = eval_snn(l_te, net, DEVICE, args.t)
            print('Accuracy of testing SNN: ', snn_acc)
            eval_snn_acc = {
                "learn": args.a_learnable,
                "shift1": args.shift1, "shift2": args.shift2,
                "L": args.l, "a": args.a, "T": args.t,
                "snn acc": snn_acc}
            with open(os.path.join(args.checkpoint, args.result, 'eval_snn_' + args.name + '.pkl'), 'wb') as f:
                pickle.dump(eval_snn_acc, f)
        elif args.mode == 'ann':
            ann_acc, _ = eval_ann(l_te, net, crit, DEVICE)
            print('Accuracy of testing ANN: {:.4f}'.format(ann_acc))
            # ## save for every a_learnable, L, a, T, and its snn accuracy
            test_ann_acc = {
                "learn": args.a_learnable,
                "shift1": args.shift1, "shift2": args.shift2,
                "L": args.l,
                "a": args.a, "T": args.t,
                "ann acc": ann_acc}
            with open(os.path.join(args.checkpoint, args.result, 'test_ann_' + args.name + '.pkl'), 'wb') as f:
                pickle.dump(test_ann_acc, f)
        else:
            AssertionError('Unrecognized mode')
    else:
        AssertionError('Unrecognized action')


if __name__ == "__main__":
    main(args)
    print(f'--checkpoint: {args.checkpoint}')
    print(f'--name: {args.name}')
