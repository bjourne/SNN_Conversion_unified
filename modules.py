# -*- coding: utf-8 -*-
from torch import nn
from torch import Tensor
from torch.nn import Module, Parameter

import torch.nn.functional as F
import torch
from spikingjelly.clock_driven import neuron


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class ScaledNeuron(nn.Module):
    def __init__(self, scale=1., shift=0.5):
        """ shift here corresponds to shift2 in the main, default value=0.5"""
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.shift = shift
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)
    def forward(self, x):
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x)* self.shift)
        x = self.neuron(x)
        self.t += 1
        return x * self.scale
    def reset(self):
        self.t = 0
        self.neuron.reset()


class ShiftNeuron(nn.Module):
    def __init__(self, scale=1., alpha=1/50000):
        super().__init__()
        self.alpha = alpha
        self.vt = 0.
        self.scale = scale
        self.neuron = neuron.IFNode(v_reset=None)
    def forward(self, x):
        x = x / self.scale
        x = self.neuron(x)
        return x * self.scale
    def reset(self):
        if self.training:
            self.vt = self.vt + self.neuron.v.reshape(-1).mean().item()*self.alpha
        self.neuron.reset()
        if self.training == False:
            self.neuron.v = self.vt


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing. """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class TCL_OLD(nn.Module):
    """ Threshold ReLU, or Clipped ReLU with up"""
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.Tensor([4.]), requires_grad=True)
    def forward(self, x):
        # x = F.relu(x, inplace='True')
        x = F.relu(x)
        x = self.up - x
        # x = F.relu(x, inplace='True')
        x = F.relu(x)
        x = self.up - x
        return x


class TCL(Module):
    """ Threshold ReLU, or Clipped ReLU with learnable self.up (V_{th})"""
    def __init__(self):
        super().__init__()
        self.up = Parameter(Tensor([4.]), requires_grad=True)
    def forward(self, x):
        x = torch.clamp(x / self.up, 0., 1.) * self.up
        return x


class SlipReLU_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, up, N, a, shift1, shift2, learnable=None):
        ctx.save_for_backward(input, up, N, a, shift1, shift2, learnable)
        x = input.clone()
        ztemp = up * torch.clamp(x / up + shift1/N, 0., 1.)
        ytemp = up * torch.clamp( 1/ N * torch.floor(N * x / up + shift2) , 0., 1.)
        output = a * ztemp + (1-a) * ytemp
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, up, N, a, shift1, shift2, learnable = ctx.saved_tensors
        grad_input = grad_up = grad_N = grad_a = grad_shift1 = grad_shift2 = grad_learn = None

        x = input.clone()

        # temp1 = abs(x - up/2 + shift1* up / N) <= (up/2)  # [-shift1* up/N, up -shift1*up/N]
        temp1 = abs(x - up/2 ) <= (up/2) + shift1 * up / N # [-shift1* up/N, up +shift1*up/N]
        # temp2 = abs(x - up/2 + shift2* up / N) <= (up/2)  #  [-shift2* up/N, up -shift2*up/N]
        temp2 = abs(x - up/2 ) <= (up/2) + shift2 * up / N # [-shift2* up/N, up +shift2*up/N]

        shift = max(shift1, shift2)
        temp = abs(x - up/2 ) <=  ( (up/2) + shift * up / N) # [-shift* up/N, up +shift*up/N]

        if ctx.needs_input_grad[0]:
            # ### when use this, [-max(shift1, shift2), -min(shift1, shift2)] whould be dx=a, [min(shift1, shift2), max(shift1, shift2)] dx=a (not 1-a)
            # ### grad_input = grad_output * a * temp1.float() + grad_output *(1-a) * temp2.float() # set x.grad =1
            grad_input = grad_output * temp.float()  # set x.grad =1 always

        # ### grad of up
        if ctx.needs_input_grad[1]:
            z1 = torch.clamp(x / up + shift1/N, 0., 1.)
            y1 = torch.clamp( 1/ N * torch.floor(N * x / up + shift2) , 0., 1.)
            up1 = a * (z1 + temp1.float() * (-x / up) )
            up2 = (1-a) * (y1 + temp2.float() * (-x / up) )
            # ### grad_up = grad_output * (up1 + up2)  ### not precise
            # ### grad_up = grad_output *  temp.float() * (up1 + up2)  ### not precise
            grad_up = grad_output * (up1 * temp1.float() + up2 * temp2.float())

        # ### grad of a
        if learnable is not None and ctx.needs_input_grad[3]:
            z1 = torch.clamp(x / up + shift1/N, 0., 1.)
            y1 = torch.clamp( 1/ N * torch.floor(N * x / up + shift2) , 0., 1.)
            # grad_a = grad_output * up * (z1 - y1) * temp.float()  ### the same
            grad_a = grad_output * up * (z1 - y1)

        return grad_input, grad_up, grad_N, grad_a, grad_shift1, grad_shift2, grad_learn
