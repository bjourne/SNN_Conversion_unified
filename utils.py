import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter
# from torch.nn import functional as F
from modules import TCL, ScaledNeuron, StraightThrough


def isActivation(name):
    if 'relu' in name.lower() or 'clamp' in name.lower() or 'clip' in name.lower() or 'slip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False


def replace_activation_by_module(model, m):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_module(module, m)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                print(module.up.item())
                model._modules[name] = m(module.up.item())
            else:
                model._modules[name] = m()
    return model





def replace_activation_by_neuron(model, shift):
    """ shift2 here corresponds to shift2 in the main, default value=0.5"""
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_activation_by_neuron(module, shift)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                model._modules[name] = ScaledNeuron(scale=module.up.item(), shift=shift)
            else:
                model._modules[name] = ScaledNeuron(scale=1., shift=shift)
    return model


def replace_maxpool2d_by_avgpool2d(model):
    """ Recursively replace max pooling by average pooling """
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding)
    return model


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


# Not necessary since we use the same weight_decay for all.
def regular_set(net, paras=([],[],[])):
    for n, module in net._modules.items():
        # print(n, module)
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras
