import math
import torch

def delphineat_gauss_activation(z):
    '''Gauss activation as defined by SharpNEAT, which is also as in DelphiNEAT.'''
    return 2 * math.exp(-1 * (z * 2.5)**2) - 1


def delphineat_sigmoid_activation(z):
    '''Sigmoidal activation function as defined in DelphiNEAT'''
    return 2.0 * (1.0 / (1.0 + math.exp(-z*5)))-1


def delphineat_gauss_torch_activation(z):
    '''PyTorch implementation of gauss activation as defined by SharpNEAT, which is also as in DelphiNEAT.'''
    return 2.0 * torch.exp(-1 * (z * 2.5) ** 2) - 1


def delphineat_sigmoid_torch_activation(z):
    '''PyTorch implementation of sigmoidal activation function as defined in DelphiNEAT'''
    return 2.0 * (1.0 / (1.0 + torch.exp(-z*5)))-1