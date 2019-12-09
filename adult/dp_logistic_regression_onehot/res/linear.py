import math
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.functional as F
from torch.nn.modules import Module

'''
Linear module modified for PX gradients
'''

class Reparametrize(Module):
    def __init__(self, in_features, bias=True):
        super(Reparametrize, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        returns \eta*\exp(log_std)+\mu, where \eta \sim N(0, I)
        """
        return F.mul(input, torch.exp(self.weight))+self.bias

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', bias=' + str(self.bias is not None) + ')'


from collections import OrderedDict as od
class ReparamXpand(nn.Module):
	def __init__(self, batch_size, input_dim):
		super(ReparamXpand, self).__init__()
		self.input_dim = input_dim
		self.reparam = Reparametrize(input_dim, bias=True)
		self.reparam.weight.data = self.reparam.weight.data.repeat(batch_size)
		self.reparam.weight.data = self.reparam.weight.data.reshape([batch_size, input_dim]) 
		self.reparam.bias.data = self.reparam.bias.data.repeat(batch_size)
		self.reparam.bias.data = self.reparam.bias.data.reshape([batch_size, input_dim]) 

		self.batch_size = batch_size

	def forward(self, x):
		draw = self.reparam(x)
		return draw
