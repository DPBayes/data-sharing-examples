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
        return input*torch.exp(self.weight)+self.bias

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', bias=' + str(self.bias is not None) + ')'


from collections import OrderedDict as od
class ReparamXpand(nn.Module):
	"""
	This class combines reparametrization trick and the expanding of parameter.
	
	Our mean field approximation of the posterior has 2 groups of parameters,
	the means and the standard deviations.
	Since we do the parameter optimization in \mathbb{R}, instead of std's we 
	consider log of std's as the other parameter.
	In this class, the variational means are denoted with model.reparam.bias and 
	the log std's with model.reparam.weight.

	Consider our minibatch consists of B samples, and that the dimension of the
	variational posterior is d. Since we want to compute per-example gradients for 
	DP-SGD, assign one parameter per sample by repeating the variational parameters
	B times. Thus, after \textit{expansion} our model parameters $p$ are in $\mathbb{R}^{B \times d}$
	and $p_{ij} = p{kj}$ for all $i,k \in [B]$. 
	This allows per-example gradients computation that is more efficient than sequentally 
	calling backward on each sample in batch.
	After we take the backward call, we need to sum the gradients and set 
	param.grad.data = summed_grad.repeat(batch_size).reshape(param.data.shape)
	for both param = model.reparam.weight and param = model.reparam.bias.
	
	"""
	def __init__(self, batch_size, input_dim, param_dims, flat_param_dims):
		super(ReparamXpand, self).__init__()
		"""
		batch_size : the number of samples used in SGD (int)
		input_dim : the dimension of variational posterior (int)
		param_dims : an ordered dictionary, where keys are the feature names
					and values (list) tell the dimension of the parameter assigned
					to the feature. For example if we have feature 'coin flip'
					which is Bernoulli distributed, and we model that with k
					component mixture of Bernoullis, we would ha
					param_dims['coin_flip'] = [k]
		flat_param_dims : a list that has the total number of parameters assigned
					for each key in param_dims. For example, if the ith element
					of param_dims would be [k,m], then the ith element of 
					flat_param_dims would be km.
		"""
		self.input_dim = input_dim
		self.param_dims = param_dims
		self.flat_param_dims = flat_param_dims
		self.batch_size = batch_size
		## Initialize the variational parameters.
		self.reparam = Reparametrize(input_dim, bias=True)
		# Now the model.reparam.weight.data \in \mathbb{R}^{d}
		# Next expand the parameters 
		self.reparam.weight.data = self.reparam.weight.data.repeat(batch_size)\
					.reshape([batch_size, self.input_dim])
		self.reparam.bias.data = self.reparam.bias.data.repeat(batch_size)\
					.reshape([batch_size, self.input_dim])
		# Now model.reparam.weight.data \in \mathbb{R}^{B \times d}

	def forward(self, x):
		"""
		x : a draw from N(0, I)
		"""
		# First reparametrize the draw and split the draw to separate block 
		# corresponding to the model parameters.
		draw = self.reparam(x).split(self.flat_param_dims.tolist(), dim=-1)
		# Now, create a dictionary where each value is reshaped to the correct
		# dimensionality.
		draws = {key : draw[i].view([self.batch_size] + value) \
				for i, [key, value] in enumerate(self.param_dims.items())}
		return od(draws)
