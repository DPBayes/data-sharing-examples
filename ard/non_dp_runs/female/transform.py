import torch
import numpy as np

from torch.nn import Softmax, LogSoftmax
from torch.nn.functional import pad
# Define centered_softmax by concatenating 0 into the end of vector to be softmaxed.

def softmax(x, dim=0, additional=0.0):
	"""
	Defines bijective softmax
	"""
	return Softmax(dim=dim).forward(pad(x, (0,1), value=additional)) 

def log_det_jacobian_softmax(y, dim=None):
	if dim!=None: return torch.sum(torch.log(y), dim=dim)
	else: return torch.sum(torch.log(y))

def logsoftmax(x, dim=0, additional=0.0):
	"""
	Defines bijective log-softmax
	"""
	return LogSoftmax(dim=dim).forward(pad(x, (0,1), value=additional))

def log_det_jacobian_logsoftmax(y, dim=None):
	if dim!=None: return torch.sum(y, dim=dim)
	else: return torch.sum(y)

def log_det_jacobian_sigmoid(y):
	return torch.log(y) + torch.log(1-y)

import numpy as np
import numpy.random as npr
def sm(x, additional=None):
	if additional is not None:
		x = np.hstack([x, additional])
	return np.exp(x)/np.sum(np.exp(x))

def smoid(x):
	return 1/(1+np.exp(-x))
