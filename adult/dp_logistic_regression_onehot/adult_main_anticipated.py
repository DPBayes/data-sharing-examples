import torch, sys, math, pickle, datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.random as npr
from itertools import count
from collections import OrderedDict


use_cuda = False
if use_cuda:
	torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
	torch.set_default_tensor_type('torch.DoubleTensor')

from linear import ReparamXpand
##################################################
### Inference ###
"""
	Runs DPVI for given parameters and returns a generative model
"""
from dpvi import DPVI
def infer(T, C, sigma, batch_size, n_mc, Optimizer, learning_rate, train_data):
	## Initialize and expand model
	
	input_dim = train_data.shape[1]-1
	model = ReparamXpand(batch_size, input_dim)
	if use_cuda:
		model.cuda()
	optimizer = Optimizer(model.parameters(), lr=learning_rate)
	model = DPVI(model, T, n_mc, train_data.shape[0], \
				batch_size, train_data, sigma, C, optimizer, use_cuda)

	## Create a generative model based on model parameters and return it
	generative_model = ReparamXpand(1, input_dim)
	generative_model.reparam.bias.data = torch.tensor(model.reparam.bias.data.cpu()[0].data.numpy(), device='cpu') 
	generative_model.reparam.weight.data = torch.tensor(model.reparam.weight.data.cpu()[0].data.numpy(), device='cpu')
	generative_model.reparam.bias.detach_()
	generative_model.reparam.weight.detach_()
	return generative_model

##################################################
### Load adult data ###
##################################################
X_train = pd.read_csv('../data/encoded_X_train.csv', sep=';')
y_train = pd.read_csv('../data/encoded_y_train.csv', sep=';', header=None)

train_data = X_train.copy()
N = len(train_data)
train_data['Target'] = y_train.values.astype('double')
train_data['Intercept'] = np.ones(N)
train_data = train_data[['Intercept']+list(X_train.columns)+['Target']]

def main():
	### Define model ###
	# Set DPVI params
	T = 10000
	n_mc = 1
	C = 5.0
	# set number of mixture components
	q = 0.002
	batch_size = int(np.floor(q*N))
	## Pick optimizer
	from torch.optim import Adam
	optimizer = Adam
	delta = 1e-5
	learning_rate = 0.01

	n_runs = 10

	learn_sigmas = 0
	if learn_sigmas:
		target_epsilons = [1.1, 2.0, 4.0, 8.0, 14.0]
		from find_sigmas import find_sigma
		sigmas_dict = {}
		epsilons_dict = {}
		for target_eps in target_epsilons:
			U = 58
			L = 0.5
			sigmas = []
			epsilons = []
			for ant_T in [2,5,10,20]:
				sigma, eps = find_sigma(target_eps, U, L, q, T, delta/2, ant_T)
				sigmas.append(sigma)
				epsilons.append(eps)
				L = sigma
			sigmas_dict[target_eps] = sigmas
			epsilons_dict[target_eps] = epsilons
		pickle.dump({'sigmas_for_eps' : sigmas_dict}, open('./res/sigmas_for_eps.p', 'wb'))
	else:
		sigmas_dict = pickle.load(open('./res/sigmas_for_eps.p', 'rb'))['sigmas_for_eps']
		target_eps = float(sys.argv[1])
		seed = int(sys.argv[2])

	torch.manual_seed(seed)
	npr.seed(seed)
	print("target eps : {}".format(target_eps))
	print("seed : {}".format(seed))

	# Learn models
	models = []
	for sigma in sigmas_dict[target_eps]:
		model = infer(T, C, sigma, batch_size, n_mc, optimizer, learning_rate, train_data)
		models.append(model)
	import datetime
	date = datetime.date.today().isoformat()
	pickle.dump(models, open('./res/models_{}_{}_{}.p'.format(date, target_eps, seed), 'wb'))
	
if __name__ == "__main__":
	main()
