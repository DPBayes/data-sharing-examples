import torch, sys, math, pickle, datetime, time
import numpy as np
import numpy.random as npr
from collections import OrderedDict

use_cuda = torch.cuda.is_available()
npr.seed(1234)
if use_cuda : 
	torch.set_default_tensor_type('torch.cuda.DoubleTensor')
	torch.cuda.manual_seed(1234)
else : 
	torch.set_default_tensor_type('torch.DoubleTensor')
	torch.manual_seed(1234)


"""
DPVI for Carat app data. Model countries with Categorical dist and apps as a binary vector.
"""
from dpvi import DPVI

def infer(T, C, sigma, batch_size, Optimizer, lr, X_apps, k):
	from linear import ReparamXpand
	N = len(X_apps)
	## Initialize and expand model
	### Define model for reparametrization
	param_dims = {'theta_unconstrained' : [k, X_apps.shape[-1]], 'pi_unconstrained' : [k-1]}
	param_dims = OrderedDict(param_dims)

	### Compute the total number of parameters in model
	input_dim = int(np.sum([np.prod(value) for value in param_dims.values()]))
	flat_param_dims = np.array([np.prod(value) for value in param_dims.values()])

	if sigma>0 : 
		model = ReparamXpand(batch_size, input_dim, param_dims, flat_param_dims)
		optimizer = Optimizer(model.parameters(), lr=lr)
	else : 
		model = ReparamXpand(1, input_dim, param_dims, flat_param_dims)
		optimizer = Optimizer(model.parameters(), lr=lr)
	if use_cuda:
		X_apps = X_apps.cuda()
		model.cuda()
	model.reparam.weight.data[:,-(k-1):].mul_(0)
	model.reparam.bias.data[:,-(k-1):].mul_(0)
	## Training model
	model = DPVI(model, T, N, batch_size, X_apps, sigma, C, optimizer)
	## Create a generative model based on model parameters and return it
	generative_model = ReparamXpand(1, input_dim, param_dims, flat_param_dims)
	generative_model.reparam.bias.detach_()
	generative_model.reparam.weight.detach_()
	generative_model.reparam.bias.data = torch.tensor(model.reparam.bias.data.cpu()[0].data.numpy(), device='cpu') 
	generative_model.reparam.weight.data = torch.tensor(model.reparam.weight.data.cpu()[0].data.numpy(), device='cpu')
	return generative_model

##################################################
def main():
	###  Set number of mixture components (k)
	k = 20
	## Training parameters
	T = 30000
	C = 1.0
	q = .001
	lr = .001
	
	### Pick dimension from argv
	d = int(sys.argv[1])
	### Compute privacy budget
	from privacy.analysis.compute_dp_sgd_privacy import compute_rdp, get_privacy_spent
	delta = 1e-5
	rdp_orders = range(2, 500)
	sigma = 2.0
	if sigma>0:
		from privacy.analysis.compute_dp_sgd_privacy import get_privacy_spent, compute_rdp
		rdp_alpha = range(2,500)
		delta = 1e-5
		print(sigma)
		rdp_eps = compute_rdp(q, sigma, T, rdp_alpha)
		epsilon = 2*get_privacy_spent(rdp_alpha, rdp_eps, target_delta = delta/2)[0]
	### Check that epsilon < 1.0
	assert(epsilon<1.0)

	### Save log
	date = datetime.date.today().isoformat()
	wall_start = time.time()
	cpu_start = time.clock()
	out_file = open("out_file_{}_{}.txt".format(date, d), "a")
	sys.stdout = out_file
	### Load carat-data
	import pandas as pd
	app_data = pd.read_csv('../data/subsets/carat_apps_sub{}.dat'.format(d), sep=' ', header=None)\
													.astype('float').values
	N = len(app_data)
	batch_size = int(N*q)
	X_apps = torch.tensor(app_data).view([N, 1, d])
	models = [] ## container to save gen_models
	for run in range(10):
		from torch.optim import Adam
		gen_model = infer(T, C, float(sigma), batch_size, Adam, lr, X_apps, k)
		models.append(gen_model)

	wall_end = time.time()
	cpu_end = time.clock()
	pickle.dump(models, open('models_{}_{}.p'.format(date, d), 'wb'))
	print('Wall time {}'.format(wall_end-wall_start))
	print('CPU time {}'.format(cpu_end-cpu_start))
	out_file.close()
	params = {'T':T, 'C':C, 'q':q, 'lr':lr, 'sigma':sigma, 'epsilon' : epsilon, 'd':d}
	pickle.dump(params, open('params_{}_{}.p'.format(date, d), 'wb'))

if __name__ == "__main__":
	main()
