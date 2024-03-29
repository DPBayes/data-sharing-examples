import torch, sys, math, pickle, datetime, time
import numpy as np
import pandas as pd
import numpy.random as npr
from itertools import count
from collections import OrderedDict

use_cuda = torch.cuda.is_available()

from linear import ReparamXpand

##################################################
### Inference ###
"""
	Runs DPVI for given parameters and returns a generative model
"""
from dpvi import DPVI
def infer(T, C, sigma, batch_size, Optimizer, learning_rate, train_data, variable_types, k):
	## Initialize and expand model
	param_dims = OrderedDict()
	for key, value in variable_types.items():
		if key == 'pi_unconstrained':
			param_dims[key] = [k-1]
		else:
			if value == 'Bernoulli':
				param_dims[key] = [k]
			elif (key=='lex.dur' and variable_types[key]==None):
				param_dims[key] = [2, k]
			elif (key=='ep' and variable_types[key]==None):
				param_dims[key] = [k]
			elif (key=='dead' and variable_types[key]==None):
				param_dims[key] = [k]
			elif value == 'Beta':
				param_dims[key] = [2, k]
			elif value == 'Categorical':
				param_dims[key] = [k, len(np.unique(train_data[key]))]
	
	input_dim = int(np.sum([np.prod(value) for value in param_dims.values()]))
	flat_param_dims = np.array([np.prod(value) for value in param_dims.values()])
	model = ReparamXpand(batch_size, input_dim, param_dims, flat_param_dims)

	### Init model close to feature means
	def logit(y):
		return torch.log(y)-torch.log(1.-y)
	def inverse_softmax(y):
		last = 1e-23*torch.ones(1) # just something small
		sum_term = -50.-torch.log(last)
		x = torch.log(y)-sum_term
		return x
	### Init model close to feature means
	## Laplace mech with small epsilon to guarantee DP of the initialization
	eps_init = 0.01
	for key in train_data.columns:	
		if variable_types[key]=='Bernoulli' or key in ['dead']:
			initialization_noise = torch.as_tensor(np.random.laplace(1./eps_init))
			param_mean = torch.as_tensor(train_data[key].mean(0))+initialization_noise/len(train_data)
			param_location = list(model.param_dims.keys()).index(key)
			init_param = logit(torch.rand(k)*(param_mean*2.-param_mean*0.5)+param_mean*0.5)

			start_index = np.sum(model.flat_param_dims[:param_location])
			model.reparam.bias.data[:, start_index:(start_index+np.sum(model.param_dims[key]))] =\
							init_param.repeat(batch_size).reshape(batch_size, *param_dims[key])
		elif variable_types[key]=='Categorical':
			freqs = np.unique(train_data[key], return_counts=1)[1]
			num_cats = len(freqs)
			initialization_noise = torch.as_tensor(np.random.laplace(1./eps_init, size=num_cats))
			param_mean = torch.as_tensor(freqs/np.sum(freqs))+initialization_noise/len(train_data)
			init_param = inverse_softmax(param_mean)
			init_param = 0.5*torch.randn(k, num_cats)+init_param
			init_param = init_param.flatten()
			param_location = list(model.param_dims.keys()).index(key)
			start_index = np.sum(model.flat_param_dims[:param_location])
			model.reparam.bias.data[:, start_index:(start_index+np.prod(model.param_dims[key]))] =\
							init_param.repeat(batch_size).reshape(batch_size, np.prod(param_dims[key]))

			
	if use_cuda:
		model.cuda()
	optimizer = Optimizer(model.parameters(), lr=learning_rate)
	N = len(train_data)
	model = DPVI(model, T, N, batch_size, train_data, sigma, C, optimizer, variable_types)

	## Create a generative model based on model parameters and return it
	generative_model = ReparamXpand(1, input_dim, param_dims, flat_param_dims)
	generative_model.reparam.bias.detach_()
	generative_model.reparam.weight.detach_()
	generative_model.reparam.bias.data = torch.tensor(model.reparam.bias.data.cpu()[0].data.numpy(), device='cpu') 
	generative_model.reparam.weight.data = torch.tensor(model.reparam.weight.data.cpu()[0].data.numpy(), device='cpu')
	#return generative_model, z_maps
	return generative_model


##################################################
### Load diabetes data ###
## Encode data
from load_diabetes import fetch_data
female_df, male_df, data_dtypes = fetch_data()
data_dtypes['G03.DDD'] = 'int64'
female_N = len(female_df)
male_N = len(male_df)

##################################################
### Define model ###
## For female

# Load variable type dictionaries for both independent and dependent types
from variable_types import independent_model as female_variable_types_

female_variable_types = female_variable_types_.copy()
female_variable_types.pop('dead')

# Pick features for training
female_features = list(female_variable_types.keys())
female_features.remove('pi_unconstrained')

# Cast features to appropriate dtypes
female_dtypes = {key:value if value!='O' else 'int64' for key, value in \
									data_dtypes[female_features].items()} 

# Pick features
female_df = female_df[female_features]

def main():
	# Set DPVI params
	T = 10000
	C = 1.0
	lr = 1e-2
	# set number of mixture components
	female_k = 10
	q = 0.005
	sigma = float(sys.argv[1])
	n_runs = int(sys.argv[2])
	seed = int(sys.argv[3])
	delta = 1e-6
	# Set optimizer
	optimizer = torch.optim.Adam
	## Set random seed
	npr.seed(seed)
	if use_cuda:
		torch.set_default_tensor_type('torch.cuda.DoubleTensor')
		torch.cuda.manual_seed(seed)
	else:
		torch.set_default_tensor_type('torch.DoubleTensor')
		torch.manual_seed(seed)

	## Compute privacy budget
	from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy import compute_rdp, get_privacy_spent
	rdp_orders = range(2, 500)
	rdp_eps = compute_rdp(q, sigma, T, rdp_orders)
	epsilon = 2*get_privacy_spent(rdp_orders, rdp_eps, target_delta=delta/2)[0]
	print("Epsilon : {}".format(epsilon))

	## Save parameters
	res_dir = './res/'
	params = {'T':T, 'C':C, 'lr':lr, 'female_k':female_k,\
				'q':q, 'sigma':sigma, 'epsilon':epsilon, 'n_runs':n_runs, 'seed':seed}
	## Determine filename
	fname_i = 0
	date = datetime.date.today().isoformat()
	fname = '{}_{}'.format(date, seed)
	while True:
		try : 
			param_file = open(res_dir+'params_{}_{}.p'.format(fname, np.round(epsilon, 2)), 'r')
			param_file.close()
			if fname_i == 0: fname += '_({})'.format(fname_i)
			else: fname = fname[:-4]+'_({})'.format(fname_i)
			fname_i += 1
		except :
			break
			
	pickle.dump(params, open(res_dir+'params_{}_{}.p'.format(fname, np.round(epsilon, 2)), 'wb'))
	learn_counter = count()
	female_models = []
	out_file = open(res_dir+'out_{}_{}.txt'.format(fname, np.round(epsilon, 2)), 'w')
	for i in range(n_runs):
		start_time = time.time()
		print(learn_counter.__next__())
		# train 
		female_model = infer(T, C, float(sigma), int(q*len(female_df)),\
			optimizer, lr, female_df, female_variable_types, female_k)
                # save results
		female_models.append(female_model)
		pickle.dump(female_models, open('./female_models/'+'female_models_{}_{}.p'\
					.format(fname, np.round(epsilon, 2)), 'wb'))
		stop_time = time.time()
		time_delta = stop_time-start_time
		out_file.writelines("Took {} seconds\n".format(time_delta))
		print("Took {} seconds\n".format(time_delta))
	out_file.close()
if __name__ == "__main__":
	main()
