import torch, sys, math, pickle, datetime, time
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
### Load diabetes data ###
## Encode data
from load_adult import fetch_data
data, original_data, maps = fetch_data()
original_data = original_data.dropna()
original_features = original_data.columns
to_remove = ['Education']
for feature in to_remove:
	del data[feature]
	del original_data[feature]

# Pick features to discretize
disc_features = ['Hours per week','Education-Num', 'Capital Loss', 'Capital Gain']
disc_features = [feature for feature in disc_features if feature not in to_remove]
original_continuous = original_data.columns[original_data.dtypes=='int']
N,d  = data.shape

# Discretize chosen features
bins = np.linspace(-1e-6,1, 17, endpoint=True)
for name, dtype in zip(data.columns, data.dtypes):
	if name in disc_features:
		data[name] = pd.cut(data[name], bins=bins, labels=range(16)).astype('int')
		original_data[name] = data[name]
		data[name] = data[name].map({key : i for i, key in enumerate(np.unique(data[name]))})
		original_data[name] = original_data[name].map({key : i \
				for i, key in enumerate(np.unique(original_data[name]))})

### Define model ###
variable_types =  {}
for col in data.columns:
	feature = data[col].dropna()
	if feature.dtype == 'float':
		variable_types[col] = 'Beta'
	else:
		num_uniques = len(np.unique(feature))
		if num_uniques==2:
			variable_types[col] = 'Bernoulli'
		else : 
			variable_types[col] = 'Categorical'
variable_types['pi_unconstrained'] = 'Categorical'
variable_types = OrderedDict(variable_types)
variable_types.pop('Target')

def main():
	# Set DPVI params
	T = 80000
	C = 2.0
	lr = .0005
	q = 0.005
	batch_size = int(q*N)
	sigma = float(sys.argv[1])
	income = sys.argv[2]
	seed = int(sys.argv[3])
	torch.manual_seed(seed)
	npr.seed(seed)
	# Set number of mixture components
	k = 10
	param_dims = OrderedDict()
	for key, value in variable_types.items():
		if key == 'pi_unconstrained':
			param_dims[key] = [k-1]
		else:
			if value == 'Bernoulli':
				param_dims[key] = [k]
			elif value == 'Categorical':
				param_dims[key] = [k, len(np.unique(data[key]))]
			elif value == 'Beta':
				param_dims[key] = [2,k]


	input_dim = int(np.sum([np.prod(value) for value in param_dims.values()]))
	flat_param_dims = np.array([np.prod(value) for value in param_dims.values()])

	rich_data = data[data['Target']==1]
	batch_size_rich = int(q*len(rich_data))
	poor_data = data[data['Target']==0]
	batch_size_poor = int(q*len(poor_data))

	### Save log
	date = datetime.date.today().isoformat()
	wall_start = time.time()
	cpu_start = time.clock()
	out_file = open("out_file_{}_{}_{}.txt".format(income, date, sigma), "a")
	sys.stdout = out_file
	print("Sigma : {}".format(sigma))

	## Containers for models
	models = []

	from torch.optim import Adam as Optimizer
	from dpvi import DPVI
	## Repeat inference 10 times
	if income == "rich" : 
		rich_model = ReparamXpand(batch_size_rich, input_dim, param_dims, flat_param_dims)
		optimizer_rich = Optimizer(rich_model.parameters(), lr=lr)
		# Init mixture fractions to N(0, exp(-2.0))
		rich_model.reparam.bias.data[:, -(k-1):] = 0.0*torch.ones_like(rich_model.reparam.bias.data[:, -(k-1):])
		rich_model.reparam.weight.data[:, -(k-1):] = -2.0*torch.ones_like(rich_model.reparam.weight.data[:, -(k-1):])
		rich_model_ = DPVI(rich_model, T, rich_data, batch_size_rich,\
				optimizer_rich, C, sigma, variable_types)
		models.append(rich_model_)
	else : 
		poor_model = ReparamXpand(batch_size_poor, input_dim, param_dims, flat_param_dims)
		optimizer_poor = Optimizer(poor_model.parameters(), lr=lr)
		poor_model.reparam.bias.data[:, -(k-1):] = 0.0*torch.ones_like(poor_model.reparam.bias.data[:, -(k-1):])
		poor_model.reparam.weight.data[:, -(k-1):] = -2.0*torch.ones_like(poor_model.reparam.weight.data[:, -(k-1):])

		poor_model_ = DPVI(poor_model, T, poor_data, batch_size_poor,\
				optimizer_poor, C, sigma, variable_types)
		models.append(poor_model_)
	wall_end = time.time()
	cpu_end = time.clock()
	print('Wall time {}'.format(wall_end-wall_start))
	print('CPU time {}'.format(cpu_end-cpu_start))

	## Compute privacy budget
	from privacy.analysis.compute_dp_sgd_privacy import compute_rdp, get_privacy_spent
	delta = 1e-5
	rdp_orders = range(2, 500)
	rdp_eps = compute_rdp(q, sigma, T, rdp_orders)
	epsilon = 2*get_privacy_spent(rdp_orders, rdp_eps, target_delta=delta/2)[0]
	
	pickle.dump(models, open('./res/models_{}_{}_{}_{}.p'.format(income, date, sigma, seed), 'wb'))
	params = {'T':T,  'C':C, 'lr':lr, 'k':k, 'q':q, 'sigma':sigma, 'epsilon':epsilon, 'seed':seed}
	pickle.dump(params, open('./res/params_{}_{}_{}_{}.p'.format(income, date, sigma, seed), 'wb'))
	out_file.close()

if __name__ == "__main__":
	main()
