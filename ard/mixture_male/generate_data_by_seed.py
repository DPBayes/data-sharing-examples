import torch, sys, math, pickle, datetime, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.random as npr
from itertools import count
from collections import OrderedDict

npr.seed(1234)
use_cuda = False
if use_cuda:
	torch.set_default_tensor_type('torch.cuda.DoubleTensor')
	torch.cuda.manual_seed(1234)
else:
	torch.set_default_tensor_type('torch.DoubleTensor')
	torch.manual_seed(1234)

##################################################
##################################################
from variable_types import independent_model
male_variable_types = independent_model

maps = pickle.load(open('maps.pickle', 'rb'))[0]
maps['age'] = lambda x : sum(maps['age_lim']*np.array([-1, 1]))*x + maps['age_lim'][0]
maps['per'] = lambda x : sum(maps['per_lim']*np.array([-1, 1]))*x + maps['per_lim'][0]

from sampler import fast_sample
from load_diabetes import decode_data

N_male = 226372
N_male_dead = 44789

def main():
	eps = float(sys.argv[1])
	seed = int(sys.argv[2])
	model_fname = os.system("ls ./male_models/ | grep {} | grep {} >> model_fnames.txt".format(eps, seed))
	model_fnames_file = open("model_fnames.txt", "r")
	model_fnames = model_fnames_file.readlines()
	model_fnames_file.close()
	alive_model_fname = [fname for fname in model_fnames if 'alive' in fname][0][:-1]
	dead_model_fname = [fname for fname in model_fnames if 'dead' in fname][0][:-1]
	os.system("rm model_fnames.txt")
	alive_male_models = pd.read_pickle('./male_models/{}'.format(alive_model_fname))
	dead_male_models = pd.read_pickle('./male_models/{}'.format(dead_model_fname))
	if len(alive_male_models) != 10 or len(dead_male_models)!=10:
		print("Too few models for seed {} and eps {}",format(seed, eps))
		return 0
	for i_rep, (alive_male_model, dead_male_model) in enumerate(zip(alive_male_models, dead_male_models)):
		alive_male_variable_types = {key : male_variable_types[key] for \
						key in alive_male_model.param_dims.keys()}
		dead_male_variable_types = {key : male_variable_types[key] for key in dead_male_model.param_dims.keys()}

		print(i_rep)
		noisy_dead_proportion = (N_male_dead+np.random.laplace(scale=(1./0.01)))/N_male
		N_syn_male_alive = int((1.-noisy_dead_proportion)*N_male) 
		N_syn_male_dead = int(noisy_dead_proportion*N_male) 
		alive_male_syn_data = fast_sample(alive_male_model, alive_male_variable_types, N_syn_male_alive)
		dead_male_syn_data = fast_sample(dead_male_model, dead_male_variable_types, N_syn_male_dead)
		alive_male_syn_data['ep'] = 0
		alive_male_syn_data['lex.dur'] = 1.0
		male_syn_data = pd.concat([alive_male_syn_data, dead_male_syn_data])
		male_syn_decoded = decode_data(male_syn_data, maps, for_poisson=False)
		male_syn_decoded.to_csv('./syn_data/male_data_{}_{}_{}.csv'.format(seed, np.round(eps, 2), i_rep), index=False)

if __name__=="__main__":
	main()
