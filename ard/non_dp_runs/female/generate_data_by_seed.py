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
female_variable_types = independent_model

maps = pickle.load(open('maps.pickle', 'rb'))[0]
maps['age'] = lambda x : sum(maps['age_lim']*np.array([-1, 1]))*x + maps['age_lim'][0]
maps['per'] = lambda x : sum(maps['per_lim']*np.array([-1, 1]))*x + maps['per_lim'][0]

from sampler import fast_sample
from load_diabetes import decode_data

N_female = 208148
N_female_dead = 40391

eps = "NONDP"

def main():
	seed = int(sys.argv[1])
	model_fname = os.system("ls ./female_models/ | grep {} | grep {} >> model_fnames.txt".format(eps, seed))
	model_fnames_file = open("model_fnames.txt", "r")
	model_fnames = model_fnames_file.readlines()
	model_fnames_file.close()
	alive_model_fname = [fname for fname in model_fnames if 'alive' in fname][0][:-1]
	dead_model_fname = [fname for fname in model_fnames if 'dead' in fname][0][:-1]
	os.system("rm model_fnames.txt")
	alive_female_models = pd.read_pickle('./female_models/{}'.format(alive_model_fname))
	dead_female_models = pd.read_pickle('./female_models/{}'.format(dead_model_fname))
	for i_rep, (alive_female_model, dead_female_model) in enumerate(zip(alive_female_models, dead_female_models)):
		alive_female_variable_types = {key : female_variable_types[key] for \
						key in alive_female_model.param_dims.keys()}
		dead_female_variable_types = {key : female_variable_types[key] for key in dead_female_model.param_dims.keys()}

		print(i_rep)
		noisy_dead_proportion = (N_female_dead+np.random.laplace(scale=(1./0.01)))/N_female
		N_syn_female_alive = int((1.-noisy_dead_proportion)*N_female) 
		N_syn_female_dead = int(noisy_dead_proportion*N_female) 
		alive_female_syn_data = fast_sample(alive_female_model, alive_female_variable_types, N_syn_female_alive)
		dead_female_syn_data = fast_sample(dead_female_model, dead_female_variable_types, N_syn_female_dead)
		alive_female_syn_data['ep'] = 0
		alive_female_syn_data['lex.dur'] = 1.0
		female_syn_data = pd.concat([alive_female_syn_data, dead_female_syn_data])
		female_syn_decoded = decode_data(female_syn_data, maps, for_poisson=False)
		female_syn_decoded.to_csv('./syn_data/female_data_{}_{}_{}.csv'.format(seed, eps, i_rep), index=False)

if __name__=="__main__":
	main()
