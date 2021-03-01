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
from variable_types import independent_model as train_variable_types_

train_variable_types_base = train_variable_types_.copy()
train_variable_types_base.pop('dead')
train_variable_types_base["is.female"] = "Bernoulli"

maps = pickle.load(open('maps.pickle', 'rb'))[0]
maps['age'] = lambda x : sum(maps['age_lim']*np.array([-1, 1]))*x + maps['age_lim'][0]
maps['per'] = lambda x : sum(maps['per_lim']*np.array([-1, 1]))*x + maps['per_lim'][0]

from sampler import fast_sample
from load_diabetes import decode_data

N_female = 208148
N_male = 226372
N = N_female + N_male

def main():
	eps = float(sys.argv[1])
	seed = int(sys.argv[2])
	model_fname = os.system("ls ./train_models/ | grep {} | grep {} >> model_fnames.txt".format(eps, seed))
	model_fnames_file = open("model_fnames.txt", "r")
	model_fnames = model_fnames_file.readlines()
	model_fnames_file.close()
	model_fname = model_fnames[0][:-1]
	os.system("rm model_fnames.txt")
	train_models = pd.read_pickle('./train_models/{}'.format(model_fname))
	for i_rep, train_model in enumerate(train_models):
		train_variable_types = {key : train_variable_types_base[key] for key in train_model.param_dims.keys()}
		print(i_rep)
		syn_data = fast_sample(train_model, train_variable_types, N)
		female_syn_data = syn_data[syn_data["is.female"]==1]
		male_syn_data = syn_data[syn_data["is.female"]==0]
		female_syn_decoded = decode_data(female_syn_data, maps, for_poisson=False)
		male_syn_decoded = decode_data(male_syn_data, maps, for_poisson=False)
		female_syn_decoded.to_csv('./syn_data/female_data_{}_{}_{}.csv'.format(seed, np.round(eps, 2), i_rep), index=False)
		male_syn_decoded.to_csv('./syn_data/male_data_{}_{}_{}.csv'.format(seed, np.round(eps, 2), i_rep), index=False)

if __name__=="__main__":
	main()
