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
female_variable_types_ = independent_model

maps = pickle.load(open('maps.pickle', 'rb'))[0]
maps['age'] = lambda x : sum(maps['age_lim']*np.array([-1, 1]))*x + maps['age_lim'][0]
maps['per'] = lambda x : sum(maps['per_lim']*np.array([-1, 1]))*x + maps['per_lim'][0]

from sampler import fast_sample
from load_diabetes import decode_data

N_female = 208148

def main():
	eps = float(sys.argv[1])
	seed = int(sys.argv[2])
	model_fname = os.system("ls ./female_models/ | grep {} | grep {} >> model_fnames.txt".format(eps, seed))
	model_fnames_file = open("model_fnames.txt", "r")
	model_fnames = model_fnames_file.readlines()
	model_fnames_file.close()
	model_fname = [fname for fname in model_fnames][0][:-1]
	print(model_fname)
	os.system("rm model_fnames.txt")
	female_models = pd.read_pickle('./female_models/{}'.format(model_fname))
	for i_rep, female_model in enumerate(female_models):
		female_variable_types = {key : female_variable_types_[key] for key in female_model.param_dims.keys()}

		print(i_rep)
		female_syn_data = fast_sample(female_model, female_variable_types, N_female)
		#female_syn_data[female_syn_data["ep"] == 0]["lex.dur"] = 1.0
		female_syn_decoded = decode_data(female_syn_data, maps, for_poisson=False)
		female_syn_decoded.to_csv('./syn_data/female_data_{}_{}_{}.csv'.format(seed, np.round(eps, 2), i_rep), index=False)

if __name__=="__main__":
	main()
