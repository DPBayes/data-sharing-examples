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
male_variable_types_ = independent_model

maps = pickle.load(open('maps.pickle', 'rb'))[0]
maps['age'] = lambda x : sum(maps['age_lim']*np.array([-1, 1]))*x + maps['age_lim'][0]
maps['per'] = lambda x : sum(maps['per_lim']*np.array([-1, 1]))*x + maps['per_lim'][0]

from sampler import fast_sample
from load_diabetes import decode_data

N_male = 226372

def main():
	eps = float(sys.argv[1])
	seed = int(sys.argv[2])
	model_fname = os.system("ls ./male_models/ | grep {} | grep {} >> model_fnames.txt".format(eps, seed))
	model_fnames_file = open("model_fnames.txt", "r")
	model_fnames = model_fnames_file.readlines()
	model_fnames_file.close()
	model_fname = [fname for fname in model_fnames][0][:-1]
	print(model_fname)
	os.system("rm model_fnames.txt")
	male_models = pd.read_pickle('./male_models/{}'.format(model_fname))
	for i_rep, male_model in enumerate(male_models):
		male_variable_types = {key : male_variable_types_[key] for key in male_model.param_dims.keys()}

		print(i_rep)
		male_syn_data = fast_sample(male_model, male_variable_types, N_male)
		#male_syn_data[male_syn_data["ep"] == 0]["lex.dur"] = 1.0
		male_syn_decoded = decode_data(male_syn_data, maps, for_poisson=False)
		male_syn_decoded.to_csv('./syn_data/male_data_{}_{}_{}.csv'.format(seed, np.round(eps, 2), i_rep), index=False)

if __name__=="__main__":
	main()
