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
def main():
	eps = float(sys.argv[1])
	os.system("python3 join_models.py {}".format(eps))
	alive_female_models = pd.read_pickle('./female_models/alive_female_models_{}.p'.format(np.round(eps, 2)))
	dead_female_models = pd.read_pickle('./female_models/dead_female_models_{}.p'.format(np.round(eps, 2)))
	for i_rep, (alive_female_model, dead_female_model) in enumerate(zip(alive_female_models, dead_female_models)):
		alive_female_variable_types = {key : female_variable_types[key] for \
						key in alive_female_model.param_dims.keys()}
		dead_female_variable_types = {key : female_variable_types[key] for key in dead_female_model.param_dims.keys()}

		print(i_rep)
		alive_female_syn_data = fast_sample(alive_female_model, alive_female_variable_types, int(208148*0.8))
		dead_female_syn_data = fast_sample(dead_female_model, dead_female_variable_types, int(208148*0.2))
		alive_female_syn_data['ep'] = 0
		alive_female_syn_data['lex.dur'] = 1.0
		female_syn_data = pd.concat([alive_female_syn_data, dead_female_syn_data])
		female_syn_decoded = decode_data(female_syn_data, maps, for_poisson=False)
		female_syn_decoded.to_csv('./syn_data/female_data_{}_{}.csv'.format(np.round(eps, 2), i_rep), index=False)

if __name__=="__main__":
	main()
