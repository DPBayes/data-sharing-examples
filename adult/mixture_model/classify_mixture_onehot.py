## Classify DPVI for separate models
import torch, sys, math, pickle, datetime, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.random as npr
from itertools import count
from collections import OrderedDict

torch.manual_seed(1234)
npr.seed(1234)

use_cuda = False
if use_cuda:
	torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
	torch.set_default_tensor_type('torch.DoubleTensor')


### Load test data and encoders
from load_adult import fetch_data
data, original_data, maps = fetch_data()
maps['Education-Num'] = []
maps['Education-Num'].append({u : i for i,u in enumerate(np.unique(original_data['Education-Num']))})
maps['Education-Num'].append({i : u for i,u in enumerate(np.unique(original_data['Education-Num']))})
original_data = original_data.dropna()
to_remove = ['Education']

N = len(data)
N_rich_true = 7508 # The number of >50K elements in training data
# Pick features to discretize
disc_features = ['Hours per week','Education-Num', 'Capital Loss', 'Capital Gain']
# Set target variable
target_variable = "Target"
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
scaler = preprocessing.StandardScaler()

X_test = pd.read_csv('../data/encoded_X_test_disc.csv', sep=';')
y_test = pd.read_csv('../data/encoded_y_test.csv', header=None).values.flatten()

## Decoding synthetic data to match test data
def decode_for_classification(X_syn):
	## Decode features
	for col in X_syn.columns:
		if col not in disc_features:
			if data[col].dtype == 'float' and col!='Education-Num':
				min_value = maps[col][0]
				max_value = maps[col][1]
				X_syn[col] = X_syn[col]*(max_value-min_value)+min_value
			else:
				X_syn[col] = X_syn[col].map(maps[col][1])
		if col == 'Education-Num':
				X_syn[col] = X_syn[col].map(maps[col][1])

	## Decode discretized features
	bins = np.linspace(-1e-6,1, 17, endpoint=True)
	for col in disc_features:
		if col!='Education-Num':
			discr_feature = pd.cut(data[col], bins=bins, labels=range(16)).astype('int')
			decode_map = {i : u for i, u in enumerate(np.unique(discr_feature))}
			X_syn[col] = X_syn[col].map(decode_map)

	## One hot categorical features
	from utils import onehot
	onehotteds = []
	for col in X_syn.columns:
		feature =  X_syn[col]
		if (feature.dtype=='int' or feature.dtype=='O') and col not in onehotteds:
			if len(np.unique(feature))>2:
				X_syn.pop(col)
				onehotted = onehot(feature)
				X_syn = pd.concat([X_syn, onehotted], axis=1)
				onehotteds.append(col)

	## Reorder columns
	X_syn = X_syn[X_test.columns]
	return X_syn

features = list(data.columns)
features.remove('Target')


### Define model ###
variable_types =  {}
for col in data.columns:
	feature = data[col].dropna()
	if col in disc_features:
		variable_types[col] = 'Categorical'
	elif feature.dtype == 'float':
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
for feature_to_remove in to_remove:
	variable_types.pop(feature_to_remove)

res_dict = {}
syn_dpvi_classifiers = {}
from linear import ReparamXpand
from sampler import fast_sample
from utils import onehot

seeds = range(1234, 1244)
for sigma in [13.39, 7.54, 4.07, 2.14, 1.37]:
	accs = np.zeros(10)
	## Read models
	rich_models = [pd.read_pickle('./res/models_rich_2019-04-24_{}_{}.p'.format(sigma, seed))[0] for seed in seeds]
	poor_models = [pd.read_pickle('./res/models_poor_2019-04-25_{}_{}.p'.format(sigma, seed))[0] for seed in seeds]
	params = [pd.read_pickle('./res/params_rich_2019-04-25_{}_{}.p'.format(sigma, seed)) for seed in seeds][0]
	i_run = 0
	classifiers = []
	for rich_model, poor_model in zip(rich_models, poor_models):
		## Generate data
		N_rich = int(N_rich_true+np.random.laplace(scale=100)) ## Epsilon = 0.01
		N_poor = N-N_rich
		syn_rich = fast_sample(rich_model, variable_types, N_rich)
		syn_poor = fast_sample(poor_model, variable_types, N_poor)
		X_syn_dpvi = syn_rich.append(syn_poor)
		y_syn_dpvi = np.concatenate([np.ones(N_rich), np.zeros(N_poor)])

		## Decode data for classification
		X_syn_dpvi = decode_for_classification(X_syn_dpvi)
		X_syn_dpvi['Sex'] = X_syn_dpvi['Sex'].map({'Female' : 0, 'Male' : 1})
		continuous_feats = X_syn_dpvi.columns[X_syn_dpvi.dtypes=='float']
		X_syn_dpvi[continuous_feats] = pd.DataFrame(scaler.fit_transform(X_syn_dpvi[continuous_feats]\
				.astype("float64")), columns=continuous_feats)

		## Train classifier with syn_dpvi data
		cls_syn_dpvi = linear_model.LogisticRegression()
		cls_syn_dpvi.fit(X_syn_dpvi, y_syn_dpvi)
		missing_cols = [name for name in X_syn_dpvi.columns if name not in X_test.columns]
		for missing_col in missing_cols:
			X_test[missing_col] = np.zeros(len(X_test))
		X_test = X_test[X_syn_dpvi.columns]
		y_pred_syn = cls_syn_dpvi.predict(X_test)
		print(np.mean(y_pred_syn == y_test))
		accs[i_run] = np.mean(y_pred_syn == y_test)
		i_run += 1
		classifiers.append(cls_syn_dpvi)	
	res_dict[params['epsilon']+0.01] = accs
	syn_dpvi_classifiers[params['epsilon']+0.01] = classifiers

## Save results
import datetime
date = datetime.date.today().isoformat()
pickle.dump({'cls':syn_dpvi_classifiers, 'accs' : res_dict},\
		open('../plot_scripts/plot_pickles/dpvi_classifiers_{}_onehot_{}.p'.format(date, target_variable),'wb'))
