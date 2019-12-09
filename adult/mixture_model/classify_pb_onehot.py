import pandas as pd
import pickle
import numpy as np
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model

target_variable = "Target"
learn_pb = 1

np.random.seed(123)

### Load test data and encoders
from load_adult import fetch_data
data, original_data, maps = fetch_data()
original_data = original_data.dropna()
original_features = original_data.columns
to_remove = ['Education']
for feature in to_remove:
	del data[feature]
	del original_data[feature]
N = len(data)
# Pick features to discretize
disc_features = ['Hours per week','Education-Num', 'Capital Loss', 'Capital Gain']
# Set target variable
target_variable = "Target"
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
scaler = preprocessing.StandardScaler()

X_test = pd.read_csv('../data/encoded_X_test_disc.csv', sep=';')
y_test = pd.read_csv('../data/encoded_y_test.csv', header=None).values.flatten()

features = list(data.columns)
features.remove('Target')

epsilons = [1.1, 2.0, 4.0, 8.0, 14.0]
## Learn PB classifiers
res_dict = {}
syn_pb_classifiers = {}
from utils import onehot

def decode_for_classification(X_syn):
	bins = np.linspace(-1e-6,1, 17, endpoint=True)
	for name, dtype in zip(X_syn.columns, X_syn.dtypes):
		if name in disc_features:
			feature_min = X_syn[name].min()
			feature_max = X_syn[name].max()
			X_syn[name] = (X_syn[name]-feature_min)/(feature_max-feature_min)
			X_syn[name] = pd.cut(X_syn[name], bins=bins, labels=range(16)).astype('int')
			X_syn[name] = X_syn[name].map({key : i for i, key in enumerate(np.unique(X_syn[name]))})


	del X_syn['Education']
	## Relabel education number
	X_syn['Education-Num'] = X_syn['Education-Num']+1
	## One hot categorical features
	onehotteds = []
	for col in X_syn.columns:
		feature =  X_syn[col]
		if (feature.dtype=='int' or feature.dtype=='O') and col not in onehotteds:
			if len(np.unique(feature))>2:
				X_syn.pop(col)
				onehotted = onehot(feature)
				X_syn = pd.concat([X_syn, onehotted], axis=1)
				onehotteds.append(col)
	X_syn['Sex'] = X_syn['Sex'].map({'Female' : 0, 'Male' : 1})
	return X_syn

for epsilon in epsilons:
	accs = np.zeros(10)
	classifiers = []
	for rep in range(10):
		fname_pb_rich = '../privbayes/syn_data/adult_rich_{}_{}.csv'.format(epsilon, str(rep))
		fname_pb_poor = '../privbayes/syn_data/adult_poor_{}_{}.csv'.format(epsilon, str(rep))
		syn_pb_data_rich = pd.read_csv(
			fname_pb_rich,
			names=[
				"Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
				"Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
				"Hours per week", "Country"],
				sep=';',
				na_values="NaN")
		syn_pb_data_poor = pd.read_csv(
			fname_pb_poor,
			names=[
				"Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
				"Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
				"Hours per week", "Country"],
				sep=';',
				na_values="NaN")
		syn_pb_data_rich["Target"] = 1
		syn_pb_data_poor["Target"] = 0
		X_syn_pb = syn_pb_data_rich.append(syn_pb_data_poor)
		X_syn_pb = X_syn_pb.dropna()
		y_syn_pb = X_syn_pb.pop("Target")
		# Decode
		X_syn_pb = decode_for_classification(X_syn_pb)
		# Preprocess syn_pb data
		continuous_feats = X_syn_pb.columns[X_syn_pb.dtypes=='float']
		X_syn_pb[continuous_feats] = pd.DataFrame(scaler.fit_transform(X_syn_pb[continuous_feats]\
				.astype("float64")), columns=continuous_feats)
		# X_test might have less columns than the synthetic data
		not_in_test = []
		for col in X_syn_pb:
			if col not in X_test.columns:
				not_in_test.append(col)
		X_test_padded = X_test.copy()
		for col in not_in_test:
			X_test_padded[col] = np.zeros(len(X_test_padded))
		X_test_padded = X_test_padded[X_syn_pb.columns]

		# Train classifier with syn_pb data
		cls_syn_pb = linear_model.LogisticRegression()

		if y_syn_pb.sum()==0:
			y_syn_pb.iloc[-1] = 1
		cls_syn_pb.fit(X_syn_pb, y_syn_pb)
		y_pred_syn = cls_syn_pb.predict(X_test_padded)
		print(np.mean(y_pred_syn == y_test))
		accs[rep] = np.mean(y_pred_syn == y_test)
		classifiers.append(cls_syn_pb)	
	res_dict[epsilon] = accs
	syn_pb_classifiers[epsilon] = classifiers

import datetime
date = datetime.date.today().isoformat()
pickle.dump({'cls':syn_pb_classifiers, 'accs' : res_dict},\
		open('../plot_scripts/plot_pickles/pb_classifiers_{}_onehot_{}.p'.format(date, target_variable),'wb'))
