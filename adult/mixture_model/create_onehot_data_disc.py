"""
Script for creating onehot-encoded training and testing data for DP logistic regression
"""
import pandas as pd
import numpy as np

def obj2int(x):
	uniques = np.unique(x)
	return {key:i for i,key in enumerate(uniques)}, {i:key for i,key in enumerate(uniques)}

target_feature = 'Target'
continuous_features = ['Age', 'fnlwgt', 'Capital Gain', 'Capital Loss', 'Hours per week']
## Discretize certain features to make comparable with data sharing 
disc_features = ['Hours per week', 'Capital Loss', 'Capital Gain']

################ Training data ##############################
## Read original training data
original_train_data = pd.read_csv(
	"../data/adult.data",
	names=[
		"Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
		"Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
		"Hours per week", "Country", "Target"],
		sep=r'\s*,\s*',
		engine='python',
		na_values="?")
# cast continuous features to floats
for feature in continuous_features:
	original_train_data[feature] = original_train_data[feature].astype('float')
train_data = original_train_data.dropna()
y_train = train_data.pop(target_feature)
y_train[y_train=='<=50K'] = 0
y_train[y_train=='>50K'] = 1
X_train = train_data

## Education-Num and Education features contain same information, lets drop Education
X_train.pop('Education')

## One hot categorical features
from utils import onehot
onehotteds = []
for col in X_train.columns:
	feature =  X_train[col]
	if (feature.dtype=='int' or feature.dtype=='O') and col not in onehotteds:
		if len(np.unique(feature))>2:
			X_train.pop(col)
			onehotted = onehot(feature)
			X_train = pd.concat([X_train, onehotted], axis=1)
			onehotteds.append(col)

################ Training data ##############################
## Read original training data
original_test_data = pd.read_csv(
	"../data/adult.test",
	names=[
		"Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
		"Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
		"Hours per week", "Country", "Target"],
		sep=r'\s*,\s*',
		engine='python',
		na_values="?",
		comment='|')
# cast continuous features to floats
for feature in continuous_features:
	original_test_data[feature] = original_test_data[feature].astype('float')
test_data = original_test_data.dropna()
y_test = test_data.pop(target_feature)
y_test[y_test=='<=50K'] = 0
y_test[y_test=='>50K'] = 1
X_test = test_data

## Education-Num and Education features contain same information, lets drop Education
X_test.pop('Education')

## One hot categorical features
onehotteds = []
for col in X_test.columns:
	feature =  X_test[col]
	if (feature.dtype=='int' or feature.dtype=='O') and col not in onehotteds:
		if len(np.unique(feature))>2:
			X_test.pop(col)
			onehotted = onehot(feature)
			X_test = pd.concat([X_test, onehotted], axis=1)
			onehotteds.append(col)

## Make sure that columns match, test data might be missing some of the Counties(?)
not_in_test = []
for col in X_train.columns:
	if col not in list(X_test.columns):
		not_in_test.append(col)

if len(not_in_test)>0:
	not_in_test_index = [list(X_train.columns).index(col) for col in not_in_test]
	for indx, name in zip(not_in_test_index, not_in_test):
		missing_col = pd.DataFrame(np.zeros(len(X_test), dtype='int'), columns=[name], index=X_test.index)
		X_test[name] = missing_col

X_test = X_test[X_train.columns]

assert(sum([test_col == train_col for test_col, train_col in zip(list(X_test.columns), list(X_train.columns))]))

## Encode features 
import sklearn.preprocessing as preprocessing
def number_encode_features(df):
	result = df.copy()
	encoders = {}
	for column in result.columns:
		if result.dtypes[column] == np.object:
			encoders[column] = preprocessing.LabelEncoder()
			result[column] = encoders[column].fit_transform(result[column])
	return result, encoders
X_train, train_encoders = number_encode_features(X_train)
X_test, test_encoders = number_encode_features(X_test)


## Discretize chosen features
for feature in disc_features:
	# figure out max from train and test
	feature_max = max(X_train[feature].max(), X_test[feature].max())
	bins = np.linspace(-1e-6, feature_max+1, 17, endpoint=True)
	X_train[feature] = pd.cut(X_train[feature], bins=bins, labels=range(16)).astype('int')
	X_test[feature] = pd.cut(X_test[feature], bins=bins, labels=range(16)).astype('int')

## Onehot discretized features
# for train
for col in disc_features:
	feature =  X_train[col]
	X_train.pop(col)
	onehotted = onehot(feature)
	X_train = pd.concat([X_train, onehotted], axis=1)
# for test
for col in disc_features:
	feature =  X_test[col]
	X_test.pop(col)
	onehotted = onehot(feature)
	X_test = pd.concat([X_test, onehotted], axis=1)
## Make sure that columns match after discretization
not_in_test = []
for col in X_train.columns:
	if col not in list(X_test.columns):
		not_in_test.append(col)

if len(not_in_test)>0:
	not_in_test_index = [list(X_train.columns).index(col) for col in not_in_test]
	for indx, name in zip(not_in_test_index, not_in_test):
		missing_col = pd.DataFrame(np.zeros(len(X_test), dtype='int'), columns=[name], index=X_test.index)
		X_test[name] = missing_col

X_test = X_test[X_train.columns]

assert(sum([test_col == train_col for test_col, train_col in zip(list(X_test.columns), list(X_train.columns))]))

## Scale continuous features
continuous_features = [feature for feature in continuous_features if feature not in disc_features]
scaler = preprocessing.StandardScaler()
for feature in continuous_features:
	fit = scaler.fit(X_train[feature].values[:, np.newaxis])
	X_train[feature] = fit.transform(X_train[feature].values[:, np.newaxis])
	X_test[feature] = fit.transform(X_test[feature].values[:, np.newaxis])
## Save datas
X_train.to_csv('./onehotted_data/encoded_X_train_disc.csv', sep=';', index=False)
X_test.to_csv('./onehotted_data/encoded_X_test_disc.csv', sep=';', index=False)
