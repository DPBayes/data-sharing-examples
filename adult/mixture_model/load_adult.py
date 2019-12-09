
import numpy as np
import pandas as pd

def obj2int(x):
	uniques = np.unique(x)
	return {key:i for i,key in enumerate(uniques)}, {i:key for i,key in enumerate(uniques)}


def fetch_data(include_test=False):
	original_data = pd.read_csv(
		"../data/adult.data",
		names=[
			"Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
			"Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
			"Hours per week", "Country", "Target"],
			sep=r'\s*,\s*',
			engine='python',
			na_values="?")
	data = original_data.dropna()
	## Privbayes was trained with both train and test data, so try that with DPVI
	test_data = pd.read_csv('../data/adult.test', sep=r'\s*,\s*', comment="|", header = None, names=data.columns,\
							engine="python", na_values="?")
	test_data = test_data.dropna()
	if include_test : data = data.append(test_data)

	maps = {}

	# Encode variables
	for col in data.columns:
		if data[col].dtype in ['int', 'float']:
			min_value = np.min(data[col])
			max_value = np.max(data[col])
			maps[col] = [min_value, max_value]
			data[col] = np.clip((data[col]-min_value)/(max_value-min_value), 1e-6, 1-1e-6)
		if data[col].dtype == 'O':
			maps[col] = obj2int(data[col])
			data[col] = data[col].map(maps[col][0])
	return data, original_data, maps

def decode_data(syn_data, maps):
	synthetic_data = pd.DataFrame()
	for col in syn_data.columns:
		decode_map = maps[col]
		if type(decode_map)==list:
			min_value = decode_map[0]
			max_value = decode_map[1]
			synthetic_data[col] = syn_data[col]*(max_value-min_value)+min_value
		else:
			synthetic_data[col] = syn_data[col].map(decode_map[1])
	return synthetic_data
