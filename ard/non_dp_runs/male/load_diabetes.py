#.. oleelliset muuttujat, muita ei kannata käyttää:

#lex.Xst : status (0/1) kuollut vai elossa

#Relevel(age.cat,list(1:2,3,4,5,6,7:9)): ikä, luokiteltu uudelleen

#per.cat: kalenteriaika (vuosi)

#(C10AA.DDD>0): indikaattori onko statiini käytössä seurannan alussa  ATC-koodi C10AA vai ei ( https://www.whocc.no/atc_ddd_index/ täältä koodi)

#DM.type : diabeteksen tyyppi

#.i.cancer: onko syöpä seurannan alussa (0/1)

#factor(shp); sairaahoitopiiri faktorina eli lukittelumuuttujana

#lex.dur: seuranta-ajan pituus (vuosissa) ; seuranta-aikaa kertoo myös muuttuja fu

# varmasti myös sukupuoli on merkittävä

import numpy as np
import pandas as pd
import pickle

def get_path():
	path_file = open('path.txt', 'r')
	path = path_file.readlines()[0][:-1]
	path_file.close()
	return path

def fetch_data(return_maps=False):
	path = get_path()
	female_data = pd.read_csv(path + 'female_aggregated_data.csv', delimiter=',')
	male_data = pd.read_csv(path + 'male_aggregated_data.csv', delimiter=',')
	female_data['sex'] = 'female'
	male_data['sex'] = 'male'
	# Stack male and female data
	data = female_data.append(male_data)
	data.pop('Unnamed: 0')
	# Add death as a feature. Individual is dead if he/she didn't survive until end of follow-up ('EOF')
	data['dead'] = 1*(data['EOF']<data['EOF'].max())
	# Reindex 'shp' feature
	data['shp'] = data['shp']-1
	train_data = np.zeros(data.shape)
	maps = {}
	maps['shp'] = {i:i+1 for i in range(data['shp'].max()+1)}
	# Encode features to numericals
	for i, name in enumerate(data.columns):
		if data[name].dtype == 'object':
			column = data[name]
			uniques = np.unique(column)
			maps[name] = {elem:unique for elem, unique in enumerate(uniques)}
			#column = pd.Series(column, dtype='category').cat.rename_categories(range(np.unique(column).shape[0]))
			#train_data[:,i] = column.values
			train_data[:,i] = column.map({unique:elem for elem, unique in enumerate(uniques)})
		else:
			train_data[:,i] = data[name].values

	# For C10AA.DDD and G03.DDD (female only) we consider only whether the attribute is positive or 0.
	train_data[:, data.columns.tolist().index('C10AA.DDD')] =\
									train_data[:, data.columns.tolist().index('C10AA.DDD')]>0
	train_data[:, data.columns.tolist().index('G03.DDD')] =\
									train_data[:, data.columns.tolist().index('G03.DDD')]>0

	# Normalize age to (0, 1)
	age = train_data[:, data.columns.tolist().index('age')]
	age_max = age.max()
	normalized_age = age/age_max
	#normalized_age = np.clip(normalized_age, 1e-6, 1-1e-5)
	normalized_age = np.clip(normalized_age, 0.01, 0.99)
	train_data[:, data.columns.tolist().index('age')] = normalized_age
	maps['age'] = lambda x: x*age_max
	# Normalize per to (0, 1) and lex.dur according to per to (0,1)
	per = data['per']
	lex_dur = data['lex.dur']
	end_of_fu = (per+lex_dur).max()
	normalized_lex_dur = np.clip(lex_dur/(end_of_fu-per), 1e-6, 1-1e-6)
	normalized_per = np.clip((per-per.min())/(per.max()-per.min()), 1e-6, 1-1e-6)
	train_data[:,  data.columns.tolist().index('per')] = normalized_per
	train_data[:,  data.columns.tolist().index('lex.dur')] = normalized_lex_dur
	per_min, per_max = per.min(), per.max()
	maps['per'] = lambda x: x*(per_max-per_min)+per_min
	# Separate female and male data
	sex_indx = data.columns.tolist().index('sex')
	female_train_data = train_data[np.where(train_data[:,sex_indx]==0), :][0]
	female_train_data = np.delete(female_train_data, sex_indx ,1)

	male_train_data = train_data[np.where(train_data[:,sex_indx]==1), :][0]
	male_train_data = np.delete(male_train_data, sex_indx ,1)
	
	# Cast np.arrays to pandas
	data.pop('sex')
	female_df = pd.DataFrame(female_train_data, columns = data.columns)
	male_df = pd.DataFrame(male_train_data, columns = data.columns)
	if return_maps:
		return female_df, male_df, data.dtypes, maps
	else: return female_df, male_df, data.dtypes

def decode_data(syn_data, maps, for_poisson=True):
	data = syn_data.copy()
	if 'ade' in data.columns:
		alive = 1*(data['ade']==0)
		data['dead'] = 1-alive
		data['ep'] = 1*(data['ade']==2)
	for key in maps.keys():
		if key in data.columns:data[key] = data[key].map(maps[key])
	min_per, max_per = np.round(maps['per'](np.arange(0,2)))
	data['per.cat'] = pd.cut(data['per'], np.arange(min_per, max_per+1),\
									labels=list(maps['per.cat'].values()))
	data['lex.dur'] = (2013-data['per'])*data['lex.dur']

	### Data for Poisson model
	if for_poisson:
		data_poisson = data.copy()
		data_poisson['age'] = data_poisson['age']+1e-6
		data_poisson['age.cat'] = pd.cut(data_poisson['age'], np.arange(0,90, 10).tolist()+[np.inf],\
													labels=list(maps['age.cat'].values()))
		data_poisson['age.cat'] = data_poisson['age.cat'].map({value:'60+' \
							if value in list(maps['age.cat'].values())[-3:] else value\
							for value in list(maps['age.cat'].values())})

		data_poisson['shp'] = data_poisson['shp'].astype('str') 
		lex_bins = [0,0.5,1,2,3,4,5,10,12.5,np.inf]
		lex_label = ['('+str(lex_bins[i])+','+str(lex_bins[i+1])+']' for i in range(len(lex_bins)-1)]
		data_poisson['lex.dur.cat'] = pd.cut(data_poisson['lex.dur'] ,lex_bins, labels = lex_label)

		data_poisson['C10AA.DDD>0'] = (data_poisson['C10AA.DDD']>0).map({False:'False', True:'True'})
		if 'G03.DDD' in data_poisson.columns:
			data_poisson['G03.DDD>0'] = (data_poisson['G03.DDD']>0).map({False:'False', True:'True'})
		data_poisson['per.cat'] = data_poisson['per.cat'].map({elem:i+2 for i, elem in \
									enumerate(np.unique(data_poisson['per.cat']))}).astype('int')
		return data, data_poisson
	else: return data
