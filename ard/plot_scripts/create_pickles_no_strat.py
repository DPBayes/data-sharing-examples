"""
Some of the runs were super unstable with smallest epsilon, so need to make some exceptions
"""
import pandas as pd
import pickle
import numpy as np
from itertools import product

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.split(script_dir)[0] + "/"

## Load DPVI fits
epsilons = [0.74, 1.99, 3.92]
#epsilons = [1.99, 3.92]
epsilons = np.array(epsilons)
seeds = range(1234,1244)
n_runs = len(seeds)*10

## No stratification
## For females
syn_no_strat_coef_female_dict = {}
syn_no_strat_p_value_female_dict = {}
for eps in epsilons:
	female_coefs = []
	for seed in seeds:
		for rep in range(10):
			try:
				female_coef_df = pd.read_csv(parent_dir+'R/ablation_study/no_strat/csvs/female_coef_matrix_dpvi_{}_{}_{}.csv'.format(seed, eps, rep), index_col=0)
				female_p_value_df = pd.read_csv(parent_dir+'R/ablation_study/no_strat/csvs/female_p_value_matrix_dpvi_{}_{}_{}.csv'.format(seed, eps, rep), index_col=0)
				if len(female_coefs)==0:
					female_coefs = female_coef_df
					female_p_values = female_p_value_df
				else:
					female_coefs = pd.concat([female_coefs, female_coef_df], axis=1)
					female_p_values = pd.concat([female_p_values, female_p_value_df], axis=1)
			except:
				pass


	syn_no_strat_coef_female_df = pd.DataFrame(female_coefs.values.T, columns=female_coefs.index)
	syn_no_strat_p_value_female_df = pd.DataFrame(female_p_values.values.T, columns=female_p_values.index)

	syn_no_strat_coef_female_dict[eps] = syn_no_strat_coef_female_df
	syn_no_strat_p_value_female_dict[eps] = syn_no_strat_p_value_female_df


# For males
syn_no_strat_coef_male_dict = {}
syn_no_strat_p_value_male_dict = {}
for eps in epsilons:
	male_coefs = []
	for seed in seeds:
		for rep in range(10):
			try:
				male_coef_df = pd.read_csv(parent_dir+'R/ablation_study/no_strat/csvs/male_coef_matrix_dpvi_{}_{}_{}.csv'.format(seed, eps, rep), index_col=0)
				male_p_value_df = pd.read_csv(parent_dir+'R/ablation_study/no_strat/csvs/male_p_value_matrix_dpvi_{}_{}_{}.csv'.format(seed, eps, rep), index_col=0)
				if len(male_coefs)==0:
					male_coefs = male_coef_df
					male_p_values = male_p_value_df
				else:
					male_coefs = pd.concat([male_coefs, male_coef_df], axis=1)
					male_p_values = pd.concat([male_p_values, male_p_value_df], axis=1)
			except:
				pass


	syn_no_strat_coef_male_df = pd.DataFrame(male_coefs.values.T, columns=male_coefs.index)
	syn_no_strat_p_value_male_df = pd.DataFrame(male_p_values.values.T, columns=male_p_values.index)

	syn_no_strat_coef_male_dict[eps] = syn_no_strat_coef_male_df
	syn_no_strat_p_value_male_dict[eps] = syn_no_strat_p_value_male_df

# Save results as pickles
pickle.dump(syn_no_strat_coef_female_dict, open(script_dir+'/plot_pickles/no_strat_female_coef_dict.p', 'wb'))
pickle.dump(syn_no_strat_p_value_female_dict, open(script_dir+'/plot_pickles/no_strat_female_p_value_dict.p', 'wb'))

pickle.dump(syn_no_strat_coef_male_dict, open(script_dir+'/plot_pickles/no_strat_male_coef_dict.p', 'wb'))
pickle.dump(syn_no_strat_p_value_male_dict, open(script_dir+'/plot_pickles/no_strat_male_p_value_dict.p', 'wb'))
