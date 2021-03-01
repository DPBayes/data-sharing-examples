import pandas as pd
import pickle
import numpy as np
from itertools import product

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.split(script_dir)[0] + "/"

## Load DPVI fits
epsilons = [0.74, 1.99, 3.92]
epsilons = np.array(epsilons)
seeds = range(1234,1244)
seeds_k40 = range(12345,12355)
n_runs = len(seeds)*10

syn_dpvi_coef_female_dict = {}
syn_dpvi_coef_male_dict = {}
syn_dpvi_p_value_female_dict = {}
syn_dpvi_p_value_male_dict = {}
# For females
syn_dpvi_coef_female_dict = {}
syn_dpvi_p_value_female_dict = {}
for eps in epsilons:
	female_coefs = pd.concat([pd.read_csv(parent_dir+'R/dpvi/female/csvs/female_coef_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	female_p_values = pd.concat([pd.read_csv(parent_dir+'R/dpvi/female/csvs/female_p_value_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	syn_dpvi_coef_female_df = pd.DataFrame(female_coefs.values.T, columns=female_coefs.index)
	syn_dpvi_p_value_female_df = pd.DataFrame(female_p_values.values.T, columns=female_p_values.index)

	syn_dpvi_coef_female_dict[eps] = syn_dpvi_coef_female_df
	syn_dpvi_p_value_female_dict[eps] = syn_dpvi_p_value_female_df

# For males
syn_dpvi_coef_male_dict = {}
syn_dpvi_p_value_male_dict = {}
for eps in epsilons:
	male_coefs = pd.concat([pd.read_csv(parent_dir+'R/dpvi/male/csvs/male_coef_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	male_p_values = pd.concat([pd.read_csv(parent_dir+'R/dpvi/male/csvs/male_p_value_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	syn_dpvi_coef_male_df = pd.DataFrame(male_coefs.values.T, columns=male_coefs.index)
	syn_dpvi_p_value_male_df = pd.DataFrame(male_p_values.values.T, columns=male_p_values.index)

	syn_dpvi_coef_male_dict[eps] = syn_dpvi_coef_male_df
	syn_dpvi_p_value_male_dict[eps] = syn_dpvi_p_value_male_df

# Save results as pickles
pickle.dump(syn_dpvi_coef_female_dict, open(script_dir+'/plot_pickles/female_coef_dict.p', 'wb'))
pickle.dump(syn_dpvi_p_value_female_dict, open(script_dir+'/plot_pickles/female_p_value_dict.p', 'wb'))

pickle.dump(syn_dpvi_coef_male_dict, open(script_dir+'/plot_pickles/male_coef_dict.p', 'wb'))
pickle.dump(syn_dpvi_p_value_male_dict, open(script_dir+'/plot_pickles/male_p_value_dict.p', 'wb'))

## Load NON-DP fits

syn_nondp_coef_female_dict = {}
syn_nondp_coef_male_dict = {}
syn_nondp_p_value_female_dict = {}
syn_nondp_p_value_male_dict = {}

# For females, NON-DP
syn_nondp_coef_female_dict = {}
syn_nondp_p_value_female_dict = {}
for eps in ["NONDP"]:
	female_coefs = pd.concat([pd.read_csv(parent_dir+'R/nondp/female/csvs/female_coef_matrix_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	female_p_values = pd.concat([pd.read_csv(parent_dir+'R/nondp/female/csvs/female_p_value_matrix_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	syn_nondp_coef_female_df = pd.DataFrame(female_coefs.values.T, columns=female_coefs.index)
	syn_nondp_p_value_female_df = pd.DataFrame(female_p_values.values.T, columns=female_p_values.index)

	syn_nondp_coef_female_dict[eps] = syn_nondp_coef_female_df
	syn_nondp_p_value_female_dict[eps] = syn_nondp_p_value_female_df

# For males, NON-DP
syn_nondp_coef_male_dict = {}
syn_nondp_p_value_male_dict = {}
for eps in ["NONDP"]:
	male_coefs = pd.concat([pd.read_csv(parent_dir+'R/nondp/male/csvs/male_coef_matrix_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	male_p_values = pd.concat([pd.read_csv(parent_dir+'R/nondp/male/csvs/male_p_value_matrix_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	syn_nondp_coef_male_df = pd.DataFrame(male_coefs.values.T, columns=male_coefs.index)
	syn_nondp_p_value_male_df = pd.DataFrame(male_p_values.values.T, columns=male_p_values.index)

	syn_nondp_coef_male_dict[eps] = syn_nondp_coef_male_df
	syn_nondp_p_value_male_dict[eps] = syn_nondp_p_value_male_df

## Save results as pickles
pickle.dump(syn_nondp_coef_female_dict, open(script_dir+'/plot_pickles/female_coef_dict_NONDP.p', 'wb'))
pickle.dump(syn_nondp_p_value_female_dict, open(script_dir+'/plot_pickles/female_p_value_dict_NONDP.p', 'wb'))

pickle.dump(syn_nondp_coef_male_dict, open(script_dir+'/plot_pickles/male_coef_dict_NONDP.p', 'wb'))
pickle.dump(syn_nondp_p_value_male_dict, open(script_dir+'/plot_pickles/male_p_value_dict_NONDP.p', 'wb'))

## Load NON-DP k = 40 fits
syn_nondp_k40_coef_female_dict = {}
syn_nondp_k40_coef_male_dict = {}
syn_nondp_k40_p_value_female_dict = {}
syn_nondp_k40_p_value_male_dict = {}

# For females, NON-DP
syn_nondp_k40_coef_female_dict = {}
syn_nondp_k40_p_value_female_dict = {}
for eps in ["NONDP"]:
	female_coefs = pd.concat([pd.read_csv(parent_dir+'R/nondp/female/csvs/female_coef_matrix_k=40_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds_k40, range(10))], axis=1)

	female_p_values = pd.concat([pd.read_csv(parent_dir+'R/nondp/female/csvs/female_p_value_matrix_k=40_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds_k40, range(10))], axis=1)

	syn_nondp_k40_coef_female_df = pd.DataFrame(female_coefs.values.T, columns=female_coefs.index)
	syn_nondp_k40_p_value_female_df = pd.DataFrame(female_p_values.values.T, columns=female_p_values.index)

	syn_nondp_k40_coef_female_dict[eps] = syn_nondp_k40_coef_female_df
	syn_nondp_k40_p_value_female_dict[eps] = syn_nondp_k40_p_value_female_df

# For males, NON-DP
syn_nondp_k40_coef_male_dict = {}
syn_nondp_k40_p_value_male_dict = {}
for eps in ["NONDP"]:
	male_coefs = pd.concat([pd.read_csv(parent_dir+'R/nondp/male/csvs/male_coef_matrix_k=40_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds_k40, range(10))], axis=1)

	male_p_values = pd.concat([pd.read_csv(parent_dir+'R/nondp/male/csvs/male_p_value_matrix_k=40_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds_k40, range(10))], axis=1)

	syn_nondp_k40_coef_male_df = pd.DataFrame(male_coefs.values.T, columns=male_coefs.index)
	syn_nondp_k40_p_value_male_df = pd.DataFrame(male_p_values.values.T, columns=male_p_values.index)

	syn_nondp_k40_coef_male_dict[eps] = syn_nondp_k40_coef_male_df
	syn_nondp_k40_p_value_male_dict[eps] = syn_nondp_k40_p_value_male_df

## Save results as pickles
pickle.dump(syn_nondp_k40_coef_female_dict, open(script_dir+'/plot_pickles/female_coef_dict_NONDP_k40.p', 'wb'))
pickle.dump(syn_nondp_k40_p_value_female_dict, open(script_dir+'/plot_pickles/female_p_value_dict_NONDP_k40.p', 'wb'))

pickle.dump(syn_nondp_k40_coef_male_dict, open(script_dir+'/plot_pickles/male_coef_dict_NONDP_k40.p', 'wb'))
pickle.dump(syn_nondp_k40_p_value_male_dict, open(script_dir+'/plot_pickles/male_p_value_dict_NONDP_k40.p', 'wb'))



## Load PB fits

syn_pb_coef_female_dict = {}
syn_pb_coef_male_dict = {}

# For females
for eps in epsilons:
	eps_ = np.round(eps, 0)
	female_pb_coefs = pd.concat([pd.read_csv(parent_dir+'R/privbayes/female/csvs/female_coef_matrix_pb_{}_{}.csv'\
			.format(eps_, rep), index_col=0) for rep in range(n_runs)], axis=1)

	syn_pb_coef_female_df = pd.DataFrame(female_pb_coefs.values.T, columns=female_pb_coefs.index)
	syn_pb_coef_female_dict[eps] = syn_pb_coef_female_df
# For males
for eps in epsilons:
	eps_ = np.round(eps, 0)
	male_pb_coefs = pd.concat([pd.read_csv(parent_dir+'R/privbayes/male/csvs/male_coef_matrix_pb_{}_{}.csv'\
			.format(eps_, rep), index_col=0) for rep in range(n_runs)], axis=1)

	syn_pb_coef_male_df = pd.DataFrame(male_pb_coefs.values.T, columns=male_pb_coefs.index)
	syn_pb_coef_male_dict[eps] = syn_pb_coef_male_df

# Save results as pickles
pickle.dump(syn_pb_coef_female_dict, open(script_dir+'/plot_pickles/pb_female_coef_dict.p', 'wb'))
pickle.dump(syn_pb_coef_male_dict, open(script_dir+'/plot_pickles/pb_male_coef_dict.p', 'wb'))

## No alive/dead stratification
syn_no_ad_strat_coef_female_dict = {}
syn_no_ad_strat_coef_male_dict = {}
syn_no_ad_strat_p_value_female_dict = {}
syn_no_ad_strat_p_value_male_dict = {}
# For females
syn_no_ad_strat_coef_female_dict = {}
syn_no_ad_strat_p_value_female_dict = {}
for eps in epsilons:
	female_coefs = pd.concat([pd.read_csv(parent_dir+'R/ablation_study/no_death_strat/female/csvs/female_coef_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	female_p_values = pd.concat([pd.read_csv(parent_dir+'R/ablation_study/no_death_strat/female/csvs/female_p_value_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	syn_no_ad_strat_coef_female_df = pd.DataFrame(female_coefs.values.T, columns=female_coefs.index)
	syn_no_ad_strat_p_value_female_df = pd.DataFrame(female_p_values.values.T, columns=female_p_values.index)

	syn_no_ad_strat_coef_female_dict[eps] = syn_no_ad_strat_coef_female_df
	syn_no_ad_strat_p_value_female_dict[eps] = syn_no_ad_strat_p_value_female_df

# For males
syn_no_ad_strat_coef_male_dict = {}
syn_no_ad_strat_p_value_male_dict = {}
for eps in epsilons:
	male_coefs = pd.concat([pd.read_csv(parent_dir+'R/ablation_study/no_death_strat/male/csvs/male_coef_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	male_p_values = pd.concat([pd.read_csv(parent_dir+'R/ablation_study/no_death_strat/male/csvs/male_p_value_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	syn_no_ad_strat_coef_male_df = pd.DataFrame(male_coefs.values.T, columns=male_coefs.index)
	syn_no_ad_strat_p_value_male_df = pd.DataFrame(male_p_values.values.T, columns=male_p_values.index)

	syn_no_ad_strat_coef_male_dict[eps] = syn_no_ad_strat_coef_male_df
	syn_no_ad_strat_p_value_male_dict[eps] = syn_no_ad_strat_p_value_male_df

# Save results as pickles
pickle.dump(syn_no_ad_strat_coef_female_dict, open(script_dir+'/plot_pickles/no_ad_strat_female_coef_dict.p', 'wb'))
pickle.dump(syn_no_ad_strat_p_value_female_dict, open(script_dir+'/plot_pickles/no_ad_strat_female_p_value_dict.p', 'wb'))
pickle.dump(syn_no_ad_strat_coef_male_dict, open(script_dir+'/plot_pickles/no_ad_strat_male_coef_dict.p', 'wb'))
pickle.dump(syn_no_ad_strat_p_value_male_dict, open(script_dir+'/plot_pickles/no_ad_strat_male_p_value_dict.p', 'wb'))
