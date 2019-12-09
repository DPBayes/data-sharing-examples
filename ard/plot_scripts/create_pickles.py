import pandas as pd
import pickle
import numpy as np
from itertools import product


## Load true fits
real_fit_male = pd.read_csv('./plot_pickles/real_male_fit.csv')
real_male_df = pd.DataFrame(real_fit_male.iloc[:,1:].values, index=real_fit_male.iloc[:,0].values, columns=real_fit_male.columns[1:])
real_fit_female = pd.read_csv('./plot_pickles/real_female_fit.csv')
real_female_df = pd.DataFrame(real_fit_female.iloc[:,1:].values, index=real_fit_female.iloc[:,0].values, columns=real_fit_female.columns[1:])

# Load raritys
male_rarity, female_rarity = pickle.load(open(path+'plot_scripts_new/plot_pickles/raritys.p', 'rb'))

## Load DPVI fits
epsilons = [0.74, 1.99, 3.92]
epsilons = np.array(epsilons)
seeds = pd.read_csv(path+'plot_scripts_new/gen_seeds.txt', header=None).values[:,0]
n_runs = len(seeds)*10

syn_dpvi_coef_female_dict = {}
syn_dpvi_coef_male_dict = {}
syn_dpvi_p_value_female_dict = {}
syn_dpvi_p_value_male_dict = {}
# For females
syn_dpvi_coef_female_dict = {}
syn_dpvi_p_value_female_dict = {}
for eps in epsilons:
	female_coefs = pd.concat([pd.read_csv(path+'R_new/dpvi_female/dpvi_csvs/female_coef_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	female_p_values = pd.concat([pd.read_csv(path+'R_new/dpvi_female/dpvi_csvs/female_p_value_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	syn_dpvi_coef_female_df = pd.DataFrame(female_coefs.values.T, columns=female_coefs.index)
	syn_dpvi_p_value_female_df = pd.DataFrame(female_p_values.values.T, columns=female_p_values.index)

	syn_dpvi_coef_female_dict[eps] = syn_dpvi_coef_female_df
	syn_dpvi_p_value_female_dict[eps] = syn_dpvi_p_value_female_df

# For males
syn_dpvi_coef_male_dict = {}
syn_dpvi_p_value_male_dict = {}
for eps in epsilons:
	male_coefs = pd.concat([pd.read_csv(path+'R_new/dpvi_male/dpvi_csvs/male_coef_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	male_p_values = pd.concat([pd.read_csv(path+'R_new/dpvi_male/dpvi_csvs/male_p_value_matrix_dpvi_{}_{}_{}.csv'\
			.format(seed, eps, rep), index_col=0) for (seed, rep) in product(seeds, range(10))], axis=1)

	syn_dpvi_coef_male_df = pd.DataFrame(male_coefs.values.T, columns=male_coefs.index)
	syn_dpvi_p_value_male_df = pd.DataFrame(male_p_values.values.T, columns=male_p_values.index)

	syn_dpvi_coef_male_dict[eps] = syn_dpvi_coef_male_df
	syn_dpvi_p_value_male_dict[eps] = syn_dpvi_p_value_male_df

# Save results as pickles
pickle.dump(syn_dpvi_coef_female_dict, open('../plot_pickles/female_coef_dict.p', 'wb'))
pickle.dump(syn_dpvi_p_value_female_dict, open('../plot_pickles/female_p_value_dict.p', 'wb'))

pickle.dump(syn_dpvi_coef_male_dict, open('../plot_pickles/male_coef_dict.p', 'wb'))
pickle.dump(syn_dpvi_p_value_male_dict, open('../plot_pickles/male_p_value_dict.p', 'wb'))


## Load PB fits

syn_pb_coef_female_dict = {}
syn_pb_coef_male_dict = {}

# For females
for eps in epsilons:
	eps_ = np.round(eps, 0)
	female_pb_coefs = pd.concat([pd.read_csv(path+'R_new/pb_female/pb_csvs/female_coef_matrix_pb_{}_{}.csv'\
			.format(eps_, rep), index_col=0) for rep in range(n_runs)], axis=1)

	syn_pb_coef_female_df = pd.DataFrame(female_pb_coefs.values.T, columns=female_pb_coefs.index)
	syn_pb_coef_female_dict[eps] = syn_pb_coef_female_df
# For males
for eps in epsilons:
	eps_ = np.round(eps, 0)
	male_pb_coefs = pd.concat([pd.read_csv(path+'R_new/pb_male/pb_csvs/male_coef_matrix_pb_{}_{}.csv'\
			.format(eps_, rep), index_col=0) for rep in range(n_runs)], axis=1)

	syn_pb_coef_male_df = pd.DataFrame(male_pb_coefs.values.T, columns=male_pb_coefs.index)
	syn_pb_coef_male_dict[eps] = syn_pb_coef_male_df

# Save results as pickles
pickle.dump(syn_pb_coef_female_dict, open('../plot_pickles/pb_female_coef_dict.p', 'wb'))
pickle.dump(syn_pb_coef_male_dict, open('../plot_pickles/pb_male_coef_dict.p', 'wb'))
