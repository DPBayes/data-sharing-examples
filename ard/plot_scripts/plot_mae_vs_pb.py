import numpy as np
import pandas as pd
import pickle

## plot conf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
width = 8.5/2.54
height = width*(3/4)
###

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

plot_path = './'
male_rarities, female_rarities = pickle.load(open(script_dir+'/plot_pickles/raritys.p', 'rb'))

## Load true fits
real_fit_male = pd.read_csv(script_dir+'/plot_pickles/real_male_fit.csv')
real_fit_female = pd.read_csv(script_dir+'/plot_pickles/real_female_fit.csv')

coef_names_male = real_fit_male['Unnamed: 0']
real_coef_male = real_fit_male['Estimate']
real_coef_male_dict = {key : real_coef_male.iloc[i]\
								for i, key in enumerate(real_fit_male['Unnamed: 0'])}

coef_names_female = real_fit_female['Unnamed: 0']
real_coef_female = real_fit_female['Estimate']

## Load DPVI fits
epsilons = [0.74, 1.99, 3.92]
epsilons = np.array(epsilons)
n_runs = 100

# Load DP results
syn_dpvi_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/female_coef_dict.p', 'rb'))
syn_dpvi_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/male_coef_dict.p', 'rb'))

syn_pb_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/pb_female_coef_dict.p', 'rb'))
syn_pb_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/pb_male_coef_dict.p', 'rb'))

## Pick significant coefficients
p_value = 0.025
significant_coef_names_male = coef_names_male[(real_fit_male['Pr(>|z|)']<p_value).values]
significant_coef_names_female = coef_names_female[(real_fit_female['Pr(>|z|)']<p_value).values]

for key, value in significant_coef_names_male.items():
	if 'shp' in value:
		significant_coef_names_male.pop(key)

for key, value in significant_coef_names_female.items():
	if 'shp' in value:
		significant_coef_names_female.pop(key)

real_significant_male = real_coef_male[(real_fit_male['Pr(>|z|)']<p_value).values]
real_significant_male = pd.DataFrame(real_significant_male[significant_coef_names_male.index].values[\
							np.newaxis], 
							columns=significant_coef_names_male.values)

real_significant_female = real_coef_female[(real_fit_female['Pr(>|z|)']<p_value).values]
real_significant_female = pd.DataFrame(real_significant_female[significant_coef_names_female.index].values[\
							np.newaxis], 
							columns=significant_coef_names_female.values)
for key in real_significant_female.columns:
	if 'shp' in key:
		real_significant_female.pop(key)


# Significant coefs for DPVI
# males
syn_dpvi_significant_male_dict = {eps : syn_dpvi_coef_male_dict[eps][significant_coef_names_male] \
		for eps in epsilons}
syn_dpvi_significant_male_mean = {eps : syn_dpvi_significant_male_dict[eps].mean(0) \
		for eps in epsilons}
syn_dpvi_significant_male_sem = {eps : syn_dpvi_significant_male_dict[eps].std(0)/np.sqrt(n_runs) \
		for eps in epsilons}

# females
syn_dpvi_significant_female_dict = {eps : syn_dpvi_coef_female_dict[eps][significant_coef_names_female] \
		for eps in epsilons}
syn_dpvi_significant_female_mean = {eps : syn_dpvi_significant_female_dict[eps].mean(0) \
		for eps in epsilons}
syn_dpvi_significant_female_sem = {eps : syn_dpvi_significant_female_dict[eps].std(0)/np.sqrt(n_runs) \
		for eps in epsilons}

# Significant coefs for PB
# males
syn_pb_significant_male_dict = {eps : syn_pb_coef_male_dict[eps][significant_coef_names_male] \
		for eps in epsilons}
syn_pb_significant_male_mean = {eps : syn_pb_significant_male_dict[eps].mean(0) \
		for eps in epsilons}
syn_pb_significant_male_sem = {eps : syn_pb_significant_male_dict[eps].std(0)/np.sqrt(n_runs) \
		for eps in epsilons}

# females
syn_pb_significant_female_dict = {eps : syn_pb_coef_female_dict[eps][significant_coef_names_female] \
		for eps in epsilons}
syn_pb_significant_female_mean = {eps : syn_pb_significant_female_dict[eps].mean(0) \
		for eps in epsilons}
syn_pb_significant_female_sem = {eps : syn_pb_significant_female_dict[eps].std(0)/np.sqrt(n_runs) \
		for eps in epsilons}

## Plot coefficients
fig, axis = plt.subplots(figsize=(width, height))

# Males
# DPVI
mean_diffs_dpvi_male = []
std_diffs_dpvi_male = []
for j, eps in enumerate(epsilons):
	diff_dpvi = np.linalg.norm(syn_dpvi_significant_male_dict[eps]-real_significant_male.values[0], 1, axis=1)
	diff_dpvi = diff_dpvi/real_significant_male.shape[1]
	mean_diff_dpvi = np.mean(diff_dpvi)
	std_diff_dpvi = np.std(diff_dpvi)/np.sqrt(n_runs)
	mean_diffs_dpvi_male.append(mean_diff_dpvi)
	std_diffs_dpvi_male.append(std_diff_dpvi)
axis.errorbar(np.round(epsilons, 0), mean_diffs_dpvi_male, yerr=std_diffs_dpvi_male,\
		label='Male, Mixture model', color='cyan')
## PB
mean_diffs_pb_male = []
std_diffs_pb_male = []
for j, eps in enumerate(epsilons):
	diff_pb = np.linalg.norm(syn_pb_significant_male_dict[eps]-real_significant_male.values[0], 1, axis=1)
	diff_pb = diff_pb/real_significant_male.shape[1]
	mean_diff_pb = np.mean(diff_pb)
	std_diff_pb = np.std(diff_pb)/np.sqrt(n_runs)
	mean_diffs_pb_male.append(mean_diff_pb)
	std_diffs_pb_male.append(std_diff_pb)
axis.errorbar(np.round(epsilons, 0), mean_diffs_pb_male, yerr=std_diffs_pb_male, label='Male, Bayes network',\
			color='cyan', linestyle=(0, (5, 10)))

## Females
# DPVI
mean_diffs_dpvi_female = []
std_diffs_dpvi_female = []
for j, eps in enumerate(epsilons):
	diff_dpvi = np.linalg.norm(syn_dpvi_significant_female_dict[eps]-real_significant_female.values[0], 1, axis=1)
	diff_dpvi = diff_dpvi/real_significant_female.shape[1]
	mean_diff_dpvi = np.mean(diff_dpvi)
	std_diff_dpvi = np.std(diff_dpvi)/np.sqrt(n_runs)
	mean_diffs_dpvi_female.append(mean_diff_dpvi)
	std_diffs_dpvi_female.append(std_diff_dpvi)
axis.errorbar(np.round(epsilons, 0), mean_diffs_dpvi_female, yerr=std_diffs_dpvi_female, label='Female, Mixture model',\
		color='magenta')

# PB
mean_diffs_pb_female = []
std_diffs_pb_female = []
for j, eps in enumerate(epsilons):
	diff_pb = np.linalg.norm(syn_pb_significant_female_dict[eps]-real_significant_female.values[0], 1, axis=1)
	diff_pb = diff_pb/real_significant_female.shape[1]
	mean_diff_pb = np.mean(diff_pb)
	std_diff_pb = np.std(diff_pb)/np.sqrt(n_runs)
	mean_diffs_pb_female.append(mean_diff_pb)
	std_diffs_pb_female.append(std_diff_pb)
axis.errorbar(np.round(epsilons, 0), mean_diffs_pb_female, yerr=std_diffs_pb_female, label='Female, Bayes network',\
		color='magenta', linestyle=(0, (5, 10)))
axis.set_xlabel(r'$\epsilon$')
axis.set_ylabel('MAE')

#handles, labels = axis.get_legend_handles_labels()
#handles[1].lines[0]._linestyle = (0, (5, 1))
#handles[1].lines[0]._linestyle = "-"
#handles[3].__dict__["lines"][0]._linestyle = "---"
fig.legend(loc=(.48,.4))
axis.set_title("ARD")

#plt.show()
plt.savefig(plot_path+'diabetes_mae_vs_pb_new.pdf', format='pdf', bbox_inches='tight')
plt.close()
