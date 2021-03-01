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
male_rarity, female_rarity = pickle.load(open(script_dir+'/plot_pickles/raritys.p', 'rb'))

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

# Cleaner coef_names
male_names = []
for value in significant_coef_names_male.values:
	if 'lex' in value:
		male_names.append('lex.dur : '+value[value.find('))')+2:])
	elif 'DM' in value:
		male_names.append(value.replace('DM.type', 'DM.type : '))
	elif 'shp' in value:
		male_names.append(value.replace('factor(shp)', 'shp'))
	elif '.i.cancer' in value:
		male_names.append(value.replace('.i.cancer', '.i.cancer : '))
	elif 'C10' in value:	
		male_names.append(value.replace('TRUE', ''))
	elif 'per' in value:	
		male_names.append('per.cat')

female_names = []
for value in significant_coef_names_female.values:
	if 'lex' in value:
		female_names.append('lex.dur : '+value[value.find('))')+2:])
	elif 'DM' in value:
		female_names.append(value.replace('DM.type', 'DM.type : '))
	elif 'shp' in value:
		female_names.append(value.replace('factor(shp)', 'shp'))
	elif '.i.cancer' in value:
		female_names.append(value.replace('.i.cancer', '.i.cancer : '))
	elif 'C10' in value:	
		female_names.append(value.replace('TRUE', ''))
	elif 'per' in value:	
		female_names.append('per.cat')

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


from collections import OrderedDict as od
female_significant_rarity = od({key:female_rarity[key] for key in list(significant_coef_names_female)})
female_significant_rarity_list = list(female_significant_rarity.values())
real_significant_female = real_significant_female[list(significant_coef_names_female)]
male_significant_rarity = od({key:male_rarity[key] for key in list(significant_coef_names_male)})
male_significant_rarity_list = list(male_significant_rarity.values())
real_significant_male = real_significant_male[list(significant_coef_names_male)]

## Join male and female
inverse = True
#aggregation = "median"
aggregation = "mean"
fig, axis = plt.subplots(figsize=(width, height))
rarity_list = female_significant_rarity_list+male_significant_rarity_list
rarity = np.sort(rarity_list)
groups = np.split(np.array(sorted(rarity_list)), 4) 
for eps in epsilons:
	diff_female = np.abs(syn_dpvi_significant_female_dict[eps].values - \
				real_significant_female.values.flatten())
	diff_male = np.abs(syn_dpvi_significant_male_dict[eps].values - \
				real_significant_male.values.flatten())
	diff = np.concatenate([diff_female, diff_male], axis=1)
	diff = diff[:, np.argsort(rarity_list)]
	group_accs_mean = np.mean(np.mean(np.split(diff, len(groups), axis=1), axis=-1), axis=1)
	group_accs_sem = np.std(np.mean(np.split(diff, len(groups), axis=1), axis=-1), axis=1)/np.sqrt(n_runs)
	axis.errorbar(np.min(groups, 1), group_accs_mean, yerr=group_accs_sem, label='$\epsilon={}$'.format(round(eps, 0)))

# plot 1/n effect
ax2 = axis.twinx()
if inverse==False:
	if aggregation == "median":
		asymptotic_error = 1./np.median(groups, axis=1)
	if aggregation == "mean":
		asymptotic_error = 1./np.mean(groups, axis=1)
else:
	if aggregation == "median":
		asymptotic_error = np.median(1./np.array(groups), axis=1)
	if aggregation == "mean":
		asymptotic_error = np.mean(1./np.array(groups), axis=1)
ax2.plot(np.min(groups, 1), asymptotic_error, linestyle="dashed", color="black", label="1/n")
ax2.set_ylabel("Inverse group size")
##
axis.set_xlabel('Number of cases')
axis.set_ylabel('Error')
axis.set_xticks(np.min(np.split(rarity, len(groups)), axis=1))
axis.set_xticklabels(["[{},{})".format(groups[i][0], groups[i+1][0]) if i<3 else\
			"[{},{})".format(groups[i][0], groups[i][-1]) for i in range(len(groups))], fontsize=6)

h1, l1 = axis.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
axis.legend(h1+h2, l1+l2)
#axis.legend()
axis.set_title("ARD")
if inverse==False:
	plt.savefig(plot_path + 'rarity_opt_{}.pdf'.format(aggregation), format='pdf', bbox_inches='tight')
else:
	plt.savefig(plot_path + 'rarity_opt_inverse_{}.pdf'.format(aggregation), format='pdf', bbox_inches='tight')
#plt.show()
plt.savefig('rarity_opt_{}.png'.format(aggregation), dpi=300, bbox_inches='tight')
plt.close()
