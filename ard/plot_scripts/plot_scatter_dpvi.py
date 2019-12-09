import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from itertools import product

path = '/u/57/jalkoj1/unix/dp-data-sharing/pnas_code/diabetes/'
plot_path = '/u/57/jalkoj1/unix/dp-data-sharing/tex/pnas/figures/'

## Load true fits
real_fit_male = pd.read_csv(path+'plot_scripts_new/plot_pickles/real_male_fit.csv')
real_fit_female = pd.read_csv(path+'plot_scripts_new/plot_pickles/real_female_fit.csv')

coef_names_male = real_fit_male['Unnamed: 0']
real_coef_male = real_fit_male['Estimate']
real_std_male = real_fit_male['Std. Error']
real_coef_male_dict = {key : real_coef_male.iloc[i]\
								for i, key in enumerate(real_fit_male['Unnamed: 0'])}

coef_names_female = real_fit_female['Unnamed: 0']
real_coef_female = real_fit_female['Estimate']
real_std_female = real_fit_female['Std. Error']


## Load DPVI fits
epsilons = [0.74, 1.99, 3.92]
epsilons = np.array(epsilons)
seeds = pd.read_csv(path+'plot_scripts_new/gen_seeds.txt', header=None).values[:,0]
n_runs = len(seeds)*10

# Load DP results
syn_dpvi_coef_female_dict = pickle.load(open('../plot_pickles/female_coef_dict.p', 'rb'))
syn_dpvi_p_value_female_dict = pickle.load(open('../plot_pickles/female_p_value_dict.p', 'rb'))

syn_dpvi_coef_male_dict = pickle.load(open('../plot_pickles/male_coef_dict.p', 'rb'))
syn_dpvi_p_value_male_dict = pickle.load(open('../plot_pickles/male_p_value_dict.p', 'rb'))

## Pick significant coefficients
p_value = 0.025
# Males
significant_coef_names_male = coef_names_male[(real_fit_male['Pr(>|z|)']<p_value).values]
for key, value in significant_coef_names_male.items():
	if 'shp' in value:
		significant_coef_names_male.pop(key)


real_significant_male = real_coef_male[(real_fit_male['Pr(>|z|)']<p_value).values]
real_significant_male = pd.DataFrame(real_significant_male[significant_coef_names_male.index].values[\
							np.newaxis], 
							columns=significant_coef_names_male.values)
real_significant_male_std = real_std_male[(real_fit_male['Pr(>|z|)']<p_value).values]
real_significant_male_std = pd.DataFrame(real_significant_male_std[significant_coef_names_male.index].values[\
							np.newaxis], columns=significant_coef_names_male.values)

# Females
significant_coef_names_female = coef_names_female[(real_fit_female['Pr(>|z|)']<p_value).values]
for key, value in significant_coef_names_female.items():
	if 'shp' in value:
		significant_coef_names_female.pop(key)

real_significant_female = real_coef_female[(real_fit_female['Pr(>|z|)']<p_value).values]
real_significant_female = pd.DataFrame(real_significant_female[significant_coef_names_female.index].values[\
							np.newaxis], 
							columns=significant_coef_names_female.values)

real_significant_female_std = real_std_female[(real_fit_female['Pr(>|z|)']<p_value).values]
real_significant_female_std = pd.DataFrame(real_significant_female_std[significant_coef_names_female.index].values[\
							np.newaxis], columns=significant_coef_names_female.values)

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

male_confounding_indx = 0
for i_name, name in enumerate(male_names):
	if 'DM' in name:
		continue
	else:
		male_names[i_name] = r'$z_{{{0}}}$'.format(male_confounding_indx)
		male_confounding_indx += 1	


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

female_confounding_indx = 0
for i_name, name in enumerate(female_names):
	if 'DM' in name:
		continue
	else:
		female_names[i_name] = r'$z_{{{0}}}$'.format(female_confounding_indx)
		female_confounding_indx += 1	

## Significant coefs for DPVI
# Males
syn_dpvi_significant_male_dict = {eps : syn_dpvi_coef_male_dict[eps][significant_coef_names_male] \
		for eps in epsilons}
syn_dpvi_significant_male_mean = {eps : syn_dpvi_significant_male_dict[eps].mean(0) \
		for eps in epsilons}
syn_dpvi_significant_male_sem = {eps : syn_dpvi_significant_male_dict[eps].std(0)/np.sqrt(n_runs) \
		for eps in epsilons}

# Females
syn_dpvi_significant_female_dict = {eps : syn_dpvi_coef_female_dict[eps][significant_coef_names_female] \
		for eps in epsilons}
syn_dpvi_significant_female_mean = {eps : syn_dpvi_significant_female_dict[eps].mean(0) \
		for eps in epsilons}
syn_dpvi_significant_female_sem = {eps : syn_dpvi_significant_female_dict[eps].std(0)/np.sqrt(n_runs) \
		for eps in epsilons}


## PLOT
plt.rcParams.update({'font.size': 7})
width = 8.7/2.54
height = width*(3/4)
# For females
fig, ax = plt.subplots(figsize=(width, height))
epsilon = epsilons[0]
ax.errorbar(syn_dpvi_significant_female_mean[epsilon], range(len(significant_coef_names_female)), \
			 xerr=syn_dpvi_significant_female_sem[epsilon], fmt='o', color='r', capsize=4,\
			 markersize=3, label='Synthetic data')
ax.errorbar(real_significant_female.values[0], range(len(significant_coef_names_female)),\
			 xerr=0.0*real_significant_female_std.values[0], fmt='o',color='g', capsize=0,\
			 markersize=5, label='Original')
ax.grid()
ax.set_yticks(range(len(female_names)))
female_clean_names = [name.replace('DM.type : ', '') if 'DM' in name else name for name in female_names ]
ax.set_yticklabels(female_clean_names)
for ii, ytick in enumerate(ax.get_yticklabels()):
	if 'DM' in male_names[ii]:
		col = 'red'
	else:
		col = 'black'
	ytick.set_color(col)
ax.set_ylabel('Coefficient')
ax.set_xlabel('Value')	
plt.legend(loc='best')
plt.title('Females, $\epsilon={}$'.format(np.round(epsilon,0)))
plt.savefig(plot_path+'female_dpvi_scatter_single_new.pdf', format='pdf', bbox_inches='tight')
plt.close()

# For males
fig, ax = plt.subplots(figsize=(width, height))
ax.errorbar(syn_dpvi_significant_male_mean[epsilon], range(len(significant_coef_names_male)), \
			 xerr=syn_dpvi_significant_male_sem[epsilon], fmt='o', color='r', capsize=4,\
			 markersize=3, label='Synthetic data')
ax.errorbar(real_significant_male.values[0], range(len(significant_coef_names_male)),\
			 xerr=0.0*real_significant_male_std.values[0], fmt='o',color='g', capsize=0,\
			 markersize=5, label='Original')
ax.grid()
ax.set_yticks(range(len(male_names)))
male_clean_names = [name.replace('DM.type : ', '') if 'DM' in name else name for name in male_names ]
ax.set_yticklabels(male_clean_names)
for ii, ytick in enumerate(ax.get_yticklabels()):
	if 'DM' in male_names[ii]:
		col = 'red'
	else:
		col = 'black'
	ytick.set_color(col)
ax.set_ylabel('Coefficient')
ax.set_xlabel('Value')	
plt.legend(loc='best')
plt.title('Males, $\epsilon={}$'.format(np.round(epsilon,0)))
plt.savefig(plot_path+'male_dpvi_scatter_single_new.pdf', format='pdf', bbox_inches='tight')
plt.close()

