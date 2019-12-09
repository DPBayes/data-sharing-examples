import pickle
import numpy as np
import pandas as pd
import torch
from itertools import product


path = '/u/57/jalkoj1/unix/dp-data-sharing/pnas_code/diabetes/'
plot_path = '/u/57/jalkoj1/unix/dp-data-sharing/tex/pnas/figures/'
male_rarities, female_rarities = pickle.load(open(path+'plot_scripts/plot_pickles/raritys.p', 'rb'))

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

## Plot conclusions
import matplotlib.pyplot as plt
from itertools import product

labels = ['Type 4', 'Type 3', 'Type 2', 'Type 1'][::-1]
dm_names = [name for name in list(list(syn_dpvi_coef_female_dict.values())[0].columns) if 'DM' in name]
significance_level = 0.05

# create crosstables
male_ct = {}
for k, eps in enumerate(epsilons):
	male_dict = {label : 0 for label in labels}
	for coef, p_value in zip(syn_dpvi_coef_male_dict[eps][dm_names].values,\
			syn_dpvi_p_value_male_dict[eps][dm_names].values):
		if np.all(p_value<significance_level) and np.all(coef>0):
			male_dict["Type 1"] += 1 # correct discovery
		elif np.any(p_value>significance_level) and np.all(coef>0):
			male_dict["Type 2"] += 1 # correct sign, not significant
		elif np.all(p_value<significance_level) and np.any(coef<0):
			male_dict["Type 3"] += 1 # wrong sign with significance
		else:
			male_dict["Type 4"] += 1 # wrong sing, not significant
	male_ct[eps] = male_dict

male_ct_sep = {}
for k, eps in enumerate(epsilons):
	male_dict = {label : {coef_name : 0 for coef_name in dm_names} for label in labels}
	for coefs, p_values in zip(syn_dpvi_coef_male_dict[eps][dm_names].values,\
			syn_dpvi_p_value_male_dict[eps][dm_names].values):
		for jj, (coef, p_value) in enumerate(zip(coefs, p_values)):
			coef_name = dm_names[jj]
			if np.all(p_value<significance_level) and np.all(coef>0):
				male_dict["Type 1"][coef_name] += 1 # correct discovery
			elif np.any(p_value>significance_level) and np.all(coef>0):
				male_dict["Type 2"][coef_name] += 1 # correct sign, not significant
			elif np.all(p_value<significance_level) and np.any(coef<0):
				male_dict["Type 3"][coef_name] += 1 # wrong sign with significance
			else:
				male_dict["Type 4"][coef_name] += 1 # wrong sing, not significant
	male_ct_sep[eps] = male_dict

female_ct = {}
for k, eps in enumerate(epsilons):
	female_dict = {label : 0 for label in labels}
	for coef, p_value in zip(syn_dpvi_coef_female_dict[eps][dm_names].values,\
			syn_dpvi_p_value_female_dict[eps][dm_names].values):
		if np.all(p_value<significance_level) and np.all(coef>0):
			female_dict["Type 1"] += 1
		elif np.any(p_value>significance_level) and np.all(coef>0):
			female_dict["Type 2"] += 1
		elif np.all(p_value<significance_level) and np.any(coef<0):
			female_dict["Type 3"] += 1
		else:
			female_dict["Type 4"] += 1
	female_ct[eps] = female_dict

female_ct_sep = {}
for k, eps in enumerate(epsilons):
	female_dict = {label : {coef_name : 0 for coef_name in dm_names} for label in labels}
	for coefs, p_values in zip(syn_dpvi_coef_female_dict[eps][dm_names].values,\
			syn_dpvi_p_value_female_dict[eps][dm_names].values):
		for jj, (coef, p_value) in enumerate(zip(coefs, p_values)):
			coef_name = dm_names[jj]
			if np.all(p_value<significance_level) and np.all(coef>0):
				female_dict["Type 1"][coef_name] += 1 # correct discovery
			elif np.any(p_value>significance_level) and np.all(coef>0):
				female_dict["Type 2"][coef_name] += 1 # correct sign, not significant
			elif np.all(p_value<significance_level) and np.any(coef<0):
				female_dict["Type 3"][coef_name] += 1 # wrong sign with significance
			else:
				female_dict["Type 4"][coef_name] += 1 # wrong sing, not significant
	female_ct_sep[eps] = female_dict

cmap = {"Type 1" : 'g', "Type 2" : 'b',  "Type 4": 'orange', "Type 3" : 'r'}
label_map = {"Type 1" : "Reproduced discovery", "Type 2" : "Correct sign, p>0.05",\
		"Type 3" : "Wrong sign, p<0.05",\
		"Type 4" : "Wrong sign, p>0.05"}
order  = ["Type 1", "Type 2",  "Type 4", "Type 3"]

### PLOT
plt.rcParams.update({'font.size': 7})
width = 8.7/2.54
height = width*(3/4)
fig, axis = plt.subplots(figsize=(width, height))
j = 0
eps = epsilons[j]
males = male_ct[eps]
male_cumsum = np.cumsum([males[label] for label in order])
# plot combined
for i, lab in enumerate(order):
	label = label_map[lab]
	if i==3:
		rects = axis.bar(j, males[lab],0.24, label=label, bottom = male_cumsum[i-1], color=cmap[lab])
	elif i>0:	
		axis.bar(j, males[lab],0.24, label=label, bottom = male_cumsum[i-1], color=cmap[lab])
	else : 
		axis.bar(j, males[lab],0.24, label=label, color=cmap[lab])
for rect in rects:
	axis.annotate('Combined', xy = (rect.get_x()+rect.get_width()/2, 0), xytext=(0,10), textcoords='offset points',\
			rotation='vertical', va='bottom', ha='center', color='w')
# plot separate
males_sep = male_ct_sep[eps]
for name_iter, coef_name in enumerate(dm_names):
	for i, lab in enumerate(order):
		male_cumsum_sep = np.cumsum([males_sep[label][coef_name] for label in order])
		label = label_map[lab]
		if i == 3:
			rects = axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name], 0.24,\
					bottom = male_cumsum_sep[i-1], color=cmap[lab])
		elif i>0:	
			axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name], 0.24,\
					bottom = male_cumsum_sep[i-1], color=cmap[lab])
		else : 
			axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name], 0.24, color=cmap[lab])
	for rect in rects:
		axis.annotate(coef_name.replace('DM.type','')+' ({}) '.format(male_rarities[coef_name]),\
				xy = (rect.get_x()+rect.get_width()/2, 0), xytext=(0,10),textcoords='offset points',\
				rotation='vertical', va='bottom', ha='center', color='w')



eps = epsilons[0]
j = 1
females = female_ct[eps]
female_cumsum = np.cumsum([females[label] for label in order])
j = 2*j
# plot combined
for i, lab in enumerate(order):
	label = label_map[lab]
	if i==3:
		rects = axis.bar(j, females[lab],0.24, bottom = female_cumsum[i-1], color=cmap[lab])
	elif i>0:	
		axis.bar(j, females[lab],0.24, bottom = female_cumsum[i-1], color=cmap[lab])
	else : 
		axis.bar(j, females[lab],0.24, color=cmap[lab])
for rect in rects:
	axis.annotate('Combined', xy = (rect.get_x()+rect.get_width()/2, 0), xytext=(0,10), textcoords='offset points',\
			rotation='vertical', va='bottom', ha='center', color='w')
# plot separate
females_sep = female_ct_sep[eps]
for name_iter, coef_name in enumerate(dm_names):
	for i, lab in enumerate(order):
		female_cumsum_sep = np.cumsum([females_sep[label][coef_name] for label in order])
		label = label_map[lab]
		if i == 3:
			rects = axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name], 0.24,\
					bottom = female_cumsum_sep[i-1], color=cmap[lab])
		elif i>0:	
			axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name], 0.24,\
					bottom = female_cumsum_sep[i-1], color=cmap[lab])
		else : 
			axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name], 0.24, color=cmap[lab])
	for rect in rects:
		axis.annotate(coef_name.replace('DM.type','')+' ({}) '.format(female_rarities[coef_name]),\
				xy = (rect.get_x()+rect.get_width()/2, 0), xytext=(0,10),textcoords='offset points',\
				rotation='vertical', va='bottom', ha='center', color='w')


fig.legend(loc='lower center', ncol=2)
axis.set_yticks(np.arange(0, n_runs+1, 20))
axis.set_yticklabels(np.arange(0, n_runs+1, 20))
axis.set_ylabel('Percentage among 100 independent runs')
axis.set_xticks([-0.375, 1.625])
axis.set_xticklabels(["Male", "Female"])
fig.subplots_adjust(bottom=0.25)
plt.savefig(plot_path + 'conclusions_bars_both.pdf', format='pdf', bbox_inches='tight')
plt.close()
