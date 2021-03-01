import pickle
import numpy as np
import pandas as pd

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

## Load DPVI fits
epsilons = [0.74, 1.99, 3.92]
epsilons = np.array(epsilons)
n_runs = 100

# Load DP results
syn_dpvi_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/female_coef_dict.p', 'rb'))
syn_dpvi_p_value_female_dict = pickle.load(open(script_dir+'/plot_pickles/female_p_value_dict.p', 'rb'))

syn_dpvi_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/male_coef_dict.p', 'rb'))
syn_dpvi_p_value_male_dict = pickle.load(open(script_dir+'/plot_pickles/male_p_value_dict.p', 'rb'))

## Plot conclusions
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

## Plot males
fig, axis = plt.subplots(figsize=(width, height))
column_width = 0.24

for j, eps in enumerate(epsilons):
	males = male_ct[eps]
	male_cumsum = np.cumsum([males[label] for label in order])
	j = 2*j
	for i, lab in enumerate(order):
		label = label_map[lab]
		if j == 0:
			if i==3:
				rects = axis.bar(j, males[lab], column_width, label=label, bottom = male_cumsum[i-1], color=cmap[lab])
			elif i>0:	
				axis.bar(j, males[lab], column_width, label=label, bottom = male_cumsum[i-1], color=cmap[lab])
			else : 
				axis.bar(j, males[lab], column_width, label=label, color=cmap[lab])
		else:
			if i==3:
				rects = axis.bar(j, males[lab], column_width, bottom = male_cumsum[i-1], color=cmap[lab])
			elif i>0:	
				axis.bar(j, males[lab], column_width, bottom = male_cumsum[i-1], color=cmap[lab])
			else : 
				axis.bar(j, males[lab], column_width, color=cmap[lab])
	for rect in rects:
		axis.annotate('Combined', xy = (rect.get_x()+rect.get_width()/2, 0), xytext=(0,10), textcoords='offset points',\
				rotation='vertical', va='bottom', ha='center', color='w')
	males_sep = male_ct_sep[eps]
	for name_iter, coef_name in enumerate(dm_names):
		for i, lab in enumerate(order):
			male_cumsum_sep = np.cumsum([males_sep[label][coef_name] for label in order])
			label = label_map[lab]
			if j == 0:
				if i == 3:
					rects = axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name],  column_width, bottom = male_cumsum_sep[i-1], color=cmap[lab])
				elif i>0:	
					axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name],  column_width, bottom = male_cumsum_sep[i-1], color=cmap[lab])
				else : 
					axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name],  column_width, color=cmap[lab])
			else:
				if i==3:	
					rects = axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name],  column_width, bottom = male_cumsum_sep[i-1], color=cmap[lab])
				elif i>0:	
					axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name],  column_width, bottom = male_cumsum_sep[i-1], color=cmap[lab])
				else : 
					axis.bar(j-1*(name_iter+1)/4, males_sep[lab][coef_name],  column_width, color=cmap[lab])
		for rect in rects:
			axis.annotate(coef_name.replace('DM.type','')+' ({}) '.format(male_rarities[coef_name]), xy = (rect.get_x()+rect.get_width()/2, 0), xytext=(0,10),textcoords='offset points',\
										rotation='vertical', va='bottom', ha='center', color='w')


fig.legend(loc='lower center', ncol=2)
axis.set_yticks(np.arange(0, n_runs+1, 20))
axis.set_yticklabels(np.arange(0, n_runs+1, 20))
axis.set_xticks([-0.375, 1.625, 3.625])
axis.set_xticklabels(np.round(epsilons, 0))
axis.set_xlabel(r'$\epsilon$')
axis.set_ylabel('Percentage of repeats')
fig.subplots_adjust(bottom=0.25)
plt.savefig(plot_path+'male_conclusions_bar_sep_new.pdf', format='pdf', bbox_inches='tight')
plt.close()

## Plot females
fig, axis = plt.subplots(figsize=(width, height))
for j, eps in enumerate(epsilons):
	females = female_ct[eps]
	female_cumsum = np.cumsum([females[label] for label in order])
	j = 2*j
	for i, lab in enumerate(order):
		label = label_map[lab]
		if j == 0:
			if i==3:
				rects = axis.bar(j, females[lab], column_width, label=label, bottom = female_cumsum[i-1], color=cmap[lab])
			elif i>0:	
				axis.bar(j, females[lab], column_width, label=label, bottom = female_cumsum[i-1], color=cmap[lab])
			else : 
				axis.bar(j, females[lab], column_width, label=label, color=cmap[lab])
		else:
			if i==3:
				rects = axis.bar(j, females[lab], column_width, bottom = female_cumsum[i-1], color=cmap[lab])
			elif i>0:	
				axis.bar(j, females[lab], column_width, bottom = female_cumsum[i-1], color=cmap[lab])
			else : 
				axis.bar(j, females[lab], column_width, color=cmap[lab])
	for rect in rects:
		axis.annotate('Combined', xy = (rect.get_x()+rect.get_width()/2, 0), xytext=(0,10), textcoords='offset points',\
				rotation='vertical', va='bottom', ha='center', color='w')
	females_sep = female_ct_sep[eps]
	for name_iter, coef_name in enumerate(dm_names):
		for i, lab in enumerate(order):
			female_cumsum_sep = np.cumsum([females_sep[label][coef_name] for label in order])
			label = label_map[lab]
			if j == 0:
				if i == 3:
					rects = axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name],  column_width, bottom = female_cumsum_sep[i-1], color=cmap[lab])
				elif i>0:	
					axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name],  column_width, bottom = female_cumsum_sep[i-1], color=cmap[lab])
				else : 
					axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name],  column_width, color=cmap[lab])
			else:
				if i==3:	
					rects = axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name],  column_width, bottom = female_cumsum_sep[i-1], color=cmap[lab])
				elif i>0:	
					axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name],  column_width, bottom = female_cumsum_sep[i-1], color=cmap[lab])
				else : 
					axis.bar(j-1*(name_iter+1)/4, females_sep[lab][coef_name],  column_width, color=cmap[lab])
		for rect in rects:
			axis.annotate(coef_name.replace('DM.type','')+' ({}) '.format(female_rarities[coef_name]), xy = (rect.get_x()+rect.get_width()/2, 0), xytext=(0,10),textcoords='offset points',\
										rotation='vertical', va='bottom', ha='center', color='w')


fig.legend(loc='lower center', ncol=2)
axis.set_yticks(np.arange(0, n_runs+1, 20))
axis.set_yticklabels(np.arange(0, n_runs+1, 20))
axis.set_xticks([-0.375, 1.625, 3.625])
axis.set_xticklabels(np.round(epsilons, 0))
axis.set_xlabel(r'$\epsilon$')
axis.set_ylabel('Percentage of repeats')
axis.set_title('Female')
fig.subplots_adjust(bottom=0.25)
plt.savefig(plot_path+'female_conclusions_bar_sep_new.pdf', format='pdf', bbox_inches='tight')
plt.close()
