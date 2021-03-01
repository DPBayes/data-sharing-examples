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

## Load true fits
real_fit_male = pd.read_csv(script_dir+'/plot_pickles/real_male_fit.csv')
real_male_df = pd.DataFrame(real_fit_male.iloc[:,1:].values, index=real_fit_male.iloc[:,0].values, columns=real_fit_male.columns[1:])
real_fit_female = pd.read_csv(script_dir+'/plot_pickles/real_female_fit.csv')
real_female_df = pd.DataFrame(real_fit_female.iloc[:,1:].values, index=real_fit_female.iloc[:,0].values, columns=real_fit_female.columns[1:])


## Load DPVI fits
epsilons = [0.74, 1.99, 3.92]
epsilons = np.array(epsilons)
n_runs = 100

# Load DP results, stratified
syn_dpvi_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/female_coef_dict.p', 'rb'))
syn_dpvi_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/male_coef_dict.p', 'rb'))

syn_dpvi_p_value_female_dict = pickle.load(open(script_dir+'/plot_pickles/female_p_value_dict.p', 'rb'))
syn_dpvi_p_value_male_dict = pickle.load(open(script_dir+'/plot_pickles/male_p_value_dict.p', 'rb'))

dm_names = [name for name in real_male_df.index if 'DM' in name]

female_dm_means = {eps : syn_dpvi_coef_female_dict[eps][dm_names].mean(0).round(3) for eps in epsilons}
female_dm_sems = {eps : (syn_dpvi_coef_female_dict[eps][dm_names].std(0)).round(3) for eps in epsilons}

male_dm_means = {eps : syn_dpvi_coef_male_dict[eps][dm_names].mean(0).round(3) for eps in epsilons}
male_dm_sems = {eps : (syn_dpvi_coef_male_dict[eps][dm_names].std(0)).round(3) for eps in epsilons}


# Load DP results, no stratification at all
#males
syn_total_no_strat_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/no_strat_male_coef_dict.p', 'rb'))
syn_total_no_strat_p_value_male_dict = pickle.load(open(script_dir+'/plot_pickles/no_strat_male_p_value_dict.p', 'rb'))

male_total_no_strat_dm_means = {eps : syn_total_no_strat_coef_male_dict[eps][dm_names].mean(0).round(3) for eps in epsilons}
male_total_no_strat_dm_sems = {eps : (syn_total_no_strat_coef_male_dict[eps][dm_names].std(0)).round(3) for eps in epsilons}

#females
syn_total_no_strat_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/no_strat_female_coef_dict.p', 'rb'))
syn_total_no_strat_p_value_female_dict = pickle.load(open(script_dir+'/plot_pickles/no_strat_female_p_value_dict.p', 'rb'))

female_total_no_strat_dm_means = {eps : syn_total_no_strat_coef_female_dict[eps][dm_names].mean(0).round(3) for eps in epsilons}
female_total_no_strat_dm_sems = {eps : (syn_total_no_strat_coef_female_dict[eps][dm_names].std(0)).round(3) for eps in epsilons}

# Load DP results, no alive death stratification
#males
syn_no_ad_strat_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/no_ad_strat_male_coef_dict.p', 'rb'))
syn_no_ad_strat_p_value_male_dict = pickle.load(open(script_dir+'/plot_pickles/no_ad_strat_male_p_value_dict.p', 'rb'))

male_no_ad_strat_dm_means = {eps : syn_no_ad_strat_coef_male_dict[eps][dm_names].mean(0).round(3) for eps in epsilons}
male_no_ad_strat_dm_sems = {eps : (syn_no_ad_strat_coef_male_dict[eps][dm_names].std(0)).round(3) for eps in epsilons}

#females
syn_no_ad_strat_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/no_ad_strat_female_coef_dict.p', 'rb'))
syn_no_ad_strat_p_value_female_dict = pickle.load(open(script_dir+'/plot_pickles/no_ad_strat_female_p_value_dict.p', 'rb'))

female_no_ad_strat_dm_means = {eps : syn_no_ad_strat_coef_female_dict[eps][dm_names].mean(0).round(3) for eps in epsilons}
female_no_ad_strat_dm_sems = {eps : (syn_no_ad_strat_coef_female_dict[eps][dm_names].std(0)).round(3) for eps in epsilons}
#####################
# plot bars

labels = ['Type 4', 'Type 3', 'Type 2', 'Type 1'][::-1]
dm_names = [name for name in list(list(syn_dpvi_coef_female_dict.values())[0].columns) if 'DM' in name]
significance_level = 0.05

# create crosstables

def create_ct(coef_dict, p_value_dict):
	ct = {}
	for eps in epsilons:
		dictionary = {label : 0 for label in labels}
		for coef, p_value in zip(coef_dict[eps][dm_names].values, p_value_dict[eps][dm_names].values):
			if np.all(p_value<significance_level) and np.all(coef>0):
				dictionary["Type 1"] += 1 # correct discovery
			elif np.any(p_value>significance_level) and np.all(coef>0):
				dictionary["Type 2"] += 1 # correct sign, not significant
			elif np.all(p_value<significance_level) and np.any(coef<0):
				dictionary["Type 3"] += 1 # wrong sign with significance
			else:
				dictionary["Type 4"] += 1 # wrong sing, not significant
		ct[eps] = dictionary
	return ct

def create_ct_sep(coef_dict, p_value_dict):
	ct_sep = {}
	for eps in epsilons:
		dictionary = {label : {coef_name : 0 for coef_name in dm_names} for label in labels}
		for coefs, p_values in zip(coef_dict[eps][dm_names].values, p_value_dict[eps][dm_names].values):
			for jj, (coef, p_value) in enumerate(zip(coefs, p_values)):
				coef_name = dm_names[jj]
				if np.all(p_value<significance_level) and np.all(coef>0):
					dictionary["Type 1"][coef_name] += 1 # correct discovery
				elif np.any(p_value>significance_level) and np.all(coef>0):
					dictionary["Type 2"][coef_name] += 1 # correct sign, not significant
				elif np.all(p_value<significance_level) and np.any(coef<0):
					dictionary["Type 3"][coef_name] += 1 # wrong sign with significance
				else:
					dictionary["Type 4"][coef_name] += 1 # wrong sing, not significant
		ct_sep[eps] = dictionary
	return ct_sep

# stratified
male_ct = create_ct(syn_dpvi_coef_male_dict, syn_dpvi_p_value_male_dict)
male_ct_sep = create_ct_sep(syn_dpvi_coef_male_dict, syn_dpvi_p_value_male_dict)

female_ct = create_ct(syn_dpvi_coef_female_dict, syn_dpvi_p_value_female_dict)
female_ct_sep = create_ct_sep(syn_dpvi_coef_female_dict, syn_dpvi_p_value_female_dict)

# unstratified
no_ad_strat_male_ct = create_ct(syn_no_ad_strat_coef_male_dict, syn_no_ad_strat_p_value_male_dict)
no_ad_strat_male_ct_sep = create_ct_sep(syn_no_ad_strat_coef_male_dict, syn_no_ad_strat_p_value_male_dict)

no_ad_strat_female_ct = create_ct(syn_no_ad_strat_coef_female_dict, syn_no_ad_strat_p_value_female_dict)
no_ad_strat_female_ct_sep = create_ct_sep(syn_no_ad_strat_coef_female_dict, syn_no_ad_strat_p_value_female_dict)
# total unstratified
total_no_strat_male_ct = create_ct(syn_total_no_strat_coef_male_dict, syn_total_no_strat_p_value_male_dict)
total_no_strat_male_ct_sep = create_ct_sep(syn_total_no_strat_coef_male_dict, syn_total_no_strat_p_value_male_dict)

total_no_strat_female_ct = create_ct(syn_total_no_strat_coef_female_dict, syn_total_no_strat_p_value_female_dict)
total_no_strat_female_ct_sep = create_ct_sep(syn_total_no_strat_coef_female_dict, syn_total_no_strat_p_value_female_dict)

# since numerical problems fitting R models with unstrat data, lets renormalize cross tables
for eps in epsilons:
	counts = [total_no_strat_male_ct[eps][label] for label in labels]
	nruns = sum(counts)
	scaled_counts = 100/nruns * np.array(counts)
	total_no_strat_male_ct[eps] = {label : count for label, count in zip(labels, scaled_counts)}
	
for eps in epsilons:
	counts = [total_no_strat_female_ct[eps][label] for label in labels]
	nruns = sum(counts)
	scaled_counts = 100/nruns * np.array(counts)
	total_no_strat_female_ct[eps] = {label : count for label, count in zip(labels, scaled_counts)}


cmap = {"Type 1" : 'g', "Type 2" : 'b',  "Type 4": 'orange', "Type 3" : 'r'}
label_map = {"Type 1" : "Reproduced discovery", "Type 2" : "Correct sign, p>0.05",\
		"Type 3" : "Wrong sign, p<0.05",\
		"Type 4" : "Wrong sign, p>0.05"}
order  = ["Type 1", "Type 2",  "Type 4", "Type 3"]

### PLOT

bar_w = 0.24
# males

fig, axis = plt.subplots(figsize=(width, height))

# plot combined for each epsilon and stratification strategy
for j, eps in enumerate(epsilons):
	males = male_ct[eps]
	male_cumsum = np.cumsum([males[label] for label in order])
	for i, lab in enumerate(order):
		label = label_map[lab]
		if i==3:
			rects = axis.bar(j, males[lab], bar_w, label=label, bottom=male_cumsum[i-1], color=cmap[lab])
		elif i>0:	
			axis.bar(j, males[lab], bar_w, label=label, bottom=male_cumsum[i-1], color=cmap[lab])
		else : 
			axis.bar(j, males[lab], bar_w, label=label, color=cmap[lab])

for j, eps in enumerate(epsilons):
	no_ad_strat_males = no_ad_strat_male_ct[eps]
	male_cumsum = np.cumsum([no_ad_strat_males[label] for label in order])
	for i, lab in enumerate(order):
		label = label_map[lab]
		if i==3:
			rects = axis.bar(j+bar_w+0.01, no_ad_strat_males[lab], bar_w,\
					bottom=male_cumsum[i-1], color=cmap[lab], hatch="o", alpha=0.99)
		elif i>0:	
			axis.bar(j+bar_w+0.01, no_ad_strat_males[lab], bar_w, \
					bottom=male_cumsum[i-1], color=cmap[lab], hatch="o", alpha=0.99)
		else : 
			axis.bar(j+bar_w+0.01, no_ad_strat_males[lab], bar_w, color=cmap[lab], hatch="o", alpha=0.99)

for j, eps in enumerate(epsilons):
	total_no_strat_males = total_no_strat_male_ct[eps]
	male_cumsum = np.cumsum([total_no_strat_males[label] for label in order])
	for i, lab in enumerate(order):
		label = label_map[lab]
		if i==3:
			rects = axis.bar(j+2*(bar_w+0.01), total_no_strat_males[lab], bar_w,\
					bottom=male_cumsum[i-1], color=cmap[lab], hatch="//", alpha=0.99)
		elif i>0:	
			axis.bar(j+2*(bar_w+0.01), total_no_strat_males[lab], bar_w, \
					bottom=male_cumsum[i-1], color=cmap[lab], hatch="//", alpha=0.99)
		else : 
			axis.bar(j+2*(bar_w+0.01), total_no_strat_males[lab], bar_w, color=cmap[lab], hatch="//", alpha=0.99)

#set x-ticks
#axis.set_xticks([j+bar_w/2 for j in range(len(epsilons))])
axis.set_xticks([j+bar_w for j in range(len(epsilons))])
axis.set_xticklabels(np.round(epsilons))
handles, labels = axis.get_legend_handles_labels()
fig.legend(handles[:4], labels[:4], loc='lower center', ncol=2)
# add another legend for bar styles
import matplotlib.patches as mpatches
from matplotlib import pyplot

patch_w = 1 
patch_h = 1
patch_total_no_strat = mpatches.Rectangle([0,0], patch_w, patch_h, facecolor='white', edgecolor='black', hatch="////")
patch_no_ad_strat = mpatches.Rectangle([0,0], patch_w, patch_h, facecolor='white', edgecolor='black', hatch="ooo")
patch_strat = mpatches.Rectangle([0,0], patch_w, patch_h, facecolor='white', edgecolor='black')
legend2 = pyplot.legend((patch_strat, patch_no_ad_strat, patch_total_no_strat), ("Stratified", "No alive/dead strat.", "Unstratified"), loc=(-.16,-.37), ncol=3)

axis.set_ylabel('Percentage of repeats')
axis.set_xlabel(r"$\epsilon$")
axis.set_title("Male")
fig.subplots_adjust(bottom=0.33)
fig.savefig(plot_path+'male_both_strat_bars.pdf', format='pdf', bbox_inches='tight')
plt.close()

# females

fig, axis = plt.subplots(figsize=(width, height))

# plot combined for each epsilon and stratification strategy
for j, eps in enumerate(epsilons):
	females = female_ct[eps]
	female_cumsum = np.cumsum([females[label] for label in order])
	for i, lab in enumerate(order):
		label = label_map[lab]
		if i==3:
			rects = axis.bar(j, females[lab], bar_w, label=label, bottom=female_cumsum[i-1], color=cmap[lab])
		elif i>0:	
			axis.bar(j, females[lab], bar_w, label=label, bottom=female_cumsum[i-1], color=cmap[lab])
		else : 
			axis.bar(j, females[lab], bar_w, label=label, color=cmap[lab])

for j, eps in enumerate(epsilons):
	no_ad_strat_females = no_ad_strat_female_ct[eps]
	female_cumsum = np.cumsum([no_ad_strat_females[label] for label in order])
	for i, lab in enumerate(order):
		label = label_map[lab]
		if i==3:
			rects = axis.bar(j+bar_w+0.01, no_ad_strat_females[lab], bar_w,\
					bottom=female_cumsum[i-1], color=cmap[lab], hatch="o", alpha=0.99)
		elif i>0:	
			axis.bar(j+bar_w+0.01, no_ad_strat_females[lab], bar_w, \
					bottom=female_cumsum[i-1], color=cmap[lab], hatch="o", alpha=0.99)
		else : 
			axis.bar(j+bar_w+0.01, no_ad_strat_females[lab], bar_w, color=cmap[lab], hatch="o", alpha=0.99)

for j, eps in enumerate(epsilons):
	total_no_strat_females = total_no_strat_female_ct[eps]
	female_cumsum = np.cumsum([total_no_strat_females[label] for label in order])
	for i, lab in enumerate(order):
		label = label_map[lab]
		if i==3:
			rects = axis.bar(j+2*(bar_w+0.01), total_no_strat_females[lab], bar_w,\
					bottom=female_cumsum[i-1], color=cmap[lab], hatch="//", alpha=0.99)
		elif i>0:	
			axis.bar(j+2*(bar_w+0.01), total_no_strat_females[lab], bar_w, \
					bottom=female_cumsum[i-1], color=cmap[lab], hatch="//", alpha=0.99)
		else : 
			axis.bar(j+2*(bar_w+0.01), total_no_strat_females[lab], bar_w, color=cmap[lab], hatch="//", alpha=0.99)

#set x-ticks
axis.set_xticks([j+bar_w for j in range(len(epsilons))])
axis.set_xticklabels(np.round(epsilons))
handles, labels = axis.get_legend_handles_labels()
fig.legend(handles[:4], labels[:4], loc='lower center', ncol=2)
# add another legend for bar styles
import matplotlib.patches as mpatches
from matplotlib import pyplot

patch_w = 1 
patch_h = 1
patch_total_no_strat = mpatches.Rectangle([0,0], patch_w, patch_h, facecolor='white', edgecolor='black', hatch="////")
patch_no_ad_strat = mpatches.Rectangle([0,0], patch_w, patch_h, facecolor='white', edgecolor='black', hatch="ooo")
patch_strat = mpatches.Rectangle([0,0], patch_w, patch_h, facecolor='white', edgecolor='black')
legend2 = pyplot.legend((patch_strat, patch_no_ad_strat, patch_total_no_strat), ("Stratified", "No alive/dead strat.", "Unstratified"), loc=(-.16,-.37), ncol=3)

axis.set_ylabel('Percentage of repeats')
axis.set_xlabel(r"$\epsilon$")
axis.set_title("Female")
fig.subplots_adjust(bottom=0.33)
fig.savefig(plot_path+'female_both_strat_bars.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()
