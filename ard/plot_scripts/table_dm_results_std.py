import pandas as pd
import pickle
import numpy as np

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

## Load true fits
real_fit_male = pd.read_csv(script_dir+'/plot_pickles/real_male_fit.csv')
real_male_df = pd.DataFrame(real_fit_male.iloc[:,1:].values, index=real_fit_male.iloc[:,0].values, columns=real_fit_male.columns[1:])
real_fit_female = pd.read_csv(script_dir+'/plot_pickles/real_female_fit.csv')
real_female_df = pd.DataFrame(real_fit_female.iloc[:,1:].values, index=real_fit_female.iloc[:,0].values, columns=real_fit_female.columns[1:])

# Load raritys
male_rarity, female_rarity = pickle.load(open(script_dir+'/plot_pickles/raritys.p', 'rb'))

## Load DPVI fits
epsilons = [0.74, 1.99, 3.92]
epsilons = np.array(epsilons)
n_runs = 100

# Load DP results
syn_dpvi_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/female_coef_dict.p', 'rb'))
syn_dpvi_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/male_coef_dict.p', 'rb'))

dm_names = [name for name in real_male_df.index if 'DM' in name]
female_dm_means = {eps : syn_dpvi_coef_female_dict[eps][dm_names].mean(0).round(3).astype('str') for eps in epsilons}
female_dm_sems = {eps : (syn_dpvi_coef_female_dict[eps][dm_names].std(0)).round(3).astype('str') for eps in epsilons}

male_dm_means = {eps : syn_dpvi_coef_male_dict[eps][dm_names].mean(0).round(3).astype('str') for eps in epsilons}
male_dm_sems = {eps : (syn_dpvi_coef_male_dict[eps][dm_names].std(0)).round(3).astype('str') for eps in epsilons}

# Load non-DP results, k = 10
syn_nondp_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/female_coef_dict_NONDP.p', 'rb'))
syn_nondp_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/male_coef_dict_NONDP.p', 'rb'))

female_dm_means_nondp = syn_nondp_coef_female_dict["NONDP"][dm_names].mean(0).round(3).astype('str')
female_dm_sems_nondp = syn_nondp_coef_female_dict["NONDP"][dm_names].std(0).round(3).astype('str') 

male_dm_means_nondp = syn_nondp_coef_male_dict["NONDP"][dm_names].mean(0).round(3).astype('str')
male_dm_sems_nondp = syn_nondp_coef_male_dict["NONDP"][dm_names].std(0).round(3).astype('str') 

# Load non-DP results, k = 40
syn_nondp_k40_coef_female_dict = pickle.load(open(script_dir+'/plot_pickles/female_coef_dict_NONDP_k40.p', 'rb'))
syn_nondp_k40_coef_male_dict = pickle.load(open(script_dir+'/plot_pickles/male_coef_dict_NONDP_k40.p', 'rb'))

female_dm_means_nondp_k40 = syn_nondp_k40_coef_female_dict["NONDP"][dm_names].mean(0).round(3).astype('str')
female_dm_sems_nondp_k40 = syn_nondp_k40_coef_female_dict["NONDP"][dm_names].std(0).round(3).astype('str') 

male_dm_means_nondp_k40 = syn_nondp_k40_coef_male_dict["NONDP"][dm_names].mean(0).round(3).astype('str')
male_dm_sems_nondp_k40 = syn_nondp_k40_coef_male_dict["NONDP"][dm_names].std(0).round(3).astype('str') 

### Create tables
## Female
female_orig = real_female_df.loc[dm_names].iloc[:, :2].round(3)
female_orig_str = ['${} \pm {}$'.format(coef, err) for coef, err in female_orig.astype('str').values]
female_syn = {eps : ['${} \pm {}$'.format(coef, err) for coef, err in zip(female_dm_means[eps], female_dm_sems[eps])] for eps in epsilons}
female_syn_nondp_str = ['${} \pm {}$'.format(coef, err) for coef, err in zip(female_dm_means_nondp, female_dm_sems_nondp)]
female_res = [[dm.replace('DM.type','') for dm in dm_names]] + [[female_rarity[dm] for dm in dm_names]]+[female_orig_str]+[female_syn[eps] for eps in epsilons] + [female_syn_nondp_str]
#columns = ['Coefficient', 'Number of cases', 'Original coef. $\pm$ Std. Error']+['($\epsilon={}$) coef. mean $\pm$ SD'.format(eps) for eps in np.round(epsilons)]+["($\epsilon=\infty$) coef. mean $\pm$ SD"]
columns = ['Coefficient', 'Number of cases', 'Original coef. $\pm$ Std. Error']+[r'$\epsilon={}$'.format(eps) for eps in np.round(epsilons)]+["$\epsilon=\infty$"]
female_table = pd.DataFrame(np.array(female_res).T, columns=columns)
female_table.to_latex(plot_path+'female_table_stds.tex', escape=False)

## Male
male_orig = real_male_df.loc[dm_names].iloc[:, :2].round(3)
male_orig_str = ['${} \pm {}$'.format(coef, err) for coef, err in male_orig.astype('str').values]
male_syn = {eps : ['${} \pm {}$'.format(coef, err) for coef, err in zip(male_dm_means[eps], male_dm_sems[eps])] for eps in epsilons}
male_syn_nondp_str = ['${} \pm {}$'.format(coef, err) for coef, err in zip(male_dm_means_nondp, male_dm_sems_nondp)]
male_res = [[dm.replace('DM.type','') for dm in dm_names]] + [[male_rarity[dm] for dm in dm_names]]+[male_orig_str]+[male_syn[eps] for eps in epsilons] + [male_syn_nondp_str]
#columns = ['Coefficient', 'Number of cases', 'Original coef. $\pm$ Std. Error']+['($\epsilon={}$) coef. mean $\pm$ SD'.format(eps) for eps in np.round(epsilons)]+["($\epsilon=\infty$) coef. mean $\pm$ SD"]
columns = ['Coefficient', 'Number of cases', 'Original coef. $\pm$ Std. Error']+[r'$\epsilon={}$'.format(eps) for eps in np.round(epsilons)]+[r"$\epsilon=\infty$"]
male_table = pd.DataFrame(np.array(male_res).T, columns=columns)
male_table.to_latex(plot_path+'male_table_stds.tex', escape=False)

### Create tables, k=40
epsilons = [1.99, 3.92]
## Female
female_orig = real_female_df.loc[dm_names].iloc[:, :2].round(3)
female_orig_str = ['${} \pm {}$'.format(coef, err) for coef, err in female_orig.astype('str').values]
female_syn = {eps : ['${} \pm {}$'.format(coef, err) for coef, err in zip(female_dm_means[eps], female_dm_sems[eps])] for eps in epsilons}
female_syn_nondp_str = ['${} \pm {}$'.format(coef, err) for coef, err in zip(female_dm_means_nondp, female_dm_sems_nondp)]
female_syn_nondp_k40_str = ['${} \pm {}$'.format(coef, err) for coef, err in zip(female_dm_means_nondp_k40, female_dm_sems_nondp_k40)]
female_res = [[dm.replace('DM.type','') for dm in dm_names]] + [[female_rarity[dm] for dm in dm_names]]+[female_orig_str]+[female_syn[eps] for eps in epsilons] + [female_syn_nondp_str] + [female_syn_nondp_k40_str]
columns = ['Coefficient', 'Number of cases', 'Original coef.']+['$\epsilon={}$'.format(eps) for eps in np.round(epsilons)]+["$\epsilon=\infty$"] + ["$\epsilon=\infty, k=40$"]
female_table = pd.DataFrame(np.array(female_res).T, columns=columns)
female_table.to_latex(plot_path+'female_table_k40.tex', escape=False)

## Male
male_orig = real_male_df.loc[dm_names].iloc[:, :2].round(3)
male_orig_str = ['${} \pm {}$'.format(coef, err) for coef, err in male_orig.astype('str').values]
male_syn = {eps : ['${} \pm {}$'.format(coef, err) for coef, err in zip(male_dm_means[eps], male_dm_sems[eps])] for eps in epsilons}
male_syn_nondp_str = ['${} \pm {}$'.format(coef, err) for coef, err in zip(male_dm_means_nondp, male_dm_sems_nondp)]
male_syn_nondp_k40_str = ['${} \pm {}$'.format(coef, err) for coef, err in zip(male_dm_means_nondp_k40, male_dm_sems_nondp_k40)]
male_res = [[dm.replace('DM.type','') for dm in dm_names]] + [[male_rarity[dm] for dm in dm_names]]+[male_orig_str]+[male_syn[eps] for eps in epsilons] + [male_syn_nondp_str] + [male_syn_nondp_k40_str]
columns = ['Coefficient', 'Number of cases', 'Original coef.']+['$\epsilon={}$'.format(eps) for eps in np.round(epsilons)]+["$\epsilon=\infty$"] + ["$\epsilon=\infty, k=40$"]
male_table = pd.DataFrame(np.array(male_res).T, columns=columns)
male_table.to_latex(plot_path+'male_table_k40.tex', escape=False)
