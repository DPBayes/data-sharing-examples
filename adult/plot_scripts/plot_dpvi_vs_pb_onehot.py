import numpy as np
import pandas as pd
import pickle

### Load test data and encoders
target_variable = "Target"

## Load results
dpvi_dict = pickle.load(open('./plot_pickles/dpvi_classifiers_2019-11-05_onehot_{}.p'.format(target_variable),'rb'))
pb_dict = pickle.load(open('./plot_pickles/pb_classifiers_2019-11-05_onehot_{}.p'.format(target_variable),'rb'))

dpvi_res = dpvi_dict['accs']
pb_res = pb_dict['accs']

dpvi_epsilons = np.sort(list(dpvi_res.keys()))
pb_epsilons = np.sort(list(pb_res.keys()))

## Take means and SEMs
n_runs_pb = 10
mean_pb = np.mean([pb_res[epsilon] for epsilon in pb_epsilons], axis=1)
sem_pb = np.std([pb_res[epsilon] for epsilon in pb_epsilons], axis=1)/np.sqrt(n_runs_pb)

n_runs_dpvi = 10
mean_dpvi = np.mean([dpvi_res[epsilon] for epsilon in dpvi_epsilons], axis=1)
sem_dpvi = np.std([dpvi_res[epsilon] for epsilon in dpvi_epsilons], axis=1)/np.sqrt(n_runs_dpvi)

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
width = 8.7/2.54
height = width*(3/4)
fig, axis = plt.subplots(figsize=(width, height))
axis.errorbar(dpvi_epsilons, mean_pb, yerr=sem_pb, label='Data sharing, Bayes network', color='b', linewidth=1)
axis.errorbar(dpvi_epsilons, mean_dpvi, yerr=sem_pb, label='Data sharing, Mixture model', color='r', linewidth=1)
axis.set_xlabel(r'$\epsilon$')
axis.set_ylabel('Accuracy')

# Plot anticipated
ant_res = pickle.load(open('./plot_pickles/anticipated_res_onehot.p', 'rb'))
ant_means = np.array(list(ant_res['means'].values())).T
ant_stds = np.array(list(ant_res['stds'].values())).T
green_cmap = plt.get_cmap('Greens')
for i, T in enumerate([2, 5, 10, 20]):
	axis.errorbar(dpvi_epsilons, ant_means[i], yerr=ant_stds[i]/np.sqrt(n_runs_dpvi),\
			label='Tailored mechanism T={}'.format(T), linestyle='--', color=green_cmap((T/20)**0.5), linewidth=1)
fig.legend(loc=(0.42,0.2))
plt.savefig('adult_vs_tailored_onehot.pdf', format='pdf', bbox_inches='tight')
plt.close()
