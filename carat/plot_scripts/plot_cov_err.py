import torch, sys, math, pickle, datetime
import numpy as np
import numpy.random as npr
from collections import OrderedDict


use_cuda = torch.cuda.is_available()
npr.seed(1234)
if use_cuda : 
	torch.set_default_tensor_type('torch.cuda.DoubleTensor')
	torch.cuda.manual_seed(1234)
else : 
	torch.set_default_tensor_type('torch.DoubleTensor')
	torch.manual_seed(1234)

### Load carat-data
fname_dpvi = '2019-03-26'
import pandas as pd
ds = [8,16,32,64,96]
dpvi_err = []
dpvi_times = []
learn = 0
if learn:
	sys.path.append('../../dpvi/')
	from sampler import fast_sample
	for d in ds:
		app_data = pd.read_csv('../../data/subsets/carat_apps_sub{}.dat'.format(d), sep=' ', header=None)\
								.astype('float').values
		N = len(app_data)
		models = pickle.load(open('../../dpvi/models_{0}/models_{0}_{1}.p'.format(fname_dpvi, d), 'rb'))
		for model in models:
			syn_app_data = fast_sample(model, N)

			syn_cov = np.cov(syn_app_data.T)
			orig_cov = np.cov(app_data.T)
			dpvi_err.append(np.linalg.norm(orig_cov-syn_cov))
		log = open('logs_{0}/out_file_{0}_{1}.txt'.format(fname_dpvi, d), 'r')
		wall_time, cpu_time = log.readlines()[-2:]
		log.close()
		wall_time = float(wall_time.strip('Wall time').strip('\n'))
		cpu_time = float(cpu_time.strip('CPU time').strip('\n'))
		dpvi_times.append((wall_time, cpu_time))
	pd.DataFrame(dpvi_err).to_csv('../plot_data/dpvi_cov_err_(8,16,32,64,96)_{}.csv'\
							.format(fname_dpvi), sep=';', header=None, index=False)
	pd.DataFrame(dpvi_times).to_csv('../plot_data/dpvi_times_(8,16,32,64,96)_{}.csv'\
							.format(fname_dpvi), sep=';', header=None, index=False)
else:
	dpvi_err = pd.read_csv('../plot_data/dpvi_cov_err_(8,16,32,64,96)_{}.csv'\
							.format(fname_dpvi), sep=';', header=None)
	dpvi_times = pd.read_csv('../plot_data/dpvi_times_(8,16,32,64,96)_{}.csv'\
							.format(fname_dpvi), sep=';', header=None)
dpvi_err=np.array(dpvi_err)
dpvi_err = np.split(dpvi_err, len(ds))
pb_err = pd.read_csv('../plot_data/pb_cov_err_(8,16,32,64,96).csv', sep=';', header=None).values[:,0]
pb_err = np.split(pb_err, len(ds))


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
width = 8.7/2.54
height = width*(3/4)
fig, ax1 = plt.subplots(figsize=(width, height))

ax1.errorbar(ds, np.mean(pb_err, 1), yerr=np.std(pb_err, 1)/np.sqrt(10),\
				label='Bayes network', color='b')
ax1.errorbar(ds, np.mean(dpvi_err, 1), yerr=np.std(dpvi_err, 1)/np.sqrt(10),\
				label='Mixture model', color='r')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('Frobenius error')
legend1 = ax1.legend(loc='best')

### Plot times
pb_times = pd.read_csv('../plot_data/pb_slurm_history.txt', comment="#", sep = ';')
pb_times_cols = [elem.strip(' ') for elem in list(pb_times.columns)]
pb_times = pd.read_csv('../plot_data/pb_slurm_history.txt', comment="#", sep = ';',\
		skiprows=[0],names=pb_times_cols)
pb_wall = pb_times.WallTime
pb_secs = []
time_conversion = [1, 60, 60*60, 24*60*60]
from itertools import chain
for elem in pb_wall:
	elems = elem.split(':')
	elems = [elem.split('-') for elem in elems]
	elems = np.array(list(chain(*elems)), dtype=np.int32)[::-1]
	pb_secs.append(np.sum([time_conversion[i]*elem for i, elem in enumerate(elems)]))
pb_secs = np.array(pb_secs)

ax2 = ax1.twinx()
ax2.plot(ds, dpvi_times[0]/10, scalex='log', color='r', linestyle='--')
ax2.plot(ds, pb_secs/10, scalex='log', color='b', linestyle='--')
ax2.set_ylabel('Wall clock time (s)')
from matplotlib.lines import Line2D
errorline = Line2D([0], [0], color='black', linestyle='-', label='Error')
timeline = Line2D([0], [0], color='black', linestyle='--', label='Runtime')
legend1_handles, legend1_labels = ax1.get_legend_handles_labels()
legend2_handles = []
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import LineCollection
legend2_handles.append(ErrorbarContainer((errorline,(), (LineCollection(np.zeros((10,10,2)), color='black'),)), has_yerr=True, has_xerr=False))
legend2_handles.append(timeline)
ax2.legend(legend2_handles, ['Error', 'Runtime'], bbox_to_anchor=(.0,0.8), loc='upper left')
plt.savefig(plot_path+'pb_vs_dpvi_frobenius_8-96_{}.pdf'.format(fname_dpvi), format='pdf', bbox_inches='tight')
plt.close()
