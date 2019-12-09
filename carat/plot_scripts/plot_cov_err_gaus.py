#### Gaussian perturbation of covariance matrix
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from collections import OrderedDict

plot_path = './'

npr.seed(1234)
learn = 0
### 
fname_gaus = '2019-04-26'
import pandas as pd
ds = [8,16,32,64,96]
gaus_err = []
delta = 1e-5
target_eps = 1.0
c = np.sqrt(2*np.log(1.25/delta))
gaus_sigma = c/target_eps
Ts = [2, 5, 10, 20]
n_runs = 10
if learn:
	for T in Ts:
		for d in ds:
			app_data = pd.read_csv('../../data/subsets/carat_apps_sub{}.dat'.format(d), sep=' ', header=None)\
									.astype('float').values
			N = len(app_data)
			orig_cov = np.cov(app_data.T)
			Delta_f = np.sqrt(d*(d-1)/2)/N # covariance sensitivity
			for i in range(n_runs):
				gaus_cov = orig_cov + T*gaus_sigma*Delta_f*npr.randn(*orig_cov.shape)
				gaus_cov = np.triu(gaus_cov, 1).T+np.triu(gaus_cov, 1)+np.eye(d)*np.diag(gaus_cov)
				gaus_err.append(np.linalg.norm(orig_cov-gaus_cov))
	pd.DataFrame(gaus_err).to_csv('gaus_cov_err_(8,16,32,64,96)_{}.csv'\
							.format(fname_gaus), sep=';', header=None, index=False)
gaus_err = pd.read_csv('../plot_data/gaus_cov_err_(8,16,32,64,96)_{}.csv'\
						.format(fname_gaus), sep=';', header=None)
gaus_err = np.array(gaus_err)
gaus_err = np.array(np.split(gaus_err, len(Ts))).squeeze(-1)

# Load dpvi errors
fname_dpvi = '2019-03-26'
dpvi_err = pd.read_csv('../plot_data/dpvi_cov_err_(8,16,32,64,96)_{}.csv'.format(fname_dpvi), sep=';', header=None)
dpvi_times = pd.read_csv('../plot_data/dpvi_times_(8,16,32,64,96)_{}.csv'.format(fname_dpvi), sep=';', header=None)
dpvi_err=np.array(dpvi_err)
dpvi_err = np.array(np.split(dpvi_err, len(ds))).squeeze(-1)

plt.rcParams.update({'font.size': 7})
width = 8.7/2.54
height = width*(3/4)
fig, ax = plt.subplots(figsize=(width, height))
green_cmap = plt.get_cmap('Greens')
for i, T in enumerate(Ts):
	gaus_error = np.split(gaus_err[i], len(ds))
	ax.errorbar(ds, np.mean(gaus_error, axis=1), yerr=np.std(gaus_error, axis=1)/np.sqrt(n_runs), \
			label = "Tailored mechanism, T={}".format(T), linestyle='--', linewidth=1,\
			color = green_cmap((T/20)**0.5))

ax.errorbar(ds, np.mean(dpvi_err, axis=1), yerr=np.std(dpvi_err, axis=1), label='Data sharing, Mixture model',\
		linewidth=1, color='r')
ax.set_xlabel('Dimension')
ax.set_ylabel('Frobenius error')
fig.legend(loc=(0.15, .62))
plt.savefig(plot_path+'gaus_vs_dpvi_frobenius_8-96_{}.pdf'.format(fname_dpvi), format='pdf', bbox_inches='tight')
plt.close()
