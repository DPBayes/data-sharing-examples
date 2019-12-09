import pickle, torch
import numpy as np
import pandas as pd

target_epsilons = [1.1, 2.0, 4.0, 8.0, 14.0]
anticipated_Ts = [2, 5, 10, 20]
models_dict = {}
for eps in target_epsilons:
	models_dict[eps] = pickle.load(open('./res/models_2019-11-05_{}.p'.format(eps), 'rb'))


X_test = pd.read_csv('./onehotted_data/encoded_X_test.csv', sep=';')
y_test = pd.read_csv('./onehotted_data/encoded_y_test.csv', sep=';', header=None).values.squeeze()

feature_names = list(X_test.columns)
X_test['Intercept'] = np.ones(len(X_test))
X_test = X_test[['Intercept'] + feature_names]

accs_dict={}
for eps in target_epsilons:
	models = models_dict[eps]
	accs = np.zeros(40)
	for i, model in enumerate(models):
		w_map = model.reparam.bias.data.numpy()
		S_N = model.reparam.weight.exp().data.numpy()**2
		mu_a = X_test.dot(w_map) 
		sigma_a2 = (X_test**2).dot(S_N)
		kappa = (1+np.pi*sigma_a2/8)**-0.5
		sigmoid = lambda x : (1+np.exp(-x))**-1
		y_pred = 1*(sigmoid(kappa*mu_a)>0.5)
		accs[i] = np.mean(y_pred==y_test)
	accs = np.array(np.split(accs, 10))
	## accs \in R^{10 x 4}, column corresponds to a anticipated runs
	accs_dict[eps]=accs

mean_accs_dict = {eps : accs_dict[eps].mean(0) for eps in target_epsilons}
std_accs_dict = {eps : accs_dict[eps].std(0) for eps in target_epsilons}

pickle.dump({'means': mean_accs_dict, 'stds': std_accs_dict},\
		open('../plot_scripts/plot_pickles/anticipated_res_onehot.p', 'wb'))
