import numpy as np
import pandas as pd
import torch
import numpy.random as npr
from transform import smoid

def fast_sample(model, variable_types, n_samples):
	draw_mu = model.forward(torch.zeros(model.input_dim))
	draw_sigma = model.forward(torch.ones(model.input_dim))    
	draw_sigma = {key : elem-draw_mu[key] for key, elem in draw_sigma.items()}

	sample_data = pd.DataFrame()
	variable_types_copy = variable_types.copy()
	if 'pi_unconstrained' in variable_types.keys():
		pi_mu = draw_mu['pi_unconstrained'].cpu().data.numpy()[0]
		pi_sigma = draw_sigma['pi_unconstrained'].cpu().data.numpy()[0]

		k = pi_mu.shape[0]+1
		ks = np.argmax(np.hstack([npr.gumbel(size=[n_samples, k-1])+pi_mu +\
						npr.randn(n_samples, pi_mu.shape[0])*pi_sigma,\
						npr.gumbel(size=[n_samples, 1])]), axis=1)

		draw_mu.pop('pi_unconstrained')
		draw_sigma.pop('pi_unconstrained')
		variable_types_copy.pop('pi_unconstrained')

	else : 
		k = draw_mu['Target'].shape[1]
		ks = npr.randint(k, size=n_samples)

	cont_features = [key for key, value in variable_types_copy.items() if value=='Beta' ]
	param_mu = {key : value.cpu().data.numpy()[0, ks] if key not in cont_features\
					else value.cpu().data.numpy()[0,:,ks]\
					for key, value in draw_mu.items()}

	param_sigma = {key : value.cpu().data.numpy()[0, ks] if key not in cont_features\
					else value.cpu().data.numpy()[0,:,ks]\
					for key, value in draw_sigma.items()}
	for key, dist in variable_types_copy.items():
		if dist == 'Categorical':
			d = param_mu[key].shape[1]
			sample_data[key] = np.argmax(npr.gumbel(size=[n_samples, d])+param_mu[key] +\
									npr.randn(n_samples, d)*param_sigma[key], axis=1)
		
		elif dist == 'Bernoulli':
			sample_data[key] = 1*(smoid(param_mu[key]+npr.randn(n_samples)*param_sigma[key])>npr.rand(n_samples))

		elif dist == 'Beta' :
			a, b = np.exp((param_mu[key]+npr.randn(n_samples, 2)*param_sigma[key]).T)
			sample_data[key] = npr.beta(a,b)

	return sample_data
