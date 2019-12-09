import numpy as np
import pandas as pd
import torch
import numpy.random as npr
from transform import smoid

def fast_sample(model, n_samples, ml_estimation=False):
	draw_mu = model.forward(torch.zeros(model.input_dim))
	draw_sigma = model.forward(torch.ones(model.input_dim))    
	if ml_estimation:
		draw_sigma = {key : 0.0*(elem-draw_mu[key]) for key, elem in draw_sigma.items()}
	else:
		draw_sigma = {key : elem-draw_mu[key] for key, elem in draw_sigma.items()}

	pi_mu = draw_mu['pi_unconstrained'].cpu().data.numpy()[0]
	pi_sigma = draw_sigma['pi_unconstrained'].cpu().data.numpy()[0]

	k = pi_mu.shape[0]+1
	ks = np.argmax(np.hstack([npr.gumbel(size=[n_samples, k-1])+pi_mu +\
						npr.randn(n_samples, pi_mu.shape[0])*pi_sigma,\
						npr.gumbel(size=[n_samples, 1])]), axis=1)
	draw_mu.pop('pi_unconstrained')
	draw_sigma.pop('pi_unconstrained')

	param_mu = {key : value.cpu().data.numpy()[0, ks] \
					for key, value in draw_mu.items()}

	param_sigma = {key : value.cpu().data.numpy()[0, ks] \
					for key, value in draw_sigma.items()}
	thetas_mu =  param_mu['theta_unconstrained']
	thetas_sigma =  param_sigma['theta_unconstrained']
	syn_app_data = 1*(smoid(npr.randn(*thetas_sigma.shape)*thetas_sigma+thetas_mu)\
								>npr.rand(*thetas_sigma.shape))
	return syn_app_data
