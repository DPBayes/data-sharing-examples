import torch, sys, math
from torch.distributions import Categorical, Bernoulli, Dirichlet, Beta

from transform import softmax, log_det_jacobian_softmax, logsoftmax, log_det_jacobian_logsoftmax, log_det_jacobian_sigmoid
from utils import logsumexp


def log_likelihood(X_apps, theta_unconstrained, pi_unconstrained):
	"""
	X_apps.shape = [n, 1, d]
	theta_unconstrained \in \mathbb{R}^{k, d}, where d is the number of app cats
	"""
	log_pi = logsoftmax(pi_unconstrained, dim=-1)
	logp_theta = torch.sum(Bernoulli(logits=theta_unconstrained).log_prob(X_apps), dim=-1)
	logps = logsumexp(log_pi+logp_theta, dim=-1)
	return logps.sum()


def log_prior(theta_unconstrained, pi_unconstrained):
	theta = torch.sigmoid(theta_unconstrained)
	pi = softmax(pi_unconstrained, dim=-1)
	""" Both tau and pi are transformed, so we need to add correction terms
		to log densities. For tau this will be theta_unconstrained (log |d/dy exp(y)| = y)
		and for pi this will be log_det_jacobian_softmax (for reasons too 
		complicated to be explained here).
	"""
	theta_logp = torch.sum(log_det_jacobian_sigmoid(theta_unconstrained))
	pi_logp = torch.sum(Dirichlet(torch.ones_like(pi)).log_prob(pi)+log_det_jacobian_softmax(pi, dim=-1))
	return pi_logp + theta_logp
