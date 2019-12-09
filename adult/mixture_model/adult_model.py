import torch
from torch.distributions import Categorical, Bernoulli, Dirichlet, Beta, Gamma, Normal 
from transform import softmax, logsoftmax, log_det_jacobian_sigmoid, log_det_jacobian_softmax,\
						 log_det_jacobian_logsoftmax
from torch.distributions import TransformedDistribution, SigmoidTransform
base_dist = Beta(1,1)
transforms = [SigmoidTransform().inv]
TransBeta = TransformedDistribution(base_dist, transforms)

##################################################
### Log-likelihood ###
def log_likelihood(X, Z, variable_types, use_cuda=False):
	"""
	Z contains all the latent variables
	"""
	if 'pi_unconstrained' in Z.keys() : k = Z['pi_unconstrained'].shape[1]+1
	else: k=1
	logps = torch.zeros([len(X), k])
	if k>1 : logps += logsoftmax(Z['pi_unconstrained'], dim=-1)
	for i, (key, z) in enumerate(Z.items()):
		if key not in ['pi_unconstrained']:
			data = torch.Tensor(X[key].values).unsqueeze(-1)
			if use_cuda : data = data.cuda()
			dist = variable_types[key]
			if dist == 'Categorical':
				alpha = softmax(z, dim=-1, additional=-50.)
				logps += Categorical(probs = alpha).log_prob(data)
			elif dist == 'Bernoulli':
				theta = z
				logps += Bernoulli(logits = theta).log_prob(data)
			elif dist == 'Beta':
				alpha, beta = torch.exp(z).transpose(0,1)
				logps += Beta(alpha, beta).log_prob(data)
	logp = torch.logsumexp(logps, dim=-1)
	return logp.sum()


##################################################
### Log-prior ###
def log_prior(Z, variable_types):
	if 'pi_unconstrained' in variable_types.keys():
		pi = softmax(Z['pi_unconstrained'][0], dim=-1)
		logp = Dirichlet(torch.ones_like(pi)).log_prob(pi)+log_det_jacobian_softmax(pi, dim=-1) 
	else:
		logp = 0
	for i, (key, z) in enumerate(Z.items()):
		z = z[0]
		if key != 'pi_unconstrained':
			if variable_types[key] == 'Categorical':
				alpha = softmax(z, dim=-1, additional=-50.)
				logp += torch.sum(Dirichlet(torch.ones_like(alpha)).log_prob(alpha) \
						+ log_det_jacobian_softmax(alpha, dim=-1), dim=-1)
			#elif variable_types[key] == 'Bernoulli':
			#	theta = torch.sigmoid(z)
			#	logp += torch.sum(Beta(torch.ones_like(theta), torch.ones_like(theta)).log_prob(theta)\
			#			+ log_det_jacobian_sigmoid(theta), dim=-1)
			elif variable_types[key] == 'Bernoulli':
				logp += TransBeta.log_prob(z).sum()
			elif variable_types[key] == 'Beta':
				alpha, beta = torch.exp(z)
				logp += torch.sum(Gamma(1.0, 1.0).log_prob(alpha) + torch.log(alpha), dim=-1)
				logp += torch.sum(Gamma(1.0, 1.0).log_prob(beta) + torch.log(beta), dim=-1)
	return torch.mean(logp)



