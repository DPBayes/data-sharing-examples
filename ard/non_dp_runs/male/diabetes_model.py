import torch
from torch.distributions import Categorical, Bernoulli, Dirichlet, Beta, Gamma 
from transform import softmax, logsoftmax, log_det_jacobian_sigmoid, log_det_jacobian_softmax,\
						 log_det_jacobian_logsoftmax

##################################################
### Log-likelihood ###

def log_likelihood(X, Z, variable_types):
	"""
	Log-likelihood of a mixture model
	X : A pandas dataframe containing the minibatch. 
	Z : A dictionary containing the draws from variational posterior for each parameter.
	variable_types : a dictionary that contains distribution name assigned to each parameter.
	"""
	k = Z['pi_unconstrained'].shape[1]+1 # the number of mixture components
	## We gather the log probabilities of each indiv in batch for each mixture component into
	## a matrix of size (B x k), where B is the batch size.
	logps = torch.zeros([len(X), k])
	## First insert the mixture weight contribution to the array
	logps += logsoftmax(Z['pi_unconstrained'], dim=-1)
	## Next loop over the features and sum the contributions to logps
	for i, (key, z) in enumerate(Z.items()):
		if key not in ['pi_unconstrained']:
			data = torch.Tensor(X[key].values).unsqueeze(-1)
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
	## Compute logsumexp over the mixture components and return the sum over data elements.
	logp = torch.logsumexp(logps, dim=-1)
	return logp.sum()


##################################################
### Log-prior ###
def log_prior(Z, variable_types):
	"""
	Z : A dictionary containing the draws from variational posterior for each parameter.
	variable_types : a dictionary that contains distribution name assigned to each parameter.
	"""
	## We proceed similarly as in log-likelihood computation, however since the elements of 
	## Z are in expanded form and the prior is not data dependent we compute the contribution
	## of only the first element of element of Z.
	pi = softmax(Z['pi_unconstrained'][0], dim=-1)
	logp = Dirichlet(torch.ones_like(pi)).log_prob(pi)+log_det_jacobian_softmax(pi, dim=-1) 
	for i, (key, z) in enumerate(Z.items()):
		if key != 'pi_unconstrained':
			z = z[0]
			if variable_types[key] == 'Categorical':
				alpha = softmax(z, dim=-1, additional=-50.)
				logp += torch.sum(Dirichlet(torch.ones_like(alpha)).log_prob(alpha) \
						+ log_det_jacobian_softmax(alpha, dim=-1), dim=-1)
			elif variable_types[key] == 'Bernoulli':
				theta = torch.sigmoid(z)
				logp += torch.sum(Beta(torch.ones_like(theta), torch.ones_like(theta)).log_prob(theta)\
						+ log_det_jacobian_sigmoid(theta), dim=-1)
			elif variable_types[key] == 'Beta':
				alpha, beta = torch.exp(z)
				logp += torch.sum(Gamma(1.0, 1.0).log_prob(alpha) + torch.log(alpha), dim=-1)
				logp += torch.sum(Gamma(1.0, 1.0).log_prob(beta) + torch.log(beta), dim=-1)
	return logp
