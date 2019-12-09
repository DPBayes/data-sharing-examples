import torch
from torch.distributions import Bernoulli, Normal

##################################################
### Log-likelihood ###
def log_likelihood(X, y, w, use_cuda=False):
	data = torch.as_tensor(X.values)
	target = torch.as_tensor(y.values)
	logits = (data*w).sum(1)
	logp = Bernoulli(logits=logits).log_prob(target)
	return logp.sum()


##################################################
### Log-prior ###
def log_prior(w):
	logp = Normal(torch.zeros_like(w), 1.0*torch.ones_like(w)).log_prob(w)
	return torch.mean(logp.sum(1))



