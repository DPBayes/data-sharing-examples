import torch, sys, math
import numpy.random as npr

log_2pi = math.log(2*math.pi)
def mvn_entropy(model):
	return 0.5*torch.mean(torch.sum(log_2pi+1+2*model.weight, dim=-1))

### DPVI for diabetes data ###
from adult_lr_model import log_likelihood, log_prior
from utils import clip

def DPVI(model, T, n_mc, N, batch_size, train_data, sigma, C, optimizer, use_cuda=False):
	input_dim = model.input_dim

	for i in range(T):
		## Take minibatch
		minibatch = train_data.sample(batch_size, replace=False)
		## Reset optimizer and ELBO
		optimizer.zero_grad()

		elbo = 0
		## Draws for mc integration
		draws = torch.randn(n_mc, input_dim)
		## MC integration for likelihood part of ELBO
		for j in range(n_mc):
			draw = model.forward(draws[j])
			log_likelihood_loss = -1./n_mc*log_likelihood(minibatch.iloc[:, :-1],\
					minibatch.iloc[:, -1].astype('double'), draw, use_cuda)
			elbo += log_likelihood_loss
			log_likelihood_loss.backward(retain_graph=True)
			
		## Clip and add noise
		if sigma>0:
			noise_w = sigma*C*torch.randn(input_dim)
			noise_b = sigma*C*torch.randn(input_dim)
			clip(model, C)
			g = torch.cat((model.reparam.weight.grad.data, model.reparam.bias.grad.data), 1).clone()
			if not torch.all(g.norm(dim=1)<(C+1e-9)):
				print(g.norm(dim=1).max())
				print(torch.any(torch.isnan(g)))
				return model
			model.reparam.weight.grad.add_(noise_w/batch_size)
			model.reparam.bias.grad.add_(noise_b/batch_size)

		## MC integration for prior part of ELBO
		for j in range(n_mc):
			draw = model.forward(draws[j])
			log_prior_loss = -(batch_size/N)*log_prior(draw)/n_mc
			elbo += log_prior_loss
			log_prior_loss.backward(retain_graph=True)

		## Add entropy to ELBO
		entropy = -(batch_size/N)*mvn_entropy(model.reparam)
		elbo += entropy
		entropy.backward(retain_graph=True)

		## Take step
		optimizer.step()
		if i % 10 == 0: 
			sys.stdout.write('\r{}% : ELBO = {}'.format(int(i*100/T),-1.*elbo.data.tolist()))
		if i == T-1: 
			sys.stdout.write('\rDone : ELBO = {}\n'.format((-1.*elbo.data.tolist())))
		sys.stdout.flush()
	return model
