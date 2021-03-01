import torch, sys, math
import numpy.random as npr


### VI for diabetes data ###
### NON PRIVATE VERSION ###

from diabetes_model import log_likelihood, log_prior

def VI(model, T, N, batch_size, train_data, optimizer, variable_types):
	input_dim = model.input_dim # This is the dimensionality of variational posterior
	for i in range(T):
		## Take minibatch
		minibatch = train_data.sample(batch_size, replace=False)
		## Reset optimizer and ELBO
		optimizer.zero_grad()
		elbo = 0
		## Draws for mc integration
		draws = torch.randn(input_dim)
		## MC integration for likelihood part of ELBO
		draw = model.forward(draws)
		log_likelihood_loss = -1.*log_likelihood(minibatch,\
				draw, variable_types)
		elbo += log_likelihood_loss
		## Take the backward call of the likelihood (the only data dependent) contribution.
		log_likelihood_loss.backward(retain_graph=True)

		log_prior_loss = -(batch_size/N)*log_prior(draw, variable_types)
		elbo += log_prior_loss
		log_prior_loss.backward(retain_graph=True)

		## Compute the entropy contribution.
		## For MVN, using our parametrization, the entropy grads w.r.t $s_q$ are ones and w.r.t $\mu_q$ zeros.
		entropy_weight_grad = -(batch_size/N)*torch.ones(input_dim)
		## add to weight grads
		model.reparam.weight.grad.data += entropy_weight_grad

		## if nans, break
		if torch.any(torch.isnan(model.reparam.bias.grad.data)) or torch.any(torch.isnan(model.reparam.weight.grad.data)):
			break
		## Finally take the gradient step.
		optimizer.step()
		if i % 10 == 0: 
			sys.stdout.write('\r{}% : ELBO = {}'.format(int(i*100/T),-1.*elbo.data.tolist()))
		if i == T-1: 
			sys.stdout.write('\rDone : ELBO = {}\n'.format((-1.*elbo.data.tolist())))
		sys.stdout.flush()
	return model
