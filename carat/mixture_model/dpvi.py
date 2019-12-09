import torch, sys, math
import numpy.random as npr
from carat_model import log_likelihood, log_prior

log_2pi = math.log(2*math.pi)

def mvn_entropy(model):
    return 0.5*torch.mean(torch.sum(log_2pi+1+2*model.weight, dim=-1))

def DPVI(model, T, N, batch_size, X_apps, sigma, C, optimizer):
	from utils import clip
	input_dim = model.reparam.weight.shape[-1]
	for i in range(T):
		draw_ = torch.randn(1, input_dim)
		## Take minibatch
		indices = npr.choice(N, batch_size, replace=False)
		## Reset optimizer and ELBO
		optimizer.zero_grad()
		elbo = 0
		## Draws for mc integration
		draw = model.forward(draw_)
		pi_unconstrained = draw['pi_unconstrained']
		theta_unconstrained = draw['theta_unconstrained']
		## MC integration for likelihood part of ELBO
		log_likelihood_loss = -1.*log_likelihood(X_apps[indices],\
				theta_unconstrained, pi_unconstrained)
		elbo += log_likelihood_loss
		log_likelihood_loss.backward(retain_graph=True)

		## Clip and perturb gradients
		if sigma>0:
			noise_b = sigma*C*torch.randn(input_dim)
			clip_bound = torch.clamp(model.reparam.bias.grad.data.norm(dim=1)/C, 1.0)
			model.reparam.bias.grad.data = model.reparam.bias.grad.data.div(clip_bound.unsqueeze(1))

			model.reparam.bias.grad.data = (model.reparam.bias.grad.data.sum(0)+noise_b)\
										.repeat(batch_size).view_as(model.reparam.bias.grad.data)
			model.reparam.weight.grad.data = model.reparam.bias.grad.data*\
					model.reparam.weight.data.exp()*draw_[0]
			ll_bias_grad = model.reparam.bias.grad.data.clone() # save likelihood_grads
			ll_weight_grad = model.reparam.weight.grad.data.clone() # save likelihood_grads
			optimizer.zero_grad()

		### MC integration for prior part of ELBO
		log_prior_loss = -1.*log_prior(theta_unconstrained[0], pi_unconstrained[0])*(batch_size/N)
		elbo += log_prior_loss 
		log_prior_loss.backward(retain_graph=True)
		if sigma>0:
			model.reparam.weight.grad.data = model.reparam.weight.grad.data[0].repeat(batch_size).\
											view_as(model.reparam.weight.grad.data)
			model.reparam.bias.grad.data = model.reparam.bias.grad.data[0].repeat(batch_size).\
											view_as(model.reparam.bias.grad.data)

		### Add entropy to ELBO
		entropy = -1.*mvn_entropy(model.reparam)*(batch_size/N)
		elbo += entropy
		entropy.backward(retain_graph=True)
		if sigma>0:
			model.reparam.weight.grad.data.add_(ll_weight_grad)
			model.reparam.bias.grad.data.add_(ll_bias_grad)

		## Check that there are no nan's and ascend of re-try
		if (torch.isnan(model.reparam.bias.grad).any() or \
				torch.isnan(model.reparam.weight.grad).any())==False:
			optimizer.step()
		else:
			sys.stdout.write('\rNans occured in iteration %d\n' %i)
			sys.stdout.flush()
			break

		if i % 10 == 0: 
			sys.stdout.write('\r{}% : ELBO = {}'.format(int(i*100/T),-1.*elbo.data.tolist()))
			
		if i == T-1: 
			sys.stdout.write('\rDone : ELBO = {}\n'.format((-1.*elbo.data.tolist())))
		sys.stdout.flush()
	return model
