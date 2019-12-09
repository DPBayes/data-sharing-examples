import torch, sys, math
from adult_model import log_likelihood, log_prior
from linear import ReparamXpand
from transform import softmax

log_2pi = math.log(2*math.pi)
def mvn_entropy(model):
	return 0.5*torch.mean(torch.sum(log_2pi+1+2*model.weight, dim=-1))

def DPVI(model, T, data, batch_size, optimizer, C, sigma, variable_types, verbose=False):
	input_dim = model.input_dim
	N = data.shape[0]
	for i in range(T):
		## Take minibatch
		minibatch = data.sample(batch_size, replace=False)
		## Reset optimizer and ELBO
		optimizer.zero_grad()
		elbo = 0
		## Draws for mc integration
		draw_ = torch.randn(1, input_dim)
		## MC integration for likelihood part of ELBO
		draw = model.forward(draw_[0])

		## Compute the log-likelihood contribution		
		log_likelihood_loss = -1*log_likelihood(minibatch,\
				draw, variable_types, use_cuda=False)
		elbo += log_likelihood_loss
		log_likelihood_loss.backward(retain_graph=True) # Backward call from the data dependent part
		## sigma is the std of DP-noise. Thus if we are running with DP, we clip and perturb
		if sigma>0:
			"""
			 Using the reparametrization trick, we can write $z = \mu_q + \exp(\log \sigma_q)*\eta$, where \eta \sim N(0,1).
			 Now, as our loss function L can essentially be written as $L = f(z; X)$, the derivative w.r.t $\mu_q$
			 will be f'(z ; X) and w.r.t $\log \sigma_q = s_q$ it will be $\exp(s)\eta f'(z; X) = \exp(s_q)\eta \nabla_{\mu_q} L$.
			 Thus it suffices to privately compute $\nabla_{\mu_q} L$ and based on that compute $\nabla_{s_q} L$ since 
			 the $\exp{s}\eta$ factor of the $\nable_{s_q} L$ will be data independent and thus be considered as post-processing.
			"""
			## Draw the DP noise from N(0, C^2 sigma^2 I), where I=eye(d)			
			noise_b = sigma*C*torch.randn(input_dim)
			## Compute the clipping scale
			clip_bound = torch.clamp(model.reparam.bias.grad.data.norm(dim=1)/C, 1.0)
			## Clip gradients 
			model.reparam.bias.grad.data = model.reparam.bias.grad.data.div(clip_bound.unsqueeze(1))
			## Add noise
			model.reparam.bias.grad.data = (model.reparam.bias.grad.data.sum(0)+noise_b)\
										.repeat(batch_size).view_as(model.reparam.bias.grad.data)
			## Using the property of reparametrization trick for mean-field Gaussian, we compute the gradient of $s_q$ using noisy gradient of $\mu_q$
			model.reparam.weight.grad.data = model.reparam.bias.grad.data*\
					model.reparam.weight.data.exp()*draw_[0]
			ll_bias_grad = model.reparam.bias.grad.data.clone() # save likelihood_grads
			ll_weight_grad = model.reparam.weight.grad.data.clone() # save likelihood_grads
			optimizer.zero_grad() # zero the gradients and proceed to computing prior and entropy contributions
			draw = model.forward(draw_[0])
			
		log_prior_loss = -(batch_size/N)*log_prior(draw, variable_types)
		elbo += log_prior_loss
		log_prior_loss.backward(retain_graph=True)
		if sigma>0:
			## Replicate prior gradient contribution to all expanded grads
			model.reparam.weight.grad.data = model.reparam.weight.grad.data[0].repeat(batch_size).\
											view_as(model.reparam.weight.grad.data)
			model.reparam.bias.grad.data = model.reparam.bias.grad.data[0].repeat(batch_size).\
											view_as(model.reparam.bias.grad.data)

		## Add entropy to ELBO
		entropy = -(batch_size/N)*mvn_entropy(model.reparam)
		elbo += entropy
		entropy.backward(retain_graph=True)
		if sigma>0:
			## Add log-likelihood grad contributions to grads
			model.reparam.weight.grad.data.add_(ll_weight_grad)
			model.reparam.bias.grad.data.add_(ll_bias_grad)
		# Average gradients
		model.reparam.bias.grad.data.mul_(N/batch_size)
		model.reparam.weight.grad.data.mul_(N/batch_size)
		
		## Take step
		optimizer.step()
		if verbose:
			if i % 10 == 0: 
				sys.stdout.write('\r{}% : ELBO = {}'.format(int(i*100/T),-1.*elbo.data.tolist()))
			if i == T-1: 
				sys.stdout.write('\rDone : ELBO = {}\n'.format((-1.*elbo.data.tolist())))
			sys.stdout.flush()

	model_ =  ReparamXpand(1, model.input_dim, model.param_dims, model.flat_param_dims)
	model_.reparam.bias.data = model.reparam.bias.data[0]
	model_.reparam.weight.data = model.reparam.weight.data[0]
	model_.reparam.bias.detach_()
	model_.reparam.weight.detach_()
	return model_
