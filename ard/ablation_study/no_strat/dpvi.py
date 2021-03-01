import torch, sys, math
import numpy.random as npr


### DPVI for diabetes data ###
from diabetes_model import log_likelihood, log_prior

def DPVI(model, T, N, batch_size, train_data, sigma, C, optimizer, variable_types):
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
			## Fetch the gradient of expanded $\mu_q$
			bias_grad = model.reparam.bias.grad.data # \in \mathbb{R}^{B \times d}
			## Compute the gradient norm for each row i.e. for each individual in batch
			bias_grad_norm = bias_grad.norm(dim=-1)
			## Compute the clipping scale. If bias_grad_norm[i]<C => C/bias_grad_norm[i]>1 => bias_grad_scale[i] = 1
			## and if bias_grad_norm[i]>C => C/bias_grad_norm[i]<1 => bias_grad_scale[i] = C/bias_grad_norm[i]
			bias_grad_scale = torch.clamp(C/bias_grad_norm, 0.0, 1.0)
			## Since bias_grad \in \mathbb{R}^{B \times d}, we need to make bias_grad_scale B x 1 dimensional in order
			## to multiply each row of bias_grad with corresponding element of bias_grad_scale.
			## Now in bias_grad_norm[i]>C => bias_grad_scale[i] = C / bias_grad_norm[i] => clipped_bias_grad[i] = bias_grad[i]*C/bias_grad_norm[i]
			## and thus bias_|| clipped_bias_grad[i] ||_2 = (C/bias_grad_norm[i])*||bias_grad[i]||_2 = C
			clipped_bias_grad = bias_grad*bias_grad_scale.unsqueeze(-1)
			## Next, we sum the gradients over batch and add the DP-noise
			noisy_bias_grad = clipped_bias_grad.sum(0)+noise_b
			## Using the property of reparametrization trick for mean-field Gaussian, we compute the gradient of $s_q$ using noisy gradient of $\mu_q$
			noisy_weight_grad = draws*noisy_bias_grad*torch.exp(model.reparam.weight.data[0])
			## Lastly store the gradients of log-likelihood
			ll_weight_grad = noisy_weight_grad.clone()
			ll_bias_grad = noisy_bias_grad.clone()
			## We zero the gradients stored in model.reparam.<param>.grad.data and proceed to compute the log-prior contribution.
			optimizer.zero_grad()

		draw = model.forward(draws)
		log_prior_loss = -(batch_size/N)*log_prior(draw, variable_types)
		elbo += log_prior_loss
		log_prior_loss.backward(retain_graph=True)

		## Store the log-prior gradients
		logprior_weight_grad = model.reparam.weight.grad.data.sum(0).clone()
		logprior_bias_grad = model.reparam.bias.grad.data.sum(0).clone()

		## Compute the entropy contribution.
		## For MVN, using our parametrization, the entropy grads w.r.t $s_q$ are ones and w.r.t $\mu_q$ zeros.
		entropy_weight_grad = -(batch_size/N)*torch.ones(input_dim)

		## Now sum all the different gradient contributions and expand to $\mathbb{R}^{B \times d}$.
		total_bias_grad = (ll_bias_grad+logprior_bias_grad).repeat(batch_size)\
									.reshape_as(model.reparam.bias.data)
		total_weight_grad = (ll_weight_grad+logprior_weight_grad+entropy_weight_grad).repeat(batch_size)\
									.reshape_as(model.reparam.weight.data)
		
		## Set the gradients to parameters.
		model.reparam.bias.grad.data = total_bias_grad
		model.reparam.weight.grad.data = total_weight_grad

		## if nans, break
		if torch.any(torch.isnan(total_bias_grad)) or torch.any(torch.isnan(total_weight_grad)):
			break
		## Finally take the gradient step.
		optimizer.step()
		if i % 10 == 0: 
			sys.stdout.write('\r{}% : ELBO = {}'.format(int(i*100/T),-1.*elbo.data.tolist()))
		if i == T-1: 
			sys.stdout.write('\rDone : ELBO = {}\n'.format((-1.*elbo.data.tolist())))
		sys.stdout.flush()
	return model
