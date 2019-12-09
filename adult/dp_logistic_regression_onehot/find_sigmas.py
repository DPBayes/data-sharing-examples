import pandas as pd
import numpy as np
from privacy.analysis.compute_dp_sgd_privacy import get_privacy_spent, compute_rdp

def find_sigma(target_eps, U, L, q, T, target_delta, ant_T=1):
	max_iter = 100
	n_iter = 0
	while True:
		m = (U+L)/2
		rdp_eps = compute_rdp(q, m, T, range(2,500))
		eps = get_privacy_spent(range(2, 500), rdp_eps, target_delta=target_delta)[0]*2*ant_T
		if np.abs(eps-target_eps)<0.01 and eps > target_eps: return m, eps
		if eps > target_eps:
			L = m
		else : 
			U = m
		n_iter += 1
		if n_iter==max_iter:
			break
	print("max nmbr of iter exceed")
	return m, eps

def find_T(target_eps, U, L, q, sigma, target_delta):
	max_iter = 100
	n_iter = 0
	while True:
		m = (U+L)/2
		rdp_eps = compute_rdp(q, sigma, m, range(2,500))
		eps = get_privacy_spent(range(2, 500), rdp_eps, target_delta=target_delta)[0]*2
		if np.abs(eps-target_eps)<0.01 and eps < target_eps: return m, eps
		if eps > target_eps:
			U = m
		else : 
			L = m
		n_iter += 1
		if n_iter==max_iter:
			break
	print("max nmbr of iter exceed")
	return m, eps

