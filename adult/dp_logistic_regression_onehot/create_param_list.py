from itertools import product

target_epsilons = [1.1, 2.0, 4.0, 8.0, 14.0]
seed_file = open('seeds.txt', 'r')
seeds = seed_file.readlines()
seed_file.close()

param_file = open('params.txt', 'w')
for eps, seed in product(target_epsilons, seeds):
	param_string = str(eps) + " " + str(seed)
	param_file.writelines(param_string)
param_file.close()
