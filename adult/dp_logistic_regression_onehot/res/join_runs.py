import pickle

target_epsilons = [1.1, 2.0, 4.0, 8.0, 14.0]
seeds = range(1234,1244)

for eps in target_epsilons:
	models = []
	for seed in seeds:
		models_ = pickle.load(open('models_2019-11-05_{}_{}.p'.format(eps, seed), 'rb'))
		models += models_
	pickle.dump(models, open('models_2019-11-05_{}.p'.format(eps), 'wb'))
