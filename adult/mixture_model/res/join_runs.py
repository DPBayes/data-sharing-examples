import pickle
import sys

seeds = range(1234, 1244)

def main():
	sigma = sys.argv[1]
	models = []
	for seed in seeds:
		model = pickle.load(open('models_poor_2019-04-25_{}_{}.p'.format(sigma, seed), 'rb'))[0]
		models.append(model)
	pickle.dump(models, open('models_poor_2019-04-25_{}.p'.format(sigma), 'wb'))

if __name__ == "__main__":
	main()
