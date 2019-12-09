import sys, os
import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict as od


def main():
	eps = sys.argv[1]
	for ins in ['alive', 'dead']:
		model_fnames = os.system("ls ./female_models/ |grep {}_female_models_2019 |grep {} >> model_fnames.txt".format(ins, eps))
		model_fnames_file = open("model_fnames.txt", "r")
		model_fnames = model_fnames_file.readlines()
		model_fnames_file.close()
		os.system("rm model_fnames.txt")
		len_per_seeds = {}
		used_seeds = []
		names_to_pick = []
		for i_name, name in enumerate(model_fnames):
			name = './female_models/'+name.strip('\n')
			seed = name.split('_')[5]
			model = pd.read_pickle(name)
			if seed in used_seeds:
				if len(model) > len_per_seeds[seed]:
					names_to_pick.remove([name for name in names_to_pick if seed in name][0])
					names_to_pick.append(name)
					len_per_seeds[seed] = len(model)
			else:
				names_to_pick.append(name)
				len_per_seeds[seed] = len(model)
				used_seeds.append(seed)
		models = []
		for name in names_to_pick:
			model = pd.read_pickle(name)
			models = models + model

		print("Joined {} runs, total number : {}".format(eps, len(models)))
		pickle.dump(models, open("./female_models/{}_female_models_{}.p".format(ins, eps), "wb"))

if __name__=="__main__":
	main()
