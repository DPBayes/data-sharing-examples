This folder among contains the code for running the mixture model with Adult (UCI) dataset.
To reproduce the results:
	* Learn probabilistic models using mixture_main.py in ./mixture_model
	* The script takes three parameters the perturbation level sigma, the income class (poor/rich) and the random seed (seeds and sigmas are listed in seeds.txt, sigmas.txt)
	* After learning the models, run create_onehot_data.py and create_onehot_data_disc.py to create onehotted versions of the data for classification and classify_mixture_onehot.py in ./mixture_model which will learn the classifiers
	* To obtain the comparison against tailored mechanism, run adult_main_anticipated.py in ./dp_logistic_regression with parameters described in params.txt
