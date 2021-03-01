import torch, sys, math, pickle, datetime, time
import numpy as np
import pandas as pd
import numpy.random as npr
from itertools import count
from collections import OrderedDict


use_cuda = torch.cuda.is_available()

from linear import ReparamXpand

##################################################
### Inference ###
"""
        Runs DPVI for given parameters and returns a generative model
"""
from vi import VI
def infer(T, batch_size, Optimizer, learning_rate, train_data, variable_types, k):
        ## Initialize and expand model
        param_dims = OrderedDict()
        for key, value in variable_types.items():
                if key == 'pi_unconstrained':
                        param_dims[key] = [k-1]
                else:
                        if value == 'Bernoulli':
                                param_dims[key] = [k]
                        elif (key=='lex.dur' and variable_types[key]==None):
                                param_dims[key] = [2, k]
                        elif (key=='ep' and variable_types[key]==None):
                                param_dims[key] = [k]
                        elif (key=='dead' and variable_types[key]==None):
                                param_dims[key] = [k]
                        elif value == 'Beta':
                                param_dims[key] = [2, k]
                        elif value == 'Categorical':
                                param_dims[key] = [k, len(np.unique(train_data[key]))]
        
        input_dim = int(np.sum([np.prod(value) for value in param_dims.values()]))
        flat_param_dims = np.array([np.prod(value) for value in param_dims.values()])
        model = ReparamXpand(1, input_dim, param_dims, flat_param_dims)
        model.reparam.bias.data = model.reparam.bias.data.flatten()
        model.reparam.weight.data = model.reparam.weight.data.flatten()

        ### Init model close to feature means
        def logit(y):
                return torch.log(y)-torch.log(1.-y)
        def inverse_softmax(y):
                last = 1e-23*torch.ones(1) # just something small
                sum_term = -50.-torch.log(last)
                x = torch.log(y)-sum_term
                return x
        ### Init model close to feature means
        ## Laplace mech with small epsilon to guarantee DP of the initialization
        for key in train_data.columns:  
                if variable_types[key]=='Bernoulli' or key in ['dead']:
                        param_mean = torch.as_tensor(train_data[key].mean(0))
                        param_location = list(model.param_dims.keys()).index(key)
                        init_param = logit(torch.rand(k)*(param_mean*2.-param_mean*0.5)+param_mean*0.5)

                        start_index = np.sum(model.flat_param_dims[:param_location])
                        model.reparam.bias.data[start_index:(start_index+np.sum(model.param_dims[key]))] =\
                                                        init_param
                elif variable_types[key]=='Categorical':
                        freqs = np.unique(train_data[key], return_counts=1)[1]
                        num_cats = len(freqs)
                        param_mean = torch.as_tensor(freqs/np.sum(freqs))
                        init_param = inverse_softmax(param_mean)
                        init_param = 0.5*torch.randn(k, num_cats)+init_param
                        init_param = init_param.flatten()
                        param_location = list(model.param_dims.keys()).index(key)
                        start_index = np.sum(model.flat_param_dims[:param_location])
                        model.reparam.bias.data[start_index:(start_index+np.prod(model.param_dims[key]))] =\
                                                        init_param

                        

        if use_cuda:
                model.cuda()
        optimizer = Optimizer(model.parameters(), lr=learning_rate)
        N = len(train_data)
        model = VI(model, T, N, batch_size, train_data, optimizer, variable_types)

        ## Create a generative model based on model parameters and return it
        generative_model = ReparamXpand(1, input_dim, param_dims, flat_param_dims)
        generative_model.reparam.bias.detach_()
        generative_model.reparam.weight.detach_()
        generative_model.reparam.bias.data = torch.tensor(model.reparam.bias.data.cpu().data.numpy(), device='cpu') 
        generative_model.reparam.weight.data = torch.tensor(model.reparam.weight.data.cpu().data.numpy(), device='cpu')
        #return generative_model, z_maps
        return generative_model


##################################################
### Load diabetes data ###
## Encode data
from load_diabetes import fetch_data
female_df, male_df, data_dtypes = fetch_data()
data_dtypes['G03.DDD'] = 'int64'
female_N = len(female_df)
male_N = len(male_df)

##################################################
### Define model ###
## For female

# Load variable type dictionaries for both independent and dependent types
from variable_types import independent_model as female_variable_types_

dead_female_variable_types = female_variable_types_.copy()
dead_female_variable_types.pop('dead')
alive_female_variable_types = dead_female_variable_types.copy()
alive_female_variable_types.pop('ep')
alive_female_variable_types.pop('lex.dur')

# Pick features for training
female_features = list(female_variable_types_.keys())
female_features.remove('pi_unconstrained')

# Cast features to appropriate dtypes
female_dtypes = {key:value if value!='O' else 'int64' for key, value in \
                                                                        data_dtypes[female_features].items()} 

alive_features = list(alive_female_variable_types.keys())
alive_features.remove('pi_unconstrained')
dead_features = list(dead_female_variable_types.keys())
dead_features.remove('pi_unconstrained')

# Separate training datas to alives and deads
alive_female_df = female_df[female_df.dead == 0][alive_features]
dead_female_df = female_df[female_df.dead == 1][dead_features]

def main():
        # Set DPVI params
        #T = 10000
        T = 40000
        C = 1.0
        #lr = 1e-2
        lr = 1e-3
        # set number of mixture components
        female_k = 40
        q = 0.005
        n_runs = int(sys.argv[1])
        seed = int(sys.argv[2])
        # Set optimizer
        optimizer = torch.optim.Adam
        ## Set random seed
        npr.seed(seed)
        if use_cuda:
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
                torch.cuda.manual_seed(seed)
        else:
                torch.set_default_tensor_type('torch.DoubleTensor')
                torch.manual_seed(seed)

        ## Compute privacy budget
        print("NON DP RUN!! k = {}".format(female_k))

        ## Save parameters
        res_dir = './res/'
        params = {'T':T, 'C':C, 'lr':lr, 'female_k':female_k,\
                                'q':q, 'n_runs':n_runs, 'seed':seed}
        ## Determine filename
        fname_i = 0
        date = datetime.date.today().isoformat()
        fname = 'k={}_{}_{}'.format(female_k, date, seed)
        while True:
                try : 
                        param_file = open(res_dir+'params_{}_NONDP.p'.format(fname), 'r')
                        param_file.close()
                        if fname_i == 0: fname += '_({})'.format(fname_i)
                        else: fname = fname[:-4]+'_({})'.format(fname_i)
                        fname_i += 1
                except :
                        break
                        
        pickle.dump(params, open(res_dir+'params_{}_NONDP.p'.format(fname), 'wb'))
        learn_counter = count()
        alive_female_models = []
        dead_female_models = []
        out_file = open(res_dir+'out_{}_NONDP.txt'.format(fname), 'w')
        for i in range(n_runs):
                start_time = time.time()
                print(learn_counter.__next__())
                # train female and models
                # alives
                alive_female_model = infer(T, int(q*len(alive_female_df)),\
                        optimizer, lr, alive_female_df, alive_female_variable_types, female_k)
                alive_female_models.append(alive_female_model)
                pickle.dump(alive_female_models, open('./female_models/'+'alive_female_models_{}_NONDP.p'\
                                        .format(fname), 'wb'))
                # deads
                dead_female_model = infer(T, int(q*len(dead_female_df)),\
                        optimizer, lr, dead_female_df, dead_female_variable_types, female_k)
                dead_female_models.append(dead_female_model)
                pickle.dump(dead_female_models, open('./female_models/'+'dead_female_models_{}_NONDP.p'\
                                        .format(fname), 'wb'))
                stop_time = time.time()
                time_delta = stop_time-start_time
                out_file.writelines("Took {} seconds to learn alive and dead\n".format(time_delta))
                print("Took {} seconds to learn alive and dead\n".format(time_delta))
        out_file.close()
if __name__ == "__main__":
        main()
