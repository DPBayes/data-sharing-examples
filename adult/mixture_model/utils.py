import torch
import torch.nn.functional as F
import math

def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs, dim=dim)).mean(dim, keepdim=keepdim)

def clip(model, C, dim=1):
    example_norms = 0
    for p in model.parameters():
        example_norms += p.grad.data.norm(dim=dim)**2
    example_norms = torch.sqrt(example_norms)
    clip = torch.clamp(example_norms/C, 1.0)
    for p in model.parameters():
        p.grad.data = p.grad.data.div_(clip.unsqueeze(1))

def pickle_stuff(stuff, DPVI_params, pickle_name, path='./results/'):
    import pickle, datetime
    today = datetime.date.today()
    file_name_extend = '_'+str(today.day)+'_'+str(today.month)
    fne_original = file_name_extend
    if np.all(DPVI_params['sigma']==0):
        pickle_name = pickle_name+'_nondp'
    else:
        pickle_name = pickle_name+'_dp'

    fne_extend = 0
    while True:
        try:
            f = open(path+pickle_name+file_name_extend+'.p', 'rb')
            print('You are trying to override an existing pickle file: %s'%pickle_name)
            f.close()
            file_name_extend = fne_original + '('+str(fne_extend)+')'
            fne_extend+=1
        except: 
            pickle.dump(stuff, open(path+pickle_name+file_name_extend+'.p', 'wb'))
            break
    return file_name_extend

def onehot(x):
	import pandas as pd
	import numpy as np
	name = x.name
	uniques = np.unique(x.values)
	# cast to [0,..,len(uniques)]
	x_mapped = x.map({u : i for i, u in enumerate(uniques)})
	y = np.zeros([len(x), len(uniques)], dtype='int')
	y[np.arange(len(y)), x_mapped.values] = 1
	return pd.DataFrame(y, columns=[name+' : {}'.format(u) for u in uniques], index = x.index)
