from __future__ import print_function
import numpy as np
import pickle
import pdb

def load_train_data(i):
    with open('./Data/train/noisy_'+i+'.txt', 'rb') as fpd:
        dat = pickle.load(fpd)
    with open('./Data/train/clean.txt', 'rb') as fpl:
        lab = pickle.load(fpl)
    #pdb.set_trace()
    return dict(x=dat, y=lab)            

def load_test_data(i):
    with open('./Data/test/noisy_'+i+'.txt', 'rb') as fpd:
        dat = pickle.load(fpd)
    with open('./Data/test/clean.txt', 'rb') as fpl:
        lab = pickle.load(fpl)
    #pdb.set_trace()

    return dict(x = dat, y = lab)

def load_sum_train_data():
    with open('./Data/train/noisy_3.txt', 'rb') as fp:
        data_1 = pickle.load(fp)                
    with open('./Data/train/noisy_6.txt', 'rb') as fp:
        data_2 = pickle.load(fp)                
    with open('./Data/train/noisy_9.txt', 'rb') as fp:
        data_3 = pickle.load(fp)
    dat = np.concatenate([data_1, data_2, data_3], axis=0)
    with open('./Data/train/clean.txt', 'rb') as fp:
        lab = pickle.load(fp)

    return dict(x=dat, y=lab)
