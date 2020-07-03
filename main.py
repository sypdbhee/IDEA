import argparse, os
import numpy as np
import tensorflow as tf
from data import load_train_data, load_test_data
from data import load_sum_train_data
from model import train, test
from soft import s_train, s_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str)
    parser.add_argument('--gpus', type=str, default='2') #determine gpu to use
    parser.add_argument('--mode', type=str, default='train') 
    parser.add_argument('--path', type=str) #determine path to save
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.mode == 'train':
        data = load_train_data(args.data)
        train(data, args.path)
    elif args.mode == 's_train':
        data = load_sum_train_data()
        s_train(data, args.path)
    elif args.mode == 's_test':
        data = load_test_data(args.data)
        s_test(data, args.path, args.data)

