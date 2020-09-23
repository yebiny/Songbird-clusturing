import os, sys
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

sys.path.append('../')
from basicDrawing import *

def save_data_split(data_path, save_path, split_size):

    data = np.load(data_path)
    x_data = np.expand_dims(data, axis=3)    
    print('* Convert data shape ', data.shape, ' to ',x_data.shape)
    
    x_train, x_test, _,_ = train_test_split( x_data, x_data, test_size=split_size, random_state=321)
    x_valid, x_test, _,_ = train_test_split( x_test, x_test, test_size=split_size, random_state=321)
    print('* Splited data shape: ', x_train.shape, x_valid.shape, x_test.shape)
    
    np.save(save_path+'/x_train', x_train)
    np.save(save_path+'/x_valdi', x_valid)
    np.save(save_path+'/x_test', x_test)


def main():
    opt = argparse.ArgumentParser()
    opt.add_argument(dest='data_path', type=str, help='datasets path')
    opt.add_argument(dest='save_path', type=str, help='save path')
    opt.add_argument('-s',dest='split_size', type=float, default=0.2, help='test split size')
    argv = opt.parse_args()

    save_data_split(argv.data_path, argv.save_path, argv.split_size) 

if __name__=='__main__':
    main()
