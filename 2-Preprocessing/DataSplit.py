import numpy as np
from sklearn.model_selection import train_test_split
from  def_dict import *

def save_data_split(data_path, x_data, y_data, test_size):

    x_path = data_path+'/'+x_data
    y_path = data_path+'/'+y_data
    x_data = np.load(x_path)
    y_data = np.load(y_path)

    x_data = np.reshape((x_data/255), (x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
    y_data = y_data

    x_train, x_test, y_train, y_test = train_test_split( x_data,  y_data, test_size=test_size, random_state=321)
    print('* Training data shape: ', x_train.shape, y_train.shape)
    print('* Test data shape : ', x_test.shape, y_test.shape)
    
    np.save(data_path+'/x_train', x_train)
    np.save(data_path+'/y_train', y_train)
    np.save(data_path+'/x_test', x_test)
    np.save(data_path+'/y_test', y_test)
