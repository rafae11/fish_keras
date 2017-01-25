from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D
from keras.engine.topology import Merge
from keras.layers import merge
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D

import PIL
import inception
import tensorflow as tf
import keras
import glob
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

base_path = '/media/rafael/Data1/sample/'

train_path = base_path + 'train'
validation_path = base_path + 'valid'

class_names = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
s = pd.Series(class_names)

class_one_hot = pd.get_dummies(s).values.tolist()

flist = class_names
#flist = ['ALB_SCALED','BET_SCALED','DOL_SCALED','LAG_SCALED','OTHER_SCALED','SHARK_SCALED','NoF_SCALED','YFT_SCALED']

#pickle files
ft_image_path = 'fishy_train_image.pkl'
ft_label_path = 'fishy_train_label.pkl'
fv_image_path = 'fishy_valid_image.pkl'
fv_label_path = 'fishy_valid_label.pkl'
ft_test_path  = 'fishy_test_image.pkl'

test_path = glob.glob('/media/rafael/Data1/sample/test/test_stg1/*.jpg')

def append_image_paths(index,ftype, train_image, valid_image):

    tpath = train_path + '/' + ftype + '/*.jpg'
    vpath = validation_path + '/' +  ftype + '/*.jpg'
    
    globt = glob.glob(tpath)
    globv = glob.glob(vpath)

    tvalue = len(globt)
    vvalue = len(globv)
    
    for i in range(0,tvalue):
        train_image.append(globt[i])
    
    for i in range(0,vvalue):   
        valid_image.append(globv[i])
        
def append_one_hot(index,ftype,train_label,valid_label):

    tpath = train_path + '/' + ftype + '/*.jpg'
    vpath = validation_path + '/' + ftype + '/*.jpg'
    
    tvalue = len(glob.glob(tpath))
    vvalue = len(glob.glob(vpath))
          
    for i in range(0,tvalue):        
        train_label.append(class_one_hot[index])
        
    for i in range(0,vvalue):        
        valid_label.append(class_one_hot[index])

def show_image(path):
    print(path)
    img = Image.open(path)
    plt.imshow(img)
    plt.show()
def show_list(path,label):
    for i in range(0,len(path)):
        print("%s\n%s" %(path[i],label[i]))