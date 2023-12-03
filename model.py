from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm
from keras.preprocessing import image
import numpy as np
import utils
import os

DATA_DIR = './data'
BIN_DIR = os.path.join(DATA_DIR, 'bin')

class CNN_BLSTM(object):
    
    def __init__(self):
        print('CNN_BLSTM init...', end='')
        
    def build(self):
        _input = keras.Input(shape=(None, 257))
        
        re_input = layers.Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)
        
        # CNN
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
        conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
        re_shape = layers.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)

        # BLSTM
        blstm1 = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3, 
                 recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)), 
            merge_mode='concat')(re_shape)
    
        # DNN
        flatten = TimeDistributed(layers.Flatten())(blstm1)
        dense1=TimeDistributed(Dense(128, activation='relu'))(flatten)
        dense1=Dropout(0.3)(dense1)

        frame_score=TimeDistributed(Dense(1), name='frame')(dense1)

        average_score=layers.GlobalAveragePooling1D(name='avg')(frame_score)

        model = Model(outputs=[average_score, frame_score], inputs=_input)
        self.model = model
        
        return model
    

    def flatten_predict(self, data):
        # reshape data from (1, _) to (1, 397, 257)
        unflat = data.reshape((1, 397, 257))
        
        # call prediction method
        [y_pred, _] = self.model.predict(unflat, verbose=0, batch_size=1)
        
        return y_pred
    

    def image_predict(self, im_data_list):
        preds = []
        
        if len(im_data_list) != len(self.data_instance):
            print('Error! Self.data_instance does not match im_data_list')
            exit(-1)
        
        for idx in range(len(im_data_list)):
            # call MOSNet
            [y_pred, _] = self.model.predict(self.data_instance[idx], verbose=0, batch_size=1)
            
            # convert to classification
            class_pred_idx = int(np.trunc(y_pred[0][0]))-1
            probs = [0, 0, 0, 0, 0]
            probs[class_pred_idx] = 1
            preds.append(probs)
            
        return preds
    
    
    def set_instance(self, fname_list):
        self.data_instance = []
        
        for fname in fname_list:
            # fname has format dir/[name]-spec.png
            # get [name]
            filename = fname.split('/')[-1]
            filename = filename.split('-')[0]
        
            _feat = utils.read(os.path.join(BIN_DIR,filename+'.h5'))
            _mag = _feat['mag_sgram']
            self.data_instance.append(_mag)
        

class CNN(object):
    
    def __init__(self):
        print('CNN init')
        
    def build(self):
        _input = keras.Input(shape=(None, 257))
        
        re_input = layers.Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)
        
        # CNN
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
        conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
        # DNN
        flatten = TimeDistributed(layers.Flatten())(conv4)
        dense1=TimeDistributed(Dense(64, activation='relu'))(flatten)
        dense1=Dropout(0.3)(dense1)

        frame_score=TimeDistributed(Dense(1), name='frame')(dense1)

        average_score=layers.GlobalAveragePooling1D(name='avg')(frame_score)
        
        model = Model(outputs=[average_score, frame_score], inputs=_input)
        
        return model
    
    
class BLSTM(object):
    
    def __init__(self):
        print('BLSTM init')
        
    def build(self):
        _input = keras.Input(shape=(None, 257))

        # BLSTM
        blstm1 = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3, 
                 recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)), 
            merge_mode='concat')(_input)
    
        # DNN
        flatten = TimeDistributed(layers.Flatten())(blstm1)
        dense1=TimeDistributed(Dense(64, activation='relu'))(flatten)
        dense1=Dropout(0.3)(dense1)

        frame_score=TimeDistributed(Dense(1), name='frame')(dense1)

        average_score=layers.GlobalAveragePooling1D(name='avg')(frame_score)
        
        model = Model(outputs=[average_score, frame_score], inputs=_input)
        
        return model

