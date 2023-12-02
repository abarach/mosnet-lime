"""
Author: Ada Lamba
Course: CSE 5539
Instructor: Khalili
Version: 12/02/2023

Explaining MOSNet Predictions via LIME Framework:
    In this file, we use the pretrained CNN-BLSTM model from [MOSNet](https://github.com/lochenchou/MOSNet) to predict 
    audio quality. Then, we use the [LIME](https://github.com/marcotcr/lime) to explain MOSNet's predictions. 
"""

import os
from lime.lime_tabular import LimeTabularExplainer
import model        # MOSNet model definition
import utils
import random
random.seed(1984)   # Set seed to ensure same test/train split as MOSNet original implementation
import numpy as np

DATA_DIR = './data'
BIN_DIR = os.path.join(DATA_DIR, 'bin')
PRE_TRAINED_DIR = './pre_trained'
CLASS_NAMES = ['1', '2', '3', '4', '5']

def main():
    # preprocess data
    # num_test, num_valid, and batch_size values are defined in the MOSNet train/validation/test split
    (train_feats, train_labels, test_feats, _) = preprocess(num_test=4000, num_valid=3000, batch_size=64)
    
    # setup models and explainer
    mosnet_model, explainer = init(train_feats, train_labels)
    
    # explain a data instance
    expl_idx = np.random.randint(0, test_feats.shape[0])
    explain(explainer, test_feats, expl_idx, mosnet_model)
    
     
def preprocess(num_test, num_valid, batch_size):
    print('Loading data...', end='', flush=True)
    
    # get list of input audio files, shuffle, and split up by test/train split
    mos_list = utils.read_list(os.path.join(DATA_DIR,'mos_list.txt'))
    random.shuffle(mos_list)
    test_list = mos_list[-num_test:]
    train_list= mos_list[0:-(num_test+num_valid)]

    # generate features/labels for train and test set
    (train_feats, train_labels) = get_features(train_list, batch_size)
    (test_feats, test_labels) = get_features(test_list, batch_size)
    print('Done')
    
    return (train_feats, train_labels, test_feats, test_labels)


def get_features(data_list, batch_size):
    # get one batch's worth of data from data_list
    gen = utils.data_generator(data_list, BIN_DIR, frame=False, batch_size=batch_size)
    data = next(gen)
    
    # split up features and label
    # flatten last two dimensions: (64, 397, 257) to (64, 102029)
    feats = data[0].reshape(data[0].shape[0], -1)
    labels = [int((np.trunc(i))) for i in data[1][0]]
    
    return (feats, labels)
     
     
def init(train_feats, train_labels):
    # initialize model
    mosnet = model.CNN_BLSTM()
    mosnet_model = mosnet.build()

    # load pre-trained weights
    mosnet_model.load_weights(os.path.join(PRE_TRAINED_DIR, 'cnn_blstm.h5'))
    print('Done')

    # initialize LIME explainer
    print('Initializing LIME explainer...', end='', flush=True)
    explainer = LimeTabularExplainer(training_data=train_feats, training_labels=train_labels, class_names=CLASS_NAMES)
    print('Done')
    
    return mosnet, explainer


def explain(explainer, test_feats, test_idx, model_obj):
    print(f'Explaining test instance {test_idx}...', end='', flush=True)
    exp = explainer.explain_instance(test_feats[test_idx], model_obj.flatten_predict)
    print('Done')
    exp.show_in_notebook(show_table=True, show_all=False)

if __name__ == '__main__':
    main()