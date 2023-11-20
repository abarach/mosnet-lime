"""
Author: Ada Lamba
Course: CSE 5539
Instructor: Khaliligarekani
Version: 11/13/2023

Explaining MOSNet Predictions via LIME Framework:
    In this file, we use the pretrained CNN-BLSTM model from [MOSNet](https://github.com/lochenchou/MOSNet) to predict 
    audio quality. Then, we use the [LIME](https://github.com/marcotcr/lime) to explain MOSNet's predictions. 
"""

import os
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import model        # MOSNet model definition
import utils
import random
random.seed(1984)   # Set seed to ensure same test/train split as MOSNet original implementation

DATA_DIR = './data'
BIN_DIR = BIN_DIR = os.path.join(DATA_DIR, 'bin')
PRE_TRAINED_DIR = './pre_trained'
CLASS_NAMES = ['1', '2', '3', '4', '5']
NUM_TEST=4000       # Defined in MOSNet train/validation/test split


def main():
    # setup
    init()
    
    
def init():
    """
    Import necessary models, load the pre-trained MOSNet model (CNN-BLSTM version), initialize the LIME explainer, 
    and load the input data. For the LIME framework, we use the tabular implementation. 
    There is no LIME implementation for audio, so we first attempt to use the LIME explainer on the matrix 
    representation of a sound file. 
    """
    # initialize model
    MOSNet = model.CNN_BLSTM()
    model = MOSNet.build()

    # load pre-trained weights
    model.load_weights(os.path.join(PRE_TRAINED_DIR, 'cnn_blstm.h5'))

    # initialize LIME explainer
    explainer = LimeTabularExplainer(class_names=CLASS_NAMES)

    # load input data
    mos_list = utils.read_list(os.path.join(DATA_DIR,'mos_list.txt'))
    random.shuffle(mos_list)
    test_list = mos_list[-NUM_TEST:]        # split test/train
    filepath = test_list[0].split(',')      # grab first test instance - TODO: use all test instances
    filename = filepath[0].split('.')[0]

    # get magnitude spectrogram feature (input for MOSNet)
    _feat = utils.read(os.path.join(BIN_DIR,filename+'.h5'))
    _mag = _feat['mag_sgram'] 


if __name__ == '__main__':
    main()