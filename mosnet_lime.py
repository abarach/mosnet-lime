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
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display

DATA_DIR = './data'
WAV_DIR = os.path.join(DATA_DIR, 'wav')
BIN_DIR = os.path.join(DATA_DIR, 'bin')
PRE_TRAINED_DIR = './pre_trained'
CLASS_NAMES = ['1', '2', '3', '4', '5']
NUM_TEST = 4000       # Defined in MOSNet train/validation/test split
NUM_VALID = 3000      # Defined in MOSNet train/validation/test split
BATCH_SIZE = 64

heatmap_path = os.path.join('lime/heatmaps/')

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
    mosnet = model.CNN_BLSTM()
    mosnet_model = mosnet.build()

    # load pre-trained weights
    mosnet_model.load_weights(os.path.join(PRE_TRAINED_DIR, 'cnn_blstm.h5'))

    # load input data
    mos_list = utils.read_list(os.path.join(DATA_DIR,'mos_list.txt'))
    random.shuffle(mos_list)
    test_list = mos_list[-NUM_TEST:]        # split test/train
    train_list= mos_list[0:-(NUM_TEST+NUM_VALID)]
    train_data = list(get_generated_data(utils.data_generator(train_list, BIN_DIR, frame=True, batch_size=BATCH_SIZE)))
    filepath = test_list[0].split(',')      # grab first test instance - TODO: use all test instances
    filename = filepath[0].split('.')[0]

    # get magnitude spectrogram feature (input for MOSNet)
    file = os.path.join(BIN_DIR, filename+'.h5')
    _feat = utils.read(file)
    _mag = _feat['mag_sgram']
    
    # get image of audio file
    audio_file = os.path.join(WAV_DIR, filename+'.wav')
    spec_transpose, sr = utils.get_spectrograms(audio_file)
    spec = np.transpose(spec_transpose)
    
    # plot file
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 9))
    
    # render original heatmap
    p1 = librosa.display.specshow(librosa.amplitude_to_db(spec,ref=np.max), cmap='jet',y_axis='linear', x_axis='time', sr=sr, ax=ax1)
    #ax1.set_title('Input Spectrogram\nTrue MOS: {}'.format(y)) #\nPredicted MOS: {}'.format(y, ypred[0][0]))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency (Hz)')
    f.colorbar(p1, ax=ax1, format="%+2.f dB")
    plt.savefig(os.path.join(heatmap_path, '{}_spec.jpeg'.format(filename)))
    plt.close(f)
    
    # get MOSNet prediction
    [y_pred, _] = mosnet_model.predict(_mag, verbose=0, batch_size=1)
    
    # convert to classification
    category_idx = int((np.trunc(y_pred) - 1)[0][0])
    class_probs = [0, 0, 0, 0, 0]
    class_probs[category_idx] = 1
    
    # initialize LIME explainer
    explainer = LimeTabularExplainer(training_data=train_data, class_names=CLASS_NAMES)
    #explainer.explain_instance(_mag)s


def get_generated_data(generator):
    """Given a data generator, returns all data generated.

    Args:
        generator (Generator): the Generator to produce data

    Yields:
        list: all data from the Generator
    """
    yield from generator

if __name__ == '__main__':
    main()