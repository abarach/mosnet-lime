"""
Author: Ada Lamba3.9
Course: CSE 5539
Instructor: Khalili
Version: 12/03/2023

Explaining MOSNet Predictions via LIME Framework:
    In this file, we use the pretrained CNN-BLSTM model from [MOSNet](https://github.com/lochenchou/MOSNet) to predict 
    audio quality. Then, we use the [LIME](https://github.com/marcotcr/lime) to explain MOSNet's predictions. 
"""

import os
from lime.lime_image import LimeImageExplainer
import model        # MOSNet model definition
import utils
import random
random.seed(1984)   # Set seed to ensure same test/train split as MOSNet original implementation
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
import pandas as pd

DATA_DIR = './data'
BIN_DIR = os.path.join(DATA_DIR, 'bin')
WAV_DIR = os.path.join(DATA_DIR, 'wav')
PRE_TRAINED_DIR = './pre_trained'
CLASS_NAMES = ['1', '2', '3', '4', '5']
HEATMAP_DIR = os.path.join(DATA_DIR, 'spec')
EXP_DIR = os.path.join(DATA_DIR, 'expl')


def parse_args():
    """Parses input arguments

    Returns:
        argparse.ArgumentParser: contains command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_spec', required=False, action='store_true', dest='generate_spec', help='Present if spectrograms should be generated.')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # preprocess data
    # num_test, num_valid, and batch_size values are defined in the MOSNet train/validation/test split
    preprocess(num_test=4000, generate_spec=args.generate_spec)
    
    # setup models and explainer
    mosnet_model, explainer = init()
    
    # explain a data instance
    explain_image(explainer, HEATMAP_DIR, mosnet_model)
    
     
def preprocess(num_test, generate_spec):
    """Reads in the input data, performs the train/test split, and returns
    the features for each input.

    Args:
        num_test (int): number of data to be in the test set
        generate_spec (bool): True if spectrogram images should be saved. False otherwise.
    """
    print('Loading data...', end='', flush=True)
    
    # get list of input audio files, shuffle, and split up by test/train split
    mos_list = utils.read_list(os.path.join(DATA_DIR,'mos_list.txt'))
    random.shuffle(mos_list)
    test_list = mos_list[-num_test:]
    print('Done')
    
    # generate spectrograms
    if generate_spec:
        generate_spectrograms(test_list)

def get_features(data_list, batch_size):
    """Calls the MOSNet data generator to produce a batch's worth of data. 

    Args:
        data_list (list): a list of data to generate from
        batch_size (int): batch size

    Returns:
        (list, list): a tuple contain the generated data features and labels
    """
    # get one batch's worth of data from data_list
    gen = utils.data_generator(data_list, BIN_DIR, frame=False, batch_size=batch_size)
    data = next(gen)
    
    # split up features and label
    # flatten last two dimensions: (64, 397, 257) to (64, 102029)
    feats = data[0].reshape(data[0].shape[0], -1)
    labels = [int((np.trunc(i))) for i in data[1][0]]
    
    return (feats, labels)
     
     
def init():
    """Initializes the MOSNet model and LIME framework.

    Returns:
        (model, LIME Explainer): the built MOSNet model and chosen LIME explainer object.
    """
    # initialize model
    mosnet = model.CNN_BLSTM()
    mosnet_model = mosnet.build()

    # load pre-trained weights
    mosnet_model.load_weights(os.path.join(PRE_TRAINED_DIR, 'cnn_blstm.h5'))
    print('Done')

    # initialize LIME explainer
    print('Initializing LIME explainer...', end='', flush=True)
    explainer = LimeImageExplainer()
    print('Done')
    
    return mosnet, explainer


def explain_image(explainer, img_dir, model_obj):
    """Runs the LIME explainer for all images in a given directory and saves the resulting heatmaps.

    Args:
        explainer (LimeImageExplainer): the initialized lime image explainer
        img_dir (str): a directory path containing images to explain
        model_obj (CNN_BLSTM): the pre-trained MOSNet CNN-BLSTM model
    """
    print('Generating explanations...')
    images = os.listdir(img_dir)
    images = [os.path.join(HEATMAP_DIR, i) for i in images]
    model_obj.set_instance(images)
    for img in images:
        im = transform_img_fn(img)
        exp = explainer.explain_instance(im.astype('double'), model_obj.image_predict, num_samples=100)
        true_mos = get_true_mos(img)
        save_expl_figs(exp, img.split('/')[-1].split('-')[0], true_mos)
        
    print('Done')
    

def save_expl_figs(explanation, im_file, true_mos):
    """Given an explanation, displays the heatmap and saves the file. 

    Args:
        explanation (LimeImageExplanation): the LIME Image Explanation object.
        im_file (str): the file path explained in [explanation]
        true_mos (int): the correct MOS score
    """
    #Select the same class explained on the figures above.
    ind =  explanation.top_labels[0] + 1

    #Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

    #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    plt.title(f'Explanation for {im_file}\nTrue MOS: {true_mos}, Predicted MOS: {ind}')
    plt.axis('off')
    plt.savefig(os.path.join(EXP_DIR, f'{im_file}-expl.png'))
    plt.close()


def get_true_mos(im_path):
    """Looks up the true MOS score for a given image file. 

    Args:
        im_path (str): the file path of the image to get the true MOS for

    Returns:
        int: the true MOS score for the given file
    """
    filename = im_path.split('/')[-1].split('-')[0]
    filename = filename + '.wav'
    mos_list = pd.read_csv(os.path.join(DATA_DIR, 'mos_list.txt'), sep=',')
    true_mos = mos_list.loc[mos_list['filename'] == filename, 'mos']
    
    return int(np.trunc(float(true_mos)))


def generate_spectrograms(file_list, num=10, start=0):
    """Generates magnitude spectrograms and saves the resulting image. 

    Args:
        file_list (list): a list of file names
        num (int, optional): the number of files to generate spectrograms for. Defaults to 10
        start (int, optional): the index to start at within file_list. Defaults to 0. 
    """
    print('Generating magnitude spectrograms...', end='', flush=True)
    for f in file_list[start:start+num]:
        file = f.split(',')[0]
        filename = file.split('.')[0]
        
        # calculate spectrogram
        transposed_spec, sr = utils.get_spectrograms(os.path.join(WAV_DIR, file))
        spec = np.transpose(transposed_spec)
        
        # plot
        fig, (ax1) = plt.subplots(1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), y_axis='linear', x_axis='time', sr=sr, ax=ax1)
        plt.axis('off')
        plt.savefig(os.path.join(HEATMAP_DIR, f'{filename}-spec.png'), bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
        
    print('Done')
        

def transform_img_fn(img_path):
    """Loads an image and converts it to a numpy array.

    Args:
        img_path (str): the image file path to load

    Returns:
        np.ndarray: a 2d numpy array representing the given image
    """
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    return x


if __name__ == '__main__':
    main()