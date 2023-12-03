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
from lime.lime_image import LimeImageExplainer
import model        # MOSNet model definition
import utils
import random
random.seed(1984)   # Set seed to ensure same test/train split as MOSNet original implementation
import numpy as np
import time
from keras.preprocessing import image
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
from skimage.segmentation import mark_boundaries

DATA_DIR = './data'
BIN_DIR = os.path.join(DATA_DIR, 'bin')
WAV_DIR = os.path.join(DATA_DIR, 'wav')
PRE_TRAINED_DIR = './pre_trained'
CLASS_NAMES = ['1', '2', '3', '4', '5']
HEATMAP_DIR = os.path.join(DATA_DIR, 'spec')
EXP_DIR = os.path.join(DATA_DIR, 'expl')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_spec', required=False, action='store_true', dest='generate_spec', help='Present if spectrograms should be generated.')
    parser.add_argument('--tabular', required=False, action='store_true', dest='tabular', help='Present if tabular explainer should be used. Otherwise image explainer will be used.')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # preprocess data
    # num_test, num_valid, and batch_size values are defined in the MOSNet train/validation/test split
    (train_feats, train_labels, test_feats, _) = preprocess(num_test=4000, num_valid=3000, batch_size=64, generate_spec=args.generate_spec)
    
    # setup models and explainer
    mosnet_model, explainer = init(train_feats, train_labels, tabular=args.tabular)
    
    # explain a data instance
    expl_idx = np.random.randint(0, test_feats.shape[0])
    if args.tabular:
        explain_table(explainer, test_feats, expl_idx, mosnet_model)
    else:
        explain_image(explainer, HEATMAP_DIR, mosnet_model)
    
     
def preprocess(num_test, num_valid, batch_size, generate_spec):
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
    
    # generate spectrograms
    if generate_spec:
        generate_spec(mos_list)
    
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
     
     
def init(train_feats, train_labels, tabular=False):
    """Initializes the MOSNet model and LIME framework.

    Args:
        train_feats (numpy.ndarray): a 2d numpy array of shape [# data samples, # features]
                                     where train_feats[i,j] is the jth feature of sample i. 
        train_labels (numpy.ndarray): a 1d numpy array of shape [# data samples] where 
                                      train_labels[i] is the true label for sample i. 
        tabular (bool, optional): False if a LIME tabular explainer should be used and True
                                  if an image explainer should be used. Defaults to False.

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
    if tabular:
        explainer = LimeTabularExplainer(training_data=train_feats, training_labels=train_labels, class_names=CLASS_NAMES)
    else:
        explainer = LimeImageExplainer()
    print('Done')
    
    return mosnet, explainer


def explain_table(explainer, test_feats, test_idx, model_obj):
    print(f'Explaining test instance {test_idx}...', end='', flush=True)
    tmp = time.time()
    exp = explainer.explain_instance(test_feats[test_idx], model_obj.flatten_predict)
    elapsed = (time.time() - tmp)/60.0
    print(f'Done. Time: {elapsed} minutes.')
    exp.show_in_notebook(show_table=True, show_all=False)


def explain_image(explainer, img_dir, model_obj):
    print('Generating explanations...')
    images = os.listdir(img_dir)
    images = [os.path.join(HEATMAP_DIR, i) for i in images]
    im = transform_img_fn([images[0]])[0]
    model_obj.set_instance(images)
    exp = explainer.explain_instance(im.astype('double'), model_obj.image_predict, hide_color=0, num_samples=10)
    print('Done')
    save_expl_figs(exp)
    

def save_expl_figs(explanation):
    print('Saving explanations...', end='', flush=True)
    #Select the same class explained on the figures above.
    ind =  explanation.top_labels[0]

    #Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

    #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    plt.savefig(os.path.join(EXP_DIR, 'expl.png'))
    print('Done')


def generate_spec(file_list, num=10, start=0):
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
        plt.savefig(os.path.join(HEATMAP_DIR, f'{filename}-spec.png'))
        plt.close(fig)
        
        # open, add filename as metadata, and save
        # TODO - I/O inefficient
        #im = Image.open(os.path.join(HEATMAP_DIR, f'{filename}_spec.png'))
        #metadata = PngInfo()
        #metadata.add_text('Source sound file', f'{filename}')
        #im.save(os.path.join(HEATMAP_DIR, f'{filename}_spec.png'))
        
    
    print('Done')
        

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        out.append(x)
    return np.vstack(out)


if __name__ == '__main__':
    main()