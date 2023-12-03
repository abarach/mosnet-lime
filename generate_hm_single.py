import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import utils
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
na = np.newaxis
import librosa
import librosa.display

BASE_PATH = '/Users/adabarach/Documents/OSU/2023-2024/AU23/CSE 5539 - Khaliligarekani/mosnet-lime/'
DATA_DIR = '/Users/adabarach/Documents/OSU/2023-2024/AU23/CSE 5539 - Khaliligarekani/mosnet-lime/data'

model_path = os.path.join(BASE_PATH, 'pre_trained/cnn.h5')
txt_path = os.path.join(BASE_PATH, 'pre_trained/cnn.txt')
data_list_path = os.path.join(BASE_PATH, 'data/mos_list.txt')
out_path = os.path.join(BASE_PATH, 'output/')
heatmap_path = os.path.join(BASE_PATH, 'lime/heatmaps/examples/')

WAV_DIR = os.path.join(DATA_DIR, 'wav')
BIN_DIR = os.path.join(DATA_DIR, 'bin')
PRE_TRAINED_DIR = '/Users/adabarach/Documents/OSU/2023-2024/AU23/CSE 5539 - Khaliligarekani/mosnet-lime/pre_trained'

def main():
    # pick a soundfile
    mos_list = utils.read_list(data_list_path)
    filename = mos_list[1].split(',')[0]
    true_mos = float(mos_list[1].split(',')[1])
    
    generate_heatmaps(true_mos, filename)   

def generate_heatmaps(y, filename):
    """
    Generates a figure containing (a) the original spectrogram from the file and (b) the relevance heat map and saves to a file. 
    
    Args:
        R (np.ndarray): the computed relevance matrix
        y (np.array, size (1,1)): the ground truth MOS for the audio sample
        ypred (np.array, size(1,1)): the predicted MOS for the audio sample
        filename: the file name for the audio sample
    """
    f, (ax1) = plt.subplots(1, 1, figsize=(14, 9))
    
    # render original heatmap
    spec, sr = utils.get_spectrograms(os.path.join(WAV_DIR,filename))
    spec = np.transpose(spec)       # utils transposes before returning
    p1 = librosa.display.specshow(librosa.amplitude_to_db(spec,ref=np.max),y_axis='linear', x_axis='time', sr=sr, ax=ax1)
    ax1.set_title('Input Spectrogram\nMOS: {}'.format(y))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency (Hz)')
    
    # save heatmap
    f.colorbar(p1, ax=ax1, format="%+2.f dB")
    plt.savefig(os.path.join(heatmap_path, '{}_spec.png'.format(filename)))
    plt.close(f)
 
def mask_inliers(R, m=6.):
    """
    Given a relevance matrix of shape (H, W), calculates the upper and lower threshold for outliers. 
    
    Args:
        R (np.ndarray): the (H, W) shaped relevance matrix to calculate outliers from.
        m (float): the number of standard deviations from the median outside of which data is considered outliers.
    
    Returns:
        np.ndarray: R where the outliers have been masked to np.nan. 
    """
    # normally: np.abs(R-median(d)) but we want to center around 0 not median
    d = np.abs(R)
    med_dev = np.median(d)
    stdev = d/med_dev if med_dev else np.zeros(len(med_dev))
    return np.where(stdev < m, np.nan, R)

if __name__ == '__main__':
    main()