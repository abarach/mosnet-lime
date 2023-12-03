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

data_list_path = os.path.join(BASE_PATH, 'data/mos_list.txt')
ex_path = os.path.join(BASE_PATH, 'lime/heatmaps/examples/')

WAV_DIR = os.path.join(DATA_DIR, 'wav')
FS = 16000


def main():
    # pick a soundfile
    mos_list = utils.read_list(data_list_path)
    filename = mos_list[1].split(',')[0]
    true_mos = float(mos_list[1].split(',')[1])
    
    generate_waveform(true_mos, filename)
    generate_mag_spec(true_mos, filename)   
    
    
def generate_waveform(y, filename):
    """Generates a figure containing the waveform of the input file.

    Args:
        y (np.ndarray, size (1,1)): the ground truth MOS for the audio sample
        filename (str): the file name for the audio sample
    """
    f, (ax1) = plt.subplots(1,1, figsize=(14,9))
    
    # render waveform
    signal, sr = librosa.load(os.path.join(WAV_DIR,filename), sr=FS)
    p1 = librosa.display.waveshow(signal, sr=FS, ax=ax1)
    ax1.set_title('Input Waveform\nMOS: {}'.format(y))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    
    # save waveform
    plt.savefig(os.path.join(ex_path, '{}_wav.png'.format(filename)))
    plt.close(f)
    

def generate_mag_spec(y, filename):
    """
    Generates a figure containing the magnitude spectrogram from the file. 
    
    Args:
        y (np.array, size (1,1)): the ground truth MOS for the audio sample
        filename (str): the file name for the audio sample
    """
    f, (ax1) = plt.subplots(1, 1, figsize=(14, 9))
    
    # render spectrogram
    spec, sr = utils.get_spectrograms(os.path.join(WAV_DIR,filename))
    spec = np.transpose(spec)       # utils transposes before returning
    p1 = librosa.display.specshow(librosa.amplitude_to_db(spec,ref=np.max),y_axis='linear', x_axis='time', sr=sr, ax=ax1)
    ax1.set_title('Input Spectrogram\nMOS: {}'.format(y))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency (Hz)')
    
    # save spectrogram
    f.colorbar(p1, ax=ax1, format="%+2.f dB")
    plt.savefig(os.path.join(ex_path, '{}_spec.png'.format(filename)))
    plt.close(f)


if __name__ == '__main__':
    main()