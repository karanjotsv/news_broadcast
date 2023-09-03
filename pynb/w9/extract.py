import os
import csv
import librosa
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

BASE_PATH = '/'.join(os.getcwd().split('/')[ : -2])

DATA_PATH = "gallina/mosei"
FEATURE_PATH = "feature/"

f_names = ["mel", "emobase", "stack"]

num_mfcc = 64


# ---- DATA ----
with open(os.path.join(BASE_PATH, DATA_PATH, "label.csv"), mode ='r')as f:

    reader = csv.reader(f)
    header = next(reader)
 
    rows = [line for line in reader]


def get_label(rows, f, i):
    '''sentiment label for given id'''

    labels = {"Negative": 0, "Neutral": 1, "Positive": 2}

    sub_list = [x for x, row in enumerate(rows) if f in row]
    id = [x for x, row in enumerate(rows[sub_list[0] : sub_list[-1] + 1]) if i in row]

    return(labels[rows[sub_list[id[0]]][4]])


# ---- ACOUSTIC ----
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,  # ComParE - 6000+
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_acoustic(path):
    '''extract acoustic features'''
    feat = smile.process_file(path).values.tolist()[0]

    return feat


def extract_mfcc(path, mfcc=True):
    '''extract MFCC'''
    x, sample_rate = librosa.load(path)
    # remove silence 
    x, _ = librosa.effects.trim(x) 
    
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
        result=np.hstack((mfccs))

        return result
    else:
        return None
    

def extract_librosa_features(path):
    '''extract stack of features'''
    x, sample_rate = librosa.load(path)

    hop_length = 512  # set hop length; at 22050 Hz, 512 samples ~= 23ms

    # remove vocals
    d = librosa.stft(x, hop_length=hop_length)
    s_full, phase = librosa.magphase(d)

    s_filter = librosa.decompose.nn_filter(s_full, aggregate=np.median, metric='cosine',
                                           width=int(librosa.time_to_frames(0.2, sr=sample_rate)))

    s_filter = np.minimum(s_full, s_filter)

    power, margin_v = 2, 4

    mask_v = librosa.util.softmask(s_full - s_filter, margin_v * s_filter, power=power)
    s_foreground = mask_v * s_full

    # recreate vocal_removal y
    new_d = s_foreground * phase
    x = librosa.istft(new_d)

    mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13)  # MFCC features from raw
    mfcc_delta = librosa.feature.delta(mfcc)  # first-order differences (delta features)

    s = librosa.feature.melspectrogram(y=x, sr=sample_rate, n_mels=128, fmax=8000)
    s_delta = librosa.feature.delta(s)

    spectral_centroid = librosa.feature.spectral_centroid(S=s_full)

    audio_feature = np.vstack((mfcc, mfcc_delta, s, s_delta, spectral_centroid))  # combine features

    # binning data
    jump = int(audio_feature.shape[1] / 10)
    return np.mean(librosa.util.sync(audio_feature, range(1, audio_feature.shape[1], jump)).T, axis=0)  # mean

    
import subprocess

def convert_wav(path):
    command = f"ffmpeg -y -loglevel quiet -i {path} -ab 160k -ac 2 -ar 44100 -vn i.wav"

    subprocess.call(command, shell=True)


# ---- MAIN ----
feat_1, feat_2, feat_3, label = [], [], [], []  # mfcc, emobase, stack

# iterate chunks
for chunk in ["005", "006", "007", "008", "009", "010", "011", "012"]:
    total = len(os.listdir(os.path.join(BASE_PATH, DATA_PATH, "data", chunk, "raw")))
    # iterate folders
    for dir in tqdm(os.listdir(os.path.join(BASE_PATH, DATA_PATH, "data", chunk, "raw")), desc=chunk, total=total):
        dir_path = os.path.join(BASE_PATH, DATA_PATH, "data", chunk, "raw", dir)
        # iterate files
        for i in os.listdir(dir_path):
            try:
                convert_wav(os.path.join(dir_path, i))
                    
                ft_1 = extract_mfcc("i.wav").tolist()
                # emobase
                ft_2 = extract_acoustic("i.wav")

                ft_3 = extract_librosa_features("i.wav").tolist()

                feat_1.append(ft_1); feat_2.append(ft_2); feat_3.append(ft_3)

                # fetch label            
                label.append(get_label(rows, dir, i.split('.')[0]))

            except Exception as e: pass

for feat, f in zip([feat_1, feat_2, feat_3], f_names):
    np.save(os.path.join(BASE_PATH, "pynb/w9", FEATURE_PATH, f"{f}"), np.array(feat))

np.save(os.path.join(BASE_PATH, "pynb/w9", FEATURE_PATH, "label"), np.array(label))
