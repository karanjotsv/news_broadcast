import os
import librosa
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

BASE_PATH = '/'.join(os.getcwd().split('/')[ : -2])

DATA_PATH = "gallina/ryerson/audio"
FEATURE_PATH = "feature/"

folders = ["mel", "emobase", "stack"]

num_mfcc = 64


for path in folders:
    f = os.path.join(BASE_PATH, "pynb/w8", FEATURE_PATH, path)
    
    if not os.path.exists(f):
        os.makedirs(f)


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
    command = f"ffmpeg -y -loglevel quiet -i {path} i.wav"  # -acodec pcm_s16le

    subprocess.call(command, shell=True)


# ---- MAIN ----
path = os.path.join(BASE_PATH, DATA_PATH)

dirs = os.listdir(path)
print(f"directories: {dirs}")

chunk_size = 4

for i in range(len(dirs) // chunk_size):
    # iterate chunks
    feat_1, feat_2, feat_3, label = [], [], [], []  # mfcc, emobase, stack

    for dir in os.listdir(path)[i * chunk_size : (i + 1) * chunk_size]:
        # iterate files
        for j, f in tqdm(enumerate(os.listdir(os.path.join(path, dir))), desc=f"chunk: {i}"):
            
            feat_1.append(extract_mfcc(os.path.join(path, dir, f)).tolist())
            # emobase
            feat_2.append(extract_acoustic(os.path.join(path, dir, f)))
    
            feat_3.append(extract_librosa_features(os.path.join(path, dir, f)).tolist())

            # fetch label
            label.append(int(f.split('-')[2]))

            # print(f"chunk: {i} {dir}: {j}")

    for feat, f in zip([feat_1, feat_2, feat_3], folders):
        np.save(os.path.join(BASE_PATH, "pynb/w8", FEATURE_PATH, f"{f}/{i}"), np.array(feat))
    
    np.save(f"{FEATURE_PATH}label_{i}", np.array(label))
