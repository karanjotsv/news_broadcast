import cfg

import os
import cv2
import librosa
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def get_duration(path):
    '''get meta data'''
    cap = cv2.VideoCapture(path)

    # number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    a = frames / fps

    # print(f"duration: {a} sec", f"frames: {frames}")
    return a, frames


from feat import Detector

detector = Detector()

not_cols = ['frame', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight', 'FaceScore', 'input', 'frame', 'approx_time']

cols = ['Pitch', 'Roll', 'Yaw', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU11', 
        'AU12', 'AU14', 'AU15','AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43']

def extract_visual(path, skip=2):
    '''extract facial features'''
    video_prediction = detector.detect_video(path, skip_frames=skip)

    # df = video_prediction.drop(cols, axis=1)
    df = video_prediction[cols]
    
    return list(df.mean(axis=0).values)  # df.sum(axis=0)


import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,  # emobase
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_acoustic(path):
    '''extract acoustic features'''
    feat = smile.process_file(path).values.tolist()[0]

    return feat


def extract_mfcc(path, mfcc=True):
    '''extract MFCC'''
    X, sample_rate = librosa.load(path)
    # remove silence
    X, _ = librosa.effects.trim(X) 
    
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=64).T, axis=0)
        result=np.hstack((mfccs))

        return result
    else:
        return None
    

import subprocess

def convert_wav(path):
    command = f"ffmpeg -y -loglevel quiet -i {path} -acodec pcm_s16le -ar 16000 __.wav"

    subprocess.call(command, shell=True)


num_dir = len(os.listdir(cfg.RYERSON_PATH))
print(f"directories: {os.listdir(cfg.RYERSON_PATH)}")

chunk_size = 4

for i in range(num_dir // chunk_size):
    # iterate over chunks
    aco_feat, vis_feat, label = [], [], []

    for dir in os.listdir(cfg.RYERSON_PATH)[i * chunk_size : (i + 1) * chunk_size]:
        # iterate over files
        for j, f in enumerate(os.listdir(os.path.join(cfg.RYERSON_PATH, dir))):
            # facial
            vis_feat.append(extract_visual(os.path.join(cfg.RYERSON_PATH, dir, f), skip=8))

            # acoustic
            convert_wav(os.path.join(cfg.RYERSON_PATH, dir, f))
            aco_feat.append(extract_mfcc("__.wav"))

            # fetch label
            label.append(int(f.split('-')[2]))

            print(f"chunk: {i} {dir}: {j}", f"visual: {len(vis_feat[-1])}", f"acoustic: {len(aco_feat[-1])}")

    np.save(f"{cfg.FEATURES_PATH}visual_{i}", np.array(vis_feat))
    np.save(f"{cfg.FEATURES_PATH}acoustic_{i}", np.array(aco_feat))

    np.save(f"{cfg.FEATURES_PATH}label_{i}", np.array(label))
