import warnings
warnings.filterwarnings('ignore')

import subprocess

import numpy as np
import librosa

num_mfcc = 64


def get_wav(path, start, end):
    '''extract chunk'''
    command = f"ffmpeg -loglevel quiet -y -ss {start} -to {end} -i {path} -ab 160k -ac 2 -ar 44100 -vn i.wav"

    subprocess.call(command, shell=True)


def get_acoustic(path, mfcc=True):
    '''extract MFCC'''
    x, sample_rate = librosa.load(path)
    # remove silence 
    x, _ = librosa.effects.trim(x) 
    
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
        ft=np.hstack((mfccs))

        emotion = emo.predict(ft.reshape(1, -1))
        emo_labels = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}

        sentiment = sent.predict(ft.reshape(1, -1))
        sent_labels = {0: "negative", 1: "neutral", 2: "positive"}

        return [emo_labels[emotion[0]], sent_labels[sentiment[0]]]
    else:
        return None


import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords


def word_extraction(sentence):    
    ignore = stopwords.words('english') + ["reporter:"]  
    words = re.sub("[^\w]", " ",  sentence).split()    
    
    cleaned_text = [w.lower() for w in words if w not in ignore]    
    
    return cleaned_text


import torchtext

glove = torchtext.vocab.GloVe(name='840B', dim=300)  # size = 300

size = 300

def text_embedding(text):  # list of sentences
    '''fetch glove embedding for text'''
    embedding = np.zeros(size).reshape((1, size))

    for i in text:
        text_embedding = np.zeros(size).reshape((1, size))
        count = 0

        for word in word_extraction(i):
            word_embedding = glove[word].numpy().reshape(1, -1)
            
            # check
            if not np.all(word_embedding == 0):
                text_embedding += word_embedding
                count += 1      
        
        # check for zero
        if np.all(text_embedding):
            text_embedding /= count
            
            embedding += text_embedding

    return embedding


import pickle as p

# ------ LOAD ------
import os

BASE_PATH = os.getcwd()


with open(os.path.join(BASE_PATH, "pynb/eval/emo.p"), 'rb') as f:
    emo = p.load(f)

with open(os.path.join(BASE_PATH, "pynb/eval/sent.p"), 'rb') as f:
    sent = p.load(f)

with open(os.path.join(BASE_PATH, "pynb/eval/stnc.p"), 'rb') as f:
    stnc = p.load(f)


def get_stance(embedding):
    p = stnc.predict(embedding)

    labels = {0: "con", 1: "pro", 2: "neu"}
    return labels[p[0]]


import json

with open(os.path.join(BASE_PATH, "pynb/eval/abortion.json"), 'r') as f:
    data = json.load(f)


from tqdm import tqdm

results = []
for story in tqdm(data, total=len(data)):

    trs = [t.lower().strip() for t in story["trs"].split('. ')]
    start, end = story["start"], story["end"]

    if len(start) != len(end):
        continue

    stance = get_stance(text_embedding(trs))

    get_wav(story["file_path"].split('.')[0] + ".mp4", ':'.join(start.split(':')[1 : ]), ':'.join(end.split(':')[1 : ]))

    emotion, sentiment = get_acoustic("i.wav")
    
    results.append({
        'outlet': "FNC" if "FOX-News" in story["file_path"] else "MSNBC",
        'file_path': story["file_path"],
        'trs': story["trs"],
        'start': ':'.join(start.split(':')[1 : ]),
        'end': ':'.join(end.split(':')[1 : ]),
        'stance': stance,
        'emotion': emotion,
        'sentiment': sentiment
    })


import json

with open(os.path.join(BASE_PATH, "pynb/eval/result.json"), 'w') as f:
    json.dump(results, f, indent=4)
 