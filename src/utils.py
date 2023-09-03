import os
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings('ignore')

# ------ IMPORTS ------
import subprocess
import pickle as p

import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag

import yake
import librosa


BASE_PATH = '/'.join(os.path.abspath('.').split('/')[ : -1])


# ------ LOAD ------
with open(os.path.join(BASE_PATH, "pynb/w8/svc.p"), 'rb') as f:
    emo = p.load(f)

with open(os.path.join(BASE_PATH, "pynb/w9/svc.p"), 'rb') as f:
    sent = p.load(f)

with open(os.path.join(BASE_PATH, "pynb/w10-11/lr.p"), 'rb') as f:
    stnc = p.load(f)


# ------ ACOUSTIC ------
def convert_wav(path):
    command = f"ffmpeg -loglevel quiet -y -i {path} -acodec pcm_s16le __.wav" 

    subprocess.call(command, shell=True)


def get_meta(sentence):
    '''
    fetch keywords and nouns
    '''
    extractor = yake.KeywordExtractor()
    words = extractor.extract_keywords(sentence)

    words = [i[0] for i in words if i[1] > 0.05 and len(i[0].split()) == 1]

    tagged_senten = pos_tag(sentence.split())

    proper_nouns = [word for word, pos in tagged_senten if pos == 'NNP']

    return [words, proper_nouns]


def get_sentences(segments):
    '''merge segments to sentences'''
    sentences, temp = [], ""

    for _, i  in enumerate(segments):
        # with period
        if not len(temp) and "." in i['text']:

            words, nouns = get_meta(i['text'].strip())

            sentences.append({
                'text': i['text'].strip(),
                'start': round(i['start'], 2),
                'end': round(i['end'], 2)
            }) 

            continue
        # first condition fails
        elif not len(temp):

            temp, start = i['text'], round(i['start'], 2)

            continue

        temp += i['text']

        if "." in i['text']:

            words, nouns = get_meta(temp.strip())

            sentences.append({
                'text': temp.strip(),
                'start': start,
                'end': round(i['end'], 2)
            })
            temp = ""

    return sentences


def extract_acoustic(path, mfcc=True, num_mfcc=64):
    '''extract MFCC'''
    X, sample_rate = librosa.load(path)
    # remove silence 
    X, _ = librosa.effects.trim(X) 

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
        ft = np.hstack((mfccs)).reshape(1, -1)

        emo_label = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}

        sent_label = {0: "negative", 1: "neutral", 2: "positive"}

        return emo_label[emo.predict(ft)[0]], sent_label[sent.predict(ft)[0]]
    else:
        return None


# ------ VISUAL ------
from feat import Detector

detector = Detector()

def extract_visual(path, skip=2):
    '''extract facial features'''
    cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    df = detector.detect_video(path, skip_frames=skip)

    emotion = []

    for i in df.index:
        x = {}
        for col in cols:  # parse string
            x[col] = str(round(df[col][i], 4)).split('    ')[-1].split("\n")[0] if len(str(df[col][i]).split('    ')) else str(round(df[col][i], 4))

        emotion.append(x)

    return emotion


# ------ TEXT ------
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


def get_stance(embedding):
    '''predict stance for text'''
    label = {0: "right", 1: "center", 2: "left"}

    return label[stnc.predict(embedding)[0]]
