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
with open(os.path.join(BASE_PATH, "pynb/w9/svc.p"), 'rb') as f:
    svc = p.load(f)

with open(os.path.join(BASE_PATH, "pynb/w10-11/lr.p"), 'rb') as f:
    vectorizer, lr = p.load(f)


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
                'end': round(i['end'], 2),
                'words': words,
                'nouns': nouns
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
                'end': round(i['end'], 2),
                'words': words,
                'nouns': nouns
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

        p = svc.predict(ft)

        labels = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}

        return labels[p[0]]
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
def get_stance(text, target):
    '''stance for a noun'''
    text, target = vectorizer.transform([text]), vectorizer.transform([target])
    X = np.concatenate((text.toarray(), target.toarray()), axis=1)

    p = lr.predict(X)

    labels = {0: "con", 1: "pro", 2: "neu"}

    return labels[p[0]]
