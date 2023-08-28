import warnings
warnings.filterwarnings('ignore')

import json

import torch
import whisper

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import yake
from nltk.tag import pos_tag


device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large").to(device)


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
    

# merge to sentences
def get_sentences(segments):
    '''
    merge segments to sentences
    '''
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


text = model.transcribe(cfg.path)

sentences = get_sentences(text["segments"])

with open(f"news/json/{cfg.path.split('/')[-1].split()[0]}.json", 'w') as f:
    json.dump(sentences, f, indent=4)
