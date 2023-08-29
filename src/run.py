import sys
import os
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings('ignore')

import subprocess
import argparse

import json
import torch
import whisper

from utils import *


BASE_PATH = '/'.join(os.path.abspath('.').split('/')[ : -1])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model('base').to(device)


def exit_program():
    print("aborted!")
    sys.exit(0)


def main(data, f_name):
    '''iterate transcript'''
    for i, phrase in enumerate(data):
        # get chunk
        command = f"ffmpeg -loglevel quiet -y -ss {str(phrase['start'])} -to {str(phrase['end'])} -i {os.path.join(BASE_PATH, 'news', f_name)} -c copy {os.path.join(BASE_PATH, 'src/i.mp4')}"
        subprocess.call(command, shell=True)
        
        # audio
        ac_emo = extract_acoustic(os.path.join(BASE_PATH, "src/i.mp4"))
        
        # sequence over frames
        vi_emo = extract_visual(os.path.join(BASE_PATH, "src/i.mp4"), skip=8)

        data[i]["emotion"] = ac_emo
        
        data[i]["sequence"] = vi_emo
        
        # stance towards nouns
        if not len(phrase['nouns']):
            stance = [get_stance(phrase['text'], noun) for noun in phrase['nouns']]
        else: stance = []

        data[i]["stance"] = stance
    
    with open(os.path.join(BASE_PATH, f"news/json/{f_name.split('.')[0]}.json"), 'w') as f:
        json.dump(data, f, indent=4)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--file_name", help = "file name from news folder")
    
    args = parser.parse_args()
    
    if args.file_name:
        print(f"\nloading video: {args.file_name}")

        if args.file_name.split(".")[-1] != "mp4":
            print("\nfile format should be MP4")
            exit_program()
        
        # fetch transcription
        print("\nfetching transcription .. ")
        text = model.transcribe(os.path.join(BASE_PATH, "news", args.file_name), verbose=False)

        print("\nparsing transcription .. ")
        data = get_sentences(text["segments"])

        print("\niterating over chunks .. ")
        main(data, args.file_name)

        print("\nready!")
