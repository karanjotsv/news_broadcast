### Multimodal News Broadcast Analysis
Summer of code '23 at Red Hen Lab. The work focuses on recognition of topics, emotion, valence and stance consuming all three modalities across a news story. The seven basic emotions(neutral, anger, disgust, fear, happy, sad and surprise) are considered for sentence-based chunks, shift of emotions is also recorded across frames of a chunk. 

<b>Mentors:</b> Dr. Homa Hosseinmardi, Dr. Francis Steen

[Blog](https://karanjotsinghv.notion.site/karanjotsinghv/Summer-of-Code-23-0e0582afd9294f379e3e4cc1d96b8085)

#### steps to run:
1. clone this repository to your local
```bash
git clone https://github.com/karanjotsv/news_broadcast.git
```
2. install ffmpeg(for ubuntu) and whisper
```bash
sudo apt install ffmpeg
pip install -q git+https://github.com/openai/whisper.git
``` 
3. install required python modules
```bash
pip install -r requirements.txt
```
5. change current directory to news_broadcast/src
```bash
cd news_broadcast/src
```
news_broadcast/news contains sample news stories on 'gun control', additional video files(MP4) to be added here

6. run.py takes 1 argument i.e. file name of the input video in news_broadcast/news folder
```python
python3 run.py -i de_fnc.mp4 
```
7. output JSON file is saved in news_broadcast/news/json as de_fnc.json(same as input video)
