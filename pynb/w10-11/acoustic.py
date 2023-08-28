import torch
import torch.nn as nn

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel
)

model_card = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'


class RegressionHead(nn.Module):
    '''
    classification head
    '''
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)

        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, features):
        x = features

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)

        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    '''
    speech emotion classifier
    '''
    def __init__(self, config, logits=True):
        super().__init__(config)

        self.config = config
        self.logits = logits
        
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)

        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values.reshape(1, 16000))

        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)

        logits = self.classifier(hidden_states)

        return hidden_states, logits


device = 'cpu'

processor = Wav2Vec2Processor.from_pretrained(model_card)
model = EmotionModel.from_pretrained(model_card)

def process_func(x, sampling_rate, embedding=False):
    '''
    predict emotions or extract embeddings
    '''
    y = processor(x, sampling_rate=sampling_rate)
    y = y["input_values"][0]
    y = torch.from_numpy(y).to(device)

    # run model
    with torch.no_grad():
        y = model(y)[0 if embedding else 1]

    # to array
    y = y.detach().cpu().numpy()

    return y      


import numpy as np

# dummy
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)

# process_func(signal, sampling_rate)
