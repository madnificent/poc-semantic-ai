import torch
import torch.nn as nn
from . constants import *
from transformers import AutoModel

class Abbreviator(nn.Module):

    def __init__(self):
        super().__init__()

        self.prediction_layers = [ nn.Linear( 1024, len(SYMBOLS) ) for _ in range(MAX_OUT_LEN) ]

        self.abbrnet = nn.Sequential(
            nn.Linear(FIRST_LAYER_SIZE, 2048),
            nn.Sigmoid(),
            # nn.Linear(2048, 4096),
            # nn.Sigmoid(),
            # nn.Dropout(p=0.05),
            # nn.Linear(4096, 4096),
            # nn.Sigmoid(),
            # nn.Linear(4096, 2048),
            # nn.Sigmoid(),
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            # nn.Linear(1024, LAST_LAYER_SIZE)
            nn.Linear(1024,1024)
        )

    def forward(self, input_seq):
        abbr_pred = self.abbrnet(input_seq)
        return [ layer(abbr_pred) for layer in self.prediction_layers ]
