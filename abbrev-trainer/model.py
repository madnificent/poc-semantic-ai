import torch
import torch.nn as nn
from constants import *
from transformers import AutoModel

class Abbreviator(nn.Module):

    def __init__(self):
        super().__init__()

        self.abbrnet = nn.Sequential(
            nn.Linear(FIRST_LAYER_SIZE, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, LAST_LAYER_SIZE),
            nn.Sigmoid())

    def forward(self, input_seq):
        return self.abbrnet(input_seq)
        
