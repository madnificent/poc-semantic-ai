import torch
import torch.nn as nn
from constants import *
from transformers import AutoModel

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode_embedding = nn.Sequential(
            nn.Embedding(len(SYMBOLS), 128),
            nn.Dropout(0.05)
        )

        self.encode_lstm = nn.LSTM(
            128, 256,
            3, # lstm layers
            dropout=0.05
        )

    def forward(self, content):
        embedding = self.encode_embedding(content)
        response, (hidden, cell) = self.encode_lstm( embedding )
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode_embedding = nn.Sequential(
            nn.Embedding(len(SYMBOLS), 128),
            nn.Dropout(0.05)
        )

        self.decode_lstm = nn.LSTM(
            128, 256, 3, dropout=0.05            
        )

        self.decode_predict = nn.Linear(256,len(SYMBOLS))

    def forward(self, previous_token, hidden, cell):
        embedding = self.encode_embedding( previous_token.unsqueeze(0) )
        response, (hidden, cell) = self.decode_lstm( embedding, (hidden, cell) )
        predictions = self.decode_predict(response).squeeze(0)

        return predictions, hidden, cell


class Abbreviator(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(SYMBOLS)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        hidden, cell = self.encoder(source)

        current_character = target[0]

        for idx in range(1, target_len):
            output, hidden, cell = self.decoder(current_character, hidden, cell)
            outputs[idx] = output
            prediction = output.argmax(1)
            current_character = target[idx] if random.random() < teacher_force_ratio else best_guess
        
        return outputs
