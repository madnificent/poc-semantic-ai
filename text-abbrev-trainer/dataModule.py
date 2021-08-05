import torch
import pandas as pd

from torch.utils.data import Dataset

from constants import *

def encode_index_for_char(char):
  try:
    return symbols.index(char)
  except:
    return symbols.index('?')

def encode_string(string, pad_to=False):
  encoded_tensor = [encode_index_for_char(char) for char in string]
  if pad_to:
    pad = [symbols.index("PAD")] * (pad_to - len(string))
    return torch.tensor(encoded_tensor + pad)
  else:
    return torch.tensor(encoded_tensor)

def decode_array(arr, strip_padding=True):
  return "".join([symbols[idx]
                  for idx in arr
                  if not strip_padding or idx != symbols.index("PAD")])


class AbbrevDataset(Dataset):
    def __init__(self, data):
        df = pd.DataFrame(data)

        df["long"] = df["long"].apply( lambda x: encode_string(x, pad_to=MAX_LEN) )
        df["short"] = df["short"].apply( lambda x: encode_string(x, pad_to=MAX_OUT_LEN) )

        self._df = df

    def __len__(self):
        return self._df.shape[0];

    def __getitem__(self, index):
        (longForm, shortForm) = self._df.iloc[index]
        return longForm, shortForm
