import torch
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset

from constants import *

def encode_index_for_char(char):
  try:
    return SYMBOLS.index(char)
  except:
    return SYMBOLS.index('?')

def encode_string(string, pad_to=False):
  encoded_tensor = [encode_index_for_char(char) for char in string]
  if pad_to:
    pad = [SYMBOLS.index("PAD")] * (pad_to - len(string))
    return torch.tensor(encoded_tensor + pad)
  else:
    return torch.tensor(encoded_tensor)

def decode_array(arr, strip_padding=True):
  return "".join([SYMBOLS[idx]
                  for idx in arr
                  if not strip_padding or idx != SYMBOLS.index("PAD")])


def string_to_one_hot(string, padding, num_classes=len(SYMBOLS), dtype=torch.float32):
  encoded = encode_string(string, pad_to=padding)
  expanded = F.one_hot(encoded, num_classes=num_classes)
  return torch.tensor(expanded.view(-1), dtype=dtype)


class AbbrevDataset(Dataset):
    def __init__(self, data):
        df = pd.DataFrame(data)

        df["long"] = df["long"].apply( lambda x: string_to_one_hot( x, MAX_LEN ) )
        df["short"] = df["short"].apply( lambda x: string_to_one_hot( x, MAX_OUT_LEN, dtype=torch.long ) )

        self._df = df

    def __len__(self):
        return self._df.shape[0];

    def __getitem__(self, index):
        (longForm, shortForm) = self._df.iloc[index]
        return longForm, shortForm
