import io
import json
import os
from string import Template
import torch
import torch.nn.functional as F
import numpy as np

from flask import request, jsonify

from helpers import log
from transformers import AutoTokenizer

from . constants import *

from . file_handler import postfile, get_file_by_id
from . preprocessor import Preprocessor
from . model import Abbreviator
from . utils import seed_everything

import sys


print("Hello from predictor!")

# set path for torch load model dir
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(here)

seed_everything()

def get_model(file_id):
    try:
        phys_file = get_file_by_id(file_id)
        model_file = phys_file["results"]["bindings"][0]["uri"]["value"].replace("share://", "/share/")

        log("Loading from: " + str(model_file))
        model = Abbreviator()
        weights = torch.load(model_file)
        model.load_state_dict( weights )
        log("Managed to load file")
        log(model)
        return model
    except Exception as e:
        log(e)
        return None


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

def string_to_one_hot(string, padding, num_classes=len(SYMBOLS), dtype=torch.float32):
  encoded = encode_string(string, pad_to=padding)
  log("Made encoded string")
  expanded = F.one_hot(encoded, num_classes=num_classes)
  log("Made expanded string")
  log( expanded )
  float_expanded = torch.tensor(expanded.view(-1), dtype=dtype)
  log("Made float expanded")
  return float_expanded

def decode_array(arr, strip_padding=True):
  return "".join([SYMBOLS[idx]
                  for idx in arr
                  if not strip_padding or idx != SYMBOLS.index("PAD")])





@app.route("/abbreviate", methods=["GET"])
def query_data():
    """
    Endpoint for loading data from triple store using a query file and converting it to json
    Accepted request arguments:
        - filename: filename that contains the query
        - limit: limit the amount of data retrieved per query execution, allows for possible pagination
        - global_limit: total amount of items to be retrieved
    :return: response from storing data in triple store, contains virtual file id and uri
    """

    print( "Yeah, I'll abbreviate that for you" )

    # env arguments to restrict option usage
    try:
        torch.set_printoptions(threshold=10_000)
        text = request.args.get("text")
        model_file = request.args.get("model")

        if not (text and model_file):
            return "Missing argument", 400

        model = get_model(model_file)
        if not model:
            return f"Unable to load model from file with id {model_file}", 400
        # model.eval()

        log(text)

        log( "Will convert string to vector" )

        vector = string_to_one_hot(text, MAX_LEN)

        log( "Will convert vector" )
        # print( "Current shape is:", vector.shape )
        vector = vector.view(1,-1)
        log( vector )

        # print( "Vector has shape: ", vector.shape )

        log("Ready to predict")
        predictions = model(vector)
        log("Got a prediction!")
        log(predictions)
        # log(predications.shape)
        # predictions = predictions.data.cpu().numpy()[0]

        # change the view of NP arrays to extract numbers
        split_predictions = predictions.view(MAX_OUT_LEN,len(SYMBOLS))
        log("Predictions")
        log(split_predictions)
        char_indexes = torch.argmax( split_predictions, dim=1 )
        log(char_indexes)

        result_string = decode_array( char_indexes )
        log( result_string )

        return result_string, 200
    except Exception as e:
        return str(e), 400


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """
    Default endpoint/ catch all
    :param path: requested path
    :return: debug information
    """
    return 'You want path: %s' % path, 404


if __name__ == '__main__':
    debug = os.environ.get('MODE') == "development"
    app.run(debug=debug, host='0.0.0.0', port=80)
