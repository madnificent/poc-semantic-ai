import torch

chars_lower = [ chr(code) for code in range(ord('a'),ord('z')+1)]
chars_upper = [ chr(code) for code in range(ord('A'),ord('Z')+1)]
chars_special = [ code for code in " -_." ]
code_special = [ "?", "<BEG>", "<END>", "PAD" ]

SYMBOLS = chars_lower + chars_upper + chars_special + code_special

MAX_LEN=27
FIRST_LAYER_SIZE=MAX_LEN * len(SYMBOLS)

MAX_OUT_LEN=22
LAST_LAYER_SIZE=MAX_OUT_LEN * len(SYMBOLS)

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
TRAIN_FILE_PATH = './data/labeled-2.csv'
MODEL_FILE_PATH = '/share/model/predicate-model.pth'
MODEL_OVERWRITE = False
BATCH_SIZE = 2
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8
MAX_CLASS_SIZE = 20  # float("inf") for all
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
