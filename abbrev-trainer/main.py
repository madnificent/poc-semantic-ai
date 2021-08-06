import io
import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from model import Abbreviator
from constants import *
from dataModule import AbbrevDataset
from preprocessor import Preprocessor
from utils import seed_everything
from datetime import datetime
import os

seed_everything(24)

def train(data):
    train_dataset = AbbrevDataset(data)

    model = Abbreviator()
    model.to(DEVICE)

    validation_split = 0.2
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True

    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

    print('Training Set Size {}, Validation Set Size {}'.format(len(train_indices), len(val_indices)))

    # loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    # loss_fn = nn.MultiLabelMarginLoss().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters())

    model.zero_grad()
    training_acc_list, validation_acc_list = [], []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        train_correct_total = 0
        # Training Loop
        # print( "Train loop %s", epoch )
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            model.train(True)

            inputs = batch[0]
            labels = batch[1].to(DEVICE)

            # print( "inputs: %s", inputs)
            # print( "labels: %s", labels)

            logits = model(inputs)

            # print( "calculate logits" )
            # print( "logits: %s", logits)
            # print( "labels: %s", labels)
            # print( "logits shape: %s", logits.shape)
            # print( "labels shape: %s", labels.shape)

            # We should learn to take the cost into account in which we
            # only care about the "cost" of the effect.
            
            flattened_logits = torch.cat(logits).view(-1,len(SYMBOLS)) # torch.flatten(logits)
            flattened_labels = torch.flatten(labels)

            # print("doing loss calculation")

            loss = loss_fn(flattened_logits, flattened_labels)

            # print( "loss" )
            loss.backward()

            # print( "add-epoch-loss" )
            epoch_loss += loss.item()

            # print( "if accumulation step" )
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print( "Oh, doing a prediction, maybe?" )
            # _, predicted = torch.max(logits.data, 1)
            # print( "Something about correct views?" )
            # correct_reviews_in_batch = (predicted == labels).sum().item()
            # print( "Calculating correct total, maybe?" )
            # train_correct_total += correct_reviews_in_batch

        print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

        # Validation Loop
        # with torch.no_grad():
        #     val_correct_total = 0

        #     model.train(False)
        #     val_iterator = tqdm(val_loader, desc="Validation Iteration")
        #     for step, batch in enumerate(val_iterator):
        #         inputs = batch[0]

        #         labels = batch[1].to(DEVICE)
        #         logits = model(inputs)

        #         _, predicted = torch.max(logits.data, 1)
        #         correct_reviews_in_batch = (predicted == labels).sum().item()
        #         val_correct_total += correct_reviews_in_batch

        #     training_acc_list.append(train_correct_total * 100 / len(train_indices))
        #     validation_acc_list.append(val_correct_total * 100 / len(val_indices))
        #     print('Training Accuracy {:.4f} - Validation Accurracy {:.4f}'.format(
        #         train_correct_total * 100 / len(train_indices), val_correct_total * 100 / len(val_indices)))

        torch.save(model.state_dict(), MODEL_FILE_PATH[:-4] + f"_epoch-{epoch}" + ".pth")

    torch.save(model.state_dict(), MODEL_FILE_PATH)

    return MODEL_FILE_PATH
