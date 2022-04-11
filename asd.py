import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from torchvision import datasets, transforms
import torchvision.models as models
import os
import numpy as np
from random import sample
# import matplotlib.pyplot as plt
from model import AutoEncoder
path_to_dir = 'C:\\Users\\ST200423\\Desktop\\진동data 개발\\FREQUENCY\\TRAIN\\motor_x'
path_to_train = os.path.join(path_to_dir, 'motor_x.csv')
path_to_dir = 'C:\\Users\\ST200423\\Desktop\\진동data 개발\\FREQUENCY\\TEST\\motor_x'
path_to_test = os.path.join(path_to_dir, 'motor_x_.csv')

train_dataset = pd.read_csv(path_to_train)
test_dataset = pd.read_csv(path_to_test)

total_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset_ = np.array(train_dataset,dtype=float)
test_dataset = np.array(test_dataset,dtype=float)

batch_size = 1
num_epochs = 1000

train_loader = torch.utils.data.DataLoader(dataset = train_dataset_, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

AE = AutoEncoder(7971, 3985, 1000)
AE_loss = nn.MSELoss()




print(train_loader)
def train(model, Loss, optimizer, num_epochs):
    print('a')
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.

    for epoch in range(num_epochs):

        epoch_loss = 0.
        for batch_X in train_loader:
            batch_X = batch_X.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()

            # Forward Pass
            model.train()
            outputs = model(batch_X)
            train_loss = Loss(outputs, batch_X)
            epoch_loss += train_loss.data

            # Backward and optimize
            train_loss.backward()
            optimizer.step()

        train_loss_arr.append(epoch_loss / len(train_loader.dataset))

        if epoch % 10 == 0:
            model.eval()

            test_loss = 0.

            for batch_X in test_loader:
                batch_X = batch_X.type(torch.FloatTensor).to(device)

                # Forward Pass
                outputs = model(batch_X.type(torch.FloatTensor))
                batch_loss = Loss(outputs, batch_X)
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f} *'.format(epoch, num_epochs, epoch_loss,
                                                                                      test_loss))
            else:
                early_stop += 1
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, num_epochs, epoch_loss,
                                                                                    test_loss))

        if early_stop >= early_stop_max:
            break























