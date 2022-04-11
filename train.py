from asd import device
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from torchvision import datasets, transforms
import torchvision.models as models
import os
import numpy as np
from asd import AE,train_loader,train,num_epochs

AE_loss = nn.MSELoss()


AE = AE.to(device)

learning_rate = 0.35

AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
print(train_loader)
train(AE, AE_loss, AE_optimizer, num_epochs)

