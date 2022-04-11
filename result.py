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
import matplotlib.pyplot as plt
from asd import test_loader,AE,device,AE_loss
score = []


for X in test_loader:
    X = X.type(torch.FloatTensor).to(device)


    # Forward Pass
    AE_output = AE(X)
    result= AE_loss(AE_output, X).item()
    score.append(result)


print(score)



fig = plt.figure(figsize=(15, 3))
x2 = score
plt.xlabel('x10s')
plt.ylabel('score')
plt.axvline(x = 104, color='lightgray',  linestyle='--')
plt.text(55,3.9,'2021_02_01',rotation=45)
plt.axvline(x = 248, color='lightgray',  linestyle='--')
plt.text(180,3.9,'20210202_10_50',rotation=45)
plt.axvline(x = 432, color='lightgray',  linestyle='--')
plt.text(372,3.9,'20210202_11_20',rotation=45)
plt.axvline(x = 612, color='lightgray',  linestyle='--')
plt.text(562,3.9,'20210202_11_50',rotation=45)
plt.axvline(x = 792, color='lightgray',  linestyle='--')
plt.text(732,3.9,'20210202_12_20',rotation=45)
plt.axvline(x = 972, color='lightgray',  linestyle='--')
plt.text(912,3.9,'20210202_12_50',rotation=45)
plt.axvline(x = 1152, color='lightgray',  linestyle='--')
plt.text(1090,3.9,'20210202_13_20',rotation=45)
plt.axvline(x = 1020, color='r',  linestyle='--',Alpha = 0.5)
plt.text(1020,1.3,'20210202_13_09')
plt.axvline(x = 132, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 192, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 312, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 372, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 492, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 552, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 672, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 732, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 852, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 912, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 1032, color='lightgray',  linestyle='-', label='a')
plt.axvline(x = 1092, color='lightgray',  linestyle='-', label='a')
plt.plot(x2)
plt.show()