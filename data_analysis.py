from nn_dataloader import *
import matplotlib.pyplot as plt
import torch
import numpy as np

train_dataset = gamma_dataset('TrainingData_red/', 'train')
val_dataset = gamma_dataset('TrainingData_red/', 'val')

#plot the value of the train_dataest in matplotlib as hystogram
train_target = torch.Tensor.cpu(train_dataset.train_target).numpy()
train_target = train_target.squeeze(1)
plt.hist(train_target, bins=100)
#label the plot
plt.xlabel('Gamma of the target')
plt.ylabel('Number of occurences')
plt.show()

#plot the value of the val_dataset in matplotlib as hystogram
val_target = torch.Tensor.cpu(val_dataset.train_target).numpy()
val_target = val_target.squeeze(1)
plt.hist(val_target, bins=100)
#label the plot
plt.xlabel('Gamma of the target')
plt.ylabel('Number of occurences')
plt.show()



