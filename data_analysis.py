from nn_main import *
import matplotlib.pyplot as plt
import torch
import numpy as np

train_dataset = gamma_dataset('TrainingData_test/', 'train')
val_dataset = gamma_dataset('TrainingData_test/', 'val')

#plot the value of the train_dataest in matplotlib as hystogram
train_target = torch.Tensor.cpu(train_dataset.train_target).numpy()
train_target = train_target.squeeze(1)
plt.hist(train_target, bins=100)
#label the plot
plt.xlabel('Gamma of the target')
plt.ylabel('Number of occurences')
plt.show()


