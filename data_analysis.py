from nn_dataloader import *
import matplotlib.pyplot as plt
import torch
import numpy as np

data_path = 'exp_D/'

train_dataset = gamma_dataset(data_path, 'train')
val_dataset_1 = gamma_dataset(data_path, 'val')

#plot the value of the train_dataest in matplotlib as hystogram
train_target = torch.Tensor.cpu(train_dataset.train_target).numpy()
train_target = train_target.squeeze(1)
plt.hist(train_target, bins=100)
mean = np.mean(train_target)
# plot the validation in the same histogram 
val_target = torch.Tensor.cpu(val_dataset_1.train_target).numpy()
val_target = val_target.squeeze(1)
plt.hist(val_target, bins=100)
#label the plot
plt.title('Train: ' + str(mean) + '\nVal: ' + str(np.mean(val_target)))
plt.xlabel('Gamma of the target')
plt.ylabel('Number of occurences')
plt.show()

#plot the value of the val_dataset in matplotlib as hystogram
val_target = torch.Tensor.cpu(val_dataset_1.train_target).numpy()
val_target = val_target.squeeze(1)
plt.hist(val_target, bins=100)
mean = np.mean(val_target)
#label the plot
plt.title('Val: ' + str(mean))
plt.xlabel('Gamma of the target')
plt.ylabel('Number of occurences')
plt.show()



