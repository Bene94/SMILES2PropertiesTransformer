import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import progressbar as pb
import numpy as np

## load data from training_data\ trainig data called train_XXX and validation data calld val_xxx into dataloader.

class gamma_dataset(Dataset):  
    def __init__(self, root, data_type):
        self.root = root
        self.data_type = data_type
        self.train_data, self.train_target = self.load_data()
        

    def load_data(self):
        #laods the data from sefl.root  acroidng to type from batches in root
        files = os.listdir(self.root)
        #load all files into a numpy array from a cvs file

        # progress bar for loading data 
        bar = pb.ProgressBar(maxval=len(files), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
        # load first file into data then append all other files data is a .npy file
        data = np.array([])
        for i in range(0, len(files)):
            bar.update(i)
            if files[i].startswith(self.data_type):
                # when the data array is empty create lese append
                if data.size == 0:
                    data = np.load(self.root + files[i])
                else:
                    data = np.append(data, np.load(self.root + files[i]), axis=0)
                    
        target = data[:, 0]
        data = data[:, 1:]

        data = torch.tensor(data)
        target = torch.from_numpy(target)

        data = data.type(torch.cuda.ByteTensor)
        target = target.type(torch.cuda.FloatTensor)
        target = target.view((target.shape[0],1,1))
        #return the data and target
        return data, target
                    
    
    def __getitem__(self, index): 
        return self.train_data[index], self.train_target[index]

    def __len__(self):
        return len(self.train_data)

if __name__ == '__main__':
    train_dataset = gamma_dataset('TrainingData/', 'train')
    val_dataset = gamma_dataset('TrainingData/', 'val')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    train_features, train_labels = next(iter(train_dataloader))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)



