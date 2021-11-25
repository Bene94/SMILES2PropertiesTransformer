import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import progressbar as pb
import numpy as np

## load data from training_data\ trainig data called train_XXX and validation data calld val_xxx into dataloader.

class gamma_dataset(Dataset):  
    def __init__(self, root, data_type, config):
        self.root = root
        self.data_type = data_type
        self.train_data, self.train_target, self.xT, self.smile_index = self.load_data(config.test)
        if config.shift != 0:
           self.train_target = self.train_target + config.shift

        if not config.mode == 'reg':
            self.bin_data(config)

    def load_data(self,test):
        #laods the data from sefl.root  acroidng to type from batches in current direcory plus root
        files = os.listdir(self.root)
        #load all files into a numpy array from a cvs file
        dirs = os.listdir(self.root)
        files = [x for x in dirs if os.path.isfile(os.path.join(self.root, x))]
        # progress bar for loading data 
        #bar = pb.ProgressBar(maxval=len(files), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
        # load first file into data then append all other files data is a .npy file
        data = np.array([])
        for i in range(0, len(files)):
            #bar.update(i)
            if files[i].startswith(self.data_type):
                # when the data array is empty create lese append
                if data.size == 0:
                    data = np.load(self.root + files[i])
                else:
                    data = np.append(data, np.load(self.root + files[i]), axis=0)

            if test and len(data) > 5000:
                break
                    
        target = data[:, 0]
        smiles = data[:, 1:129]
        xT = data[:,129:131]
        smile_index = data[:,131:133]

        smiles = torch.tensor(smiles)
        target = torch.from_numpy(target)
        xT = torch.from_numpy(xT)

        smiles = smiles.type(torch.ByteTensor)
        target = target.type(torch.FloatTensor)
        target = target.view((target.shape[0],1,1))
        xT = xT.type(torch.FloatTensor)
        #return the data and target
        return smiles, target, xT, smile_index 

    def bin_data(self, config):
        bound = config.bound
        n_bins = config.bins
        bins = np.linspace(-bound, bound, n_bins)
        bins = np.append(bins, np.inf)
        bins = np.append(-np.inf, bins)
        self.train_target = np.digitize(self.train_target, bins)
    
    def __getitem__(self, index): 
        return self.train_target[index],  [self.train_data[index], self.xT[index]]

    def __len__(self):
        if len(self.train_data.shape) == 1:
            return 1
        return self.train_data.shape[0]

def load_data(config,local = False,test = False):

    if local:
        data_path = os.path.join('/home/bene/NNGamma/data/' + config.data_path + '/')
    else:
        data_path = os.path.join('/mnt/xprun/data/' + config.data_path + '/')

    train_dataset = gamma_dataset(data_path, 'train', config)
    val_0_dataset = gamma_dataset(data_path, 'val_0', config)
    val_1_dataset = gamma_dataset(data_path, 'val_1', config)
    val_2_dataset = gamma_dataset(data_path, 'val_2', config)

    if test:
        train_dataset.train_data = train_dataset.train_data[0:500]
        train_dataset.train_target = train_dataset.train_target[0:500]

        val_0_dataset.train_data = val_0_dataset.train_data[0:2]
        val_0_dataset.train_target = val_0_dataset.train_target[0:2]

        val_1_dataset.train_data = val_1_dataset.train_data[0:2]
        val_1_dataset.train_target = val_1_dataset.train_target[0:2]

        val_2_dataset.train_data = val_2_dataset.train_data[0:2]
        val_2_dataset.train_target = val_2_dataset.train_target[0:2]


    training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_0_data = DataLoader(val_0_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_1_data = DataLoader(val_1_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_2_data = DataLoader(val_2_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    return training_data, val_0_data, val_1_data, val_2_data

def load_data_full(config,local = False,test = False):

    if local:
        data_path = os.path.join('/home/bene/NNGamma/data/' + config.data_path + '/')
    else:
        data_path = os.path.join('/mnt/xprun/data/' + config.data_path + '/')

    train_dataset = gamma_dataset(data_path, '', config)

    if test:
        train_dataset.train_data = train_dataset.train_data[0:500]
        train_dataset.train_target = train_dataset.train_target[0:500]


    training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    return training_data




if __name__ == '__main__':
    train_dataset = gamma_dataset('TrainingData/', 'train')
    val_dataset = gamma_dataset('TrainingData/', 'val')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    train_features, train_labels = next(iter(train_dataloader))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)



