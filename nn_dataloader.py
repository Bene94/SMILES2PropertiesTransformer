import torch
from torch.utils.data import Dataset, DataLoader
import os
import progressbar as pb
import pandas as pd
import numpy as np
import time

## load data from training_data\ trainig data called train_XXX and validation data calld val_xxx into dataloader.

class gamma_dataset(Dataset):  
    def __init__(self, root, data_type, config, aug=True):
        self.root = root
        self.data_type = data_type
        
        self.train_data, self.train_target, self.xT, self.smile_index, self.index = self.load_data(config.test)
        self.load_comp_list(self)
        self.aug = aug
        
        if config.shift != 0:
           self.train_target = self.train_target + config.shift

    def load_comp_list(self, comp_list_file):
         
        comp_list = pd.read_csv(self.root + 'comp_list.csv')
        emb_list = np.empty((comp_list.shape[0],10), dtype=object)

        for i in range(0, comp_list.shape[0]):
            for j in range(0,10):
                temp = comp_list['emb' + str(j)][i]
                if str(temp) != 'nan':
                    emb_list[i,j] = np.fromstring(temp[1:-1], sep=" ", dtype=np.int)

        self.comp_list = emb_list
        self.n_alias = comp_list['n_alias']


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
            if files[i].startswith(self.data_type) and not files[i].startswith("comp_list"):
                # when the data array is empty create lese append
                if data.size == 0:
                    data = np.load(self.root + files[i])
                else:
                    data = np.append(data, np.load(self.root + files[i]), axis=0)

            if test and len(data) > 5000:
                break

        if data is None:
            print("No data " + self.data_type + " found")
            return [], [], [], [], []

        target = data[:, 0]
        smiles = data[:, 1:129]
        xT = data[:,129:131]
        smile_index = data[:,131:133]
        if data.shape[1] == 134:
            index = data[:,133]
        else:
            index = np.zeros(len(data))

        smiles = torch.tensor(smiles)
        target = torch.from_numpy(target)
        xT = torch.from_numpy(xT)

        smiles = smiles.type(torch.ByteTensor)
        target = target.type(torch.FloatTensor)
        target = target.view((target.shape[0],1,1))
        xT = xT.type(torch.FloatTensor)
        #return the data and target
        return smiles, target, xT, smile_index, index 

    def __getitem__(self,index):
        
        # time the function
        sos = np.array((1,), dtype=np.int)
        mos = np.array((2,), dtype=np.int)
        eos = np.array((3,), dtype=np.int)

        train_data = np.zeros(128, dtype=np.int)

        comp_list = self.comp_list

        SMILE1 = int(self.smile_index[index][0])
        SMILE2 = int(self.smile_index[index][1])

        if self.aug:
            rand1 = np.random.randint(0,int(self.n_alias[SMILE1]))
            rand2 = np.random.randint(0,int(self.n_alias[SMILE2]))
        else :
            rand1 = 0
            rand2 = 0

        emb = np.concatenate([sos,comp_list[SMILE1,rand1], mos, comp_list[SMILE2,rand2], eos])
        
        if len(emb) > 128:
            emb = np.concatenate([sos,comp_list[SMILE1,0], mos, comp_list[SMILE2,0], eos])

        train_data[0:len(emb)] = emb

        return self.train_target[index], [train_data, self.xT[index], self.smile_index[index], self.index[index]]

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
    pass