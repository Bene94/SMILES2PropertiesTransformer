import torch
from torch.utils.data import Dataset, DataLoader
import os
import progressbar as pb
import pandas as pd
import numpy as np
import time

## load data from training_data\ trainig data called train_XXX and validation data calld val_xxx into dataloader.

class gamma_dataset(Dataset):  
    def __init__(self, data_path, data_type, config, aug=True):
        self.data_path = data_path
        self.data_type = data_type
        
        self.data, self.train_target, self.xT, self.smile_index, self.index = self.load_data(config.test)
        if len(self.data) != 0:
            self.load_comp_list(self)
        self.aug = aug
        
        if config.shift != 0:
           self.train_target = self.train_target + config.shift

    def load_comp_list(self, comp_list_file):
         
        comp_list = pd.read_csv(self.data_path + 'comp_list.csv')
        n_emb = int((comp_list.shape[1]-1) / 2)
        emb_list = np.empty((comp_list.shape[0],n_emb), dtype=object)

        for i in range(0, comp_list.shape[0]):
            for j in range(0,n_emb):   
                temp = comp_list['emb' + str(j)][i]
                if str(temp) != 'nan':
                    emb_list[i,j] = np.fromstring(temp[1:-1], sep=" ", dtype=np.int)

        self.comp_list = emb_list
        self.n_alias = comp_list['n_alias']


    def load_data(self,test):
        files = os.listdir(self.data_path)
        #load all files into a numpy array from a cvs file
        dirs = os.listdir(self.data_path)
        files = [x for x in dirs if os.path.isfile(os.path.join(self.data_path, x))]
        li = []
        for filename in files:
            #bar.update(i)
            if filename.startswith(self.data_type) and not filename.endswith("comp_list.csv"):
                df = pd.read_csv(os.path.join(self.data_path, filename), index_col=None, header=0)
                li.append(df)

        
        if li == []:
            print("No data " + self.data_type + " found")
            return [], [], [], [], []
        
        data = pd.concat(li, axis=0, ignore_index=True)

        if test:
            data = data.iloc[0:500,:]

        # if x and T do not exist add them to the dataframe
        if "x" not in data.columns:
           data["x"] = np.zeros(data.shape[0])
        if "T" not in data.columns:
           data["T"] = np.ones(data.shape[0]) * 298.15
         

        target = torch.from_numpy(data[data.columns[data.columns.str.startswith('y')]].to_numpy()).float()      
        smile_index = torch.from_numpy(data[data.columns[data.columns.str.startswith('SMILES')]].to_numpy()).int()
        xT = torch.from_numpy(data[["x","T"]].to_numpy()).float()
        index = torch.from_numpy(data["i"].to_numpy()).int()

        return data, target, xT, smile_index, index 

    def __getitem__(self,index):
    
        if len(self.data) == 0:
            return [], [], [], [], []
        
        # time the function
        sos = np.array((1,), dtype=np.int)
        mos = np.array((2,), dtype=np.int)
        eos = np.array((3,), dtype=np.int)

        train_data = np.zeros(128, dtype=np.int)

        comp_list = self.comp_list

        smile_list = []
        for col in self.data.columns:
            if col.startswith('SMILES'):
                smile_list.append(self.data[col][index])
        
        if self.aug:
            rand_list = [ np.random.randint(0,int(self.n_alias[i])) for i in smile_list]
        else :
            rand_list = np.zeros(len(smile_list), dtype=np.int)
    
        # midel section 
        if len(smile_list) > 1:
            mid = np.concatenate([ np.concatenate((mos , self.comp_list[smile, rand])) for smile, rand in zip(smile_list[1:] , rand_list[1:])])
        else:
            mid = []
        seq = np.concatenate([sos, comp_list[smile_list[0], rand_list[0]], mid, eos])
        seq = seq[:128]
        train_data[0:len(seq)] = seq

        return self.train_target[index], [train_data, self.xT[index], self.smile_index[index], self.index[index]]

    def __len__(self):
        if len(self.data) == 0:
            return 0
        if len(self.data.shape) == 1:
            return 1
        return self.data.shape[0]

def load_data(config,local = False,test = False):

    if local:
        data_path = os.path.join('../data/' + config.data_path + '/')
    else:
        data_path = os.path.join('/mnt/xprun/data/' + config.data_path + '/')

    train_dataset = gamma_dataset(data_path, 'train', config, aug=True)
    val_0_dataset = gamma_dataset(data_path, 'val_0', config, aug=False)
    val_1_dataset = gamma_dataset(data_path, 'val_1', config, aug=False)
    val_2_dataset = gamma_dataset(data_path, 'val_2', config, aug=False)

    training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    if len(val_0_dataset) > 0:
        val_0_data = DataLoader(val_0_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    else:
        val_0_data = []
    if len(val_1_dataset) > 0:
        val_1_data = DataLoader(val_1_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    else:
        val_1_data = []
    if len(val_2_dataset) > 0:
        val_2_data = DataLoader(val_2_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    else:
        val_2_data = []

    return training_data, val_0_data, val_1_data, val_2_data

def load_data_full(config,local = False,test = False):

    if local:
        data_path = os.path.join('../data/' + config.data_path + '/')
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