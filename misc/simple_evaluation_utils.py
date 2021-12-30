import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

from data_processing.data_processing_cosmo import *

class gamma_dataset(Dataset):  
    def __init__(self, data):
        target = data[:, 0]
        smiles = data[:, 1:129]
        xT = data[:,-2:]

        smiles = np.array(smiles, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        xT = np.array(xT, dtype=np.float32)
        smiles = torch.tensor(smiles)
        target = torch.from_numpy(target)
        xT = torch.from_numpy(xT)

        smiles = smiles.type(torch.ByteTensor)
        target = target.type(torch.FloatTensor)
        target = target.view((target.shape[0],1,1))
        xT = xT.type(torch.FloatTensor)
        self.xT = xT
        self.train_data = smiles
        self.train_target = target
        self.smile_index = np.zeros((data.shape[0],2))
        self.index = np.zeros(data.shape[0])
    
    def __getitem__(self, index): 
        return self.train_target[index],  [self.train_data[index], self.xT[index], self.smile_index[index], self.index[index]]

    def __len__(self):
        if len(self.train_data.shape) == 1:
            return 1
        return self.train_data.shape[0]


def smile2input(solvent,solute,x,T):
    
    vocab_path = '../vocab/'
    vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')

    df = pd.DataFrame(columns=['SMILES','gamma','x','T'])

    cat_smile = list(vocab_dict.keys())[1] + solvent + list(vocab_dict.keys())[2] + solute + list(vocab_dict.keys())[3]

    for x_i in x:
        for T_j in T:
            df.loc[-1,'SMILES'] = cat_smile
            df.loc[-1,'gamma'] = 0
            df.loc[-1,'x'] = x_i
            df.loc[-1,'T'] = T_j
            df.index = df.index + 1



    # make new dataframe with same names
    data = apply_vocab(df, vocab_dict, np.inf , - np.inf)

    # hack this into a dataloader
    dataset = gamma_dataset(data)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    return dataloader
    

