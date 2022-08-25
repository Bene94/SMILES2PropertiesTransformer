import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

from data_processing.data_processing import *

class gamma_dataset(Dataset):  
    
    def __init__(self, comp_list, df, aug=False):

        self.data, self.train_target, self.xT, self.smile_index, self.index = self.load_data(df)
        self.aug = aug
        emb_list = np.empty((comp_list.shape[0],1), dtype=object)

        for i in range(0, comp_list.shape[0]):
            for j in range(0,1):   
                    emb_list[i,j] = comp_list['emb' + str(j)][i]
         
        self.comp_list = emb_list
        self.n_alias = comp_list['n_alias']


    def load_data(self,data):

        # check if collum lnGamma exists
        target = torch.from_numpy(data["y0"].to_numpy()).float()
            
        smile_index = data[['SMILES0','SMILES1']].to_numpy()
        
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

        SMILE1 = int(self.data["SMILES0"][index])
        SMILE2 = int(self.data["SMILES1"][index])

        if self.aug:
            rand1 = np.random.randint(0,int(self.n_alias[SMILE1]))
            rand2 = np.random.randint(0,int(self.n_alias[SMILE2]))
        else :
            rand1 = 0
            rand2 = 0

        seq = np.concatenate([sos,comp_list[SMILE1,rand1], mos, comp_list[SMILE2,rand2], eos])
        
        if len(seq) > 128:
            seq = np.concatenate([sos,comp_list[SMILE1,0], mos, comp_list[SMILE2,0], eos])
        seq = seq[:128]
        train_data[0:len(seq)] = seq

        return self.train_target[index], [train_data, self.xT[index], self.smile_index[index], self.index[index]]

    def __len__(self):
        if len(self.data) == 0:
            return 0
        if len(self.data.shape) == 1:
            return 1
        return self.data.shape[0]

def smile2input(df):
    
    vocab_path = '../vocab/'
    vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')

    df['x'] = 0
    df['y0'] = 0
    df['i'] = 0
    df.reset_index(drop=False, inplace=True)
    df.x = df['x'].astype(float)
    df.T = df['T'].astype(float)
    df.i = df['i'].astype(int)

    solvent_list = df['SMILES0'].drop_duplicates()
    solute_list = df['SMILES1'].drop_duplicates()

    complete_list = pd.concat([solute_list, solvent_list])
    complete_list = complete_list.drop_duplicates()
    complete_list.reset_index(drop=True, inplace=True)
    complete_list = pd.DataFrame({'n_alias':np.ones(complete_list.shape[0]),'SMILE0':complete_list })

    # make new dataframe with same names
    complete_list = apply_vocab(complete_list, vocab_dict)
    data_batches = prep_save(df, complete_list, batch_size=100000)

    # hack this into a dataloader
    dataset = gamma_dataset(complete_list, data_batches[0])
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    return dataloader
    

    

