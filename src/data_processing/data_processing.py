from ntpath import join
import pandas as pd
import os
import csv
from pandas.core.arrays.integer import safe_cast
import progressbar as pb
import numpy as np
import click
import sys


from pandas.core.frame import DataFrame

@click.command()

@click.option('--file_path', '-p',default=["inf_cosmo"], help='Location of raw data', multiple=True)
@click.option('--save_path', '-s', default="data", help='Location of output data')
@click.option('--vocab_path', default="vocab", help='Location of vocab')
@click.option('--ow', default=True, help='overwirte exising files in the save folder or add to them ')
@click.option('--source', default='COSMO', help='see if to select COSMO or EXP')
@click.option('--pre', default='', help='prefix for the output files')

@click.option('--frac', default=0.05, help='fraction of data to be used for testing and validation')

@click.option('--h2o', default=False, help='allows H2O in the validation set')
@click.option('--seed', default=42, help='seed of the smile sampling for validation')

def main(file_path, save_path, vocab_path, frac, seed, ow, h2o, source, pre):
    processing(file_path, save_path, vocab_path, frac, seed, ow, h2o, source, pre)

def processing(folder_name, save_path, vocab_path, frac, seed, ow, h2o, source, pre):
    
    file_path, file_out, vocab_path, alias_path  =  get_paths(save_path, vocab_path) 

    vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')
    df_join, comp_list, index_list  = load_exp_data(file_path, folder_name)
        
    comp_list = aug_data(comp_list, alias_path=alias_path)
    
    ## apply the vocab to the smiles
    comp_list = apply_vocab(comp_list, vocab_dict)
    
    if source == 'COSMO' or source == 'EXP':
        val_solvent_indx, val_solute_indx = get_idx_test_val(index_list[0], index_list[1], frac, seed)

        # removes water from the validation set
        water_index = comp_list.index[comp_list["SMILE0"]=="O"][0]    
        if (val_solute_indx == water_index).any() and not h2o:
            val_solute_indx = val_solute_indx[val_solute_indx != water_index]
        if (val_solvent_indx == water_index).any() and not h2o:
            val_solvent_indx = val_solvent_indx[val_solvent_indx != water_index]

    if source == 'COSMO':    
        df_list = split_data_test_val(df_join, val_solvent_indx, val_solute_indx, comp_list, seed)
    elif source == 'EXP':
        df_list = split_data_test_val_exp(df_join, val_solvent_indx, val_solute_indx, comp_list, seed)
    elif source == 'single':
        df_list = split_data_test_val_single(df_join, comp_list, seed, frac)
    elif source == 'noSplit':
        df_list = [df_join]

    # make input data
    for i, df in enumerate(df_list):
        if not df.empty:
            if i == 0:
                prefix = 'train'
            else:
                prefix = 'val_' + str(i-1)
            if source == 'noSplit':
                prefix = pre
            data_batches = prep_save(df, comp_list, batch_size=100000)
            save_batches(data_batches, file_out, prefix, ow)
    
    # save comp
    comp_list.to_csv(file_out + pre +'comp_list.csv', index=False)

def processing_n_in(foler_name, save_path, vocab_path, seed, ow, n, comp_list):
    
    file_path, file_out, vocab_path, alias_path  =  get_paths(save_path, vocab_path) 


    vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')
    df_join, __, ___  = load_exp_data(file_path, foler_name)
    df_list = split_data_test_val_exp_n_in_noH2O(df_join, comp_list,seed, n)

    # make input data
    for i, df in enumerate(df_list):
        if not df.empty:
            if i == 0:
                prefix = 'train'
            else:
                prefix = 'val_' + str(i-1)
            data_batches = prep_save(df, comp_list, batch_size=100000)
            save_batches(data_batches, file_out, prefix, ow) 
    # save comp
    comp_list.to_csv(file_out + 'comp_list.csv', index=False)

def processing_n_out(foler_name, save_path, vocab_path, ow, comp_list, systems, index):
     
    file_path, file_out, vocab_path, alias_path  =  get_paths(save_path, vocab_path) 

    vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')
    df_join, __, __  = load_exp_data(file_path, foler_name)
    df_list = split_data_test_val_exp_n_out(df_join, comp_list, systems, index)

    # make input data
    for i, df in enumerate(df_list):
        if not df.empty:
            if i == 0:
                prefix = 'train'
            else:
                prefix = 'val_' + str(i-1)
            data_batches = prep_save(df, comp_list, batch_size=100000)
            save_batches(data_batches, file_out, prefix, ow) 
    # save comp
    comp_list.to_csv(file_out + 'comp_list.csv', index=False)

def get_comp_list(foler_name, vocab_path):

    file_path, file_out, vocab_path, alias_path  =  get_paths('', vocab_path) 

    vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')
    
    df_join, comp_list, __  = load_exp_data(file_path, foler_name) 
    comp_list = aug_data(comp_list, alias_path=alias_path)
    comp_list = apply_vocab(comp_list, vocab_dict)

    systems = df_join.groupby(['SMILES0','SMILES1']).size().reset_index().rename(columns={0:'count'})

    return comp_list, systems, df_join

def processing_n_out_sund(foler_name, save_path, vocab_path, ow, comp_list, systems, index):
     
    file_path, file_out, vocab_path, alias_path  =  get_paths(save_path, vocab_path) 


    vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')
    df_join, __, __  = load_exp_data(file_path, foler_name)
    df_list = split_data_test_val_exp_n_out(df_join, comp_list, systems, index)

    # make input data
    for i, df in enumerate(df_list):
        if not df.empty:
            if i == 0:
                prefix = 'train'
            else:
                prefix = 'val_' + str(i-1)
            data_batches = prep_save_sund(df, comp_list, batch_size=100000)
            save_batches(data_batches, file_out, prefix, ow) 
    # save comp
    comp_list.to_csv(file_out + 'comp_list.csv', index=False)

def load_exp_data(file_path, foler_names):
    #load the data from the experiment 
    df = pd.DataFrame()

    for folder_name in foler_names:
        # list all files in the folder
        files = os.listdir(file_path + folder_name)
        # load all files into a panda
        bar = pb.ProgressBar(maxval=len(files), widgets=["Processing files from " + folder_name + ": ",pb.Timer(), pb.Bar('=', '[', ']'), pb.ETA()])
        bar.start()
        for i, file in enumerate(files):
            bar.update(i)
            temp_path = os.path.join(file_path, folder_name, file)
            temp_df = pd.read_csv(temp_path, sep=',', index_col=None)
            df = pd.concat([df, temp_df], ignore_index=True)
        bar.finish()
    # remove all nan rows
    if not 'i' in df.columns:
        df.loc[:,'i'] = np.array(range(len(df))) 
    
    df.loc[df.i.isna(),'i'] = 0

    df = df.dropna()
    df = df.reset_index(drop=True)

    comp_list = []
    for col in df.columns:
        if col.startswith('SMILES'):
            comp = df[col].drop_duplicates().to_list()
            comp_list = comp_list + comp
    # remove duplicates
    comp_list = list(dict.fromkeys(comp_list))

    complete_list = pd.DataFrame({'n_alias':np.ones(len(comp_list)),'SMILE0':comp_list})

    index_list = []
    for col in df.columns:
        if col.startswith('SMILES'):
            components = df[col].drop_duplicates() 
            index = complete_list.index[complete_list['SMILE0'].isin(components)]
            index_list.append(index)

    return df, complete_list,index_list 

def load_vocab(file_path,vocab_name):
    
    #load the vocab from a .cvs file and adds it to a dictoinary
    with open(file_path + vocab_name + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        vocab_dict = {}
        for row in csv_reader:
            vocab_dict[row[0]] = int(row[1])
        
    return vocab_dict

def aug_data(comp_list,alias_path):

    alias_dict = np.load(alias_path, allow_pickle=True).item()
    bar = pb.ProgressBar(maxval=len(comp_list), widgets=['Get Aliases: ',pb.Timer(), pb.Bar('=', '[', ']'), pb.ETA()])
    bar.start()
    #add the alias to the new collumn alias to comp_list
    for i in range(len(comp_list)):
        bar.update(i)
        for j, alias in enumerate(alias_dict[comp_list.loc[i,'SMILE0']]):
            comp_list.loc[i,'SMILE' +str(j+1)] = alias
        comp_list.loc[i,'n_alias'] = len(alias_dict[comp_list.loc[i,'SMILE0']]) +1 
    bar.finish()
    return comp_list

def prep_save(df, comp_list, batch_size):
    # converts the dataframe in np-arrays for saving and splits its in batches
    smile_dict = pd.Series(comp_list.index.values,index=comp_list.SMILE0.values).to_dict()
    for col in df.columns:
        if col.startswith('SMILES'):
            df[col] = df[col].map(smile_dict).astype(int)
    df = df.dropna()
    # split the dataframe into batches
    df_list = np.array_split(df, int(np.ceil(len(df)/batch_size)))
    return df_list

def prep_save_sund(df, comp_list, batch_size):
    df.rename(columns={'SMILES0':'Solute_SMILES', 'SMILES1':'Solvent_SMILES', 'y0':'Literature'}, inplace=True)
    
    df_list = np.array_split(df, int(np.ceil(len(df)/batch_size)))
    return df_list

def apply_vocab(comp_list, vocab_dict):
    
    # add a collumn for each SMILE and alias with the embedding
    for j, columns in enumerate(comp_list.columns):
        if not columns == 'n_alias':
            for i in range(len(comp_list)):
                if i == 0:
                    comp_list.insert(comp_list.shape[1],'emb'+str(j-1),np.nan)
                    comp_list['emb' + str(j-1)] = comp_list['emb'+str(j-1)].astype(object, copy=True)
                if not pd.isnull(comp_list.loc[i,columns]):
                    if comp_list.loc[i,columns] == 'O':
                        comp_list.at[i,'emb'+str(j-1)] = np.array([vocab_dict['H2O']])
                    else:
                        comp_list.at[i,'emb'+str(j-1)] = np.array([vocab_dict [char] for char in comp_list.loc[i,columns]])            

    return comp_list

def get_idx_test_val(solvent_indx, solute_indx, frac, seed): 

    np.random.seed(seed)
    val_solvent_indx= np.random.choice(solvent_indx, int(frac*len(solvent_indx)), replace=False, )
    np.random.seed(seed)
    val_solute_indx= np.random.choice(solute_indx, int(frac*len(solute_indx)), replace=False, )

    return val_solvent_indx, val_solute_indx

def split_data_test_val(df_join, val_solvent_indx, val_solute_indx, comp_list,seed):
     
    df_temp = df_join.copy()
    
    #reste index
    df_temp.reset_index(drop=True, inplace=True) 
    df_temp_val_0 = df_temp.loc[(df_temp.iloc[:,0].isin(comp_list.loc[val_solute_indx,'SMILE0'])) & (df_temp.iloc[:,1].isin(comp_list.loc[val_solvent_indx,'SMILE0']))]
    df_temp_val_1 = df_temp.loc[(df_temp.iloc[:,0].isin(comp_list.loc[val_solute_indx,'SMILE0'])) ^ (df_temp.iloc[:,1].isin(comp_list.loc[val_solvent_indx,'SMILE0']))]
    
    # df_temp is the dataframe that is not in the validation set
    df_temp_train = df_temp.loc[ ~df_temp.index.isin(df_temp_val_0.index) & ~df_temp.index.isin(df_temp_val_1.index)]
    
    #sample 10 % of the dataframe that is not in the validation set
    df_temp_val_2 = df_temp_train.sample(frac=0.1, replace=False, random_state=seed)
    df_temp_train = df_temp_train.drop(df_temp_val_2.index)

    # remove all nan form df_temp_train
    df_temp_train = df_temp_train.dropna()
    df_temp_train = df_temp_train.reset_index(drop=True)
    df_temp_val_0 = df_temp_val_0.reset_index(drop=True)
    df_temp_val_1 = df_temp_val_1.reset_index(drop=True)
    df_temp_val_2 = df_temp_val_2.reset_index(drop=True)

    return [df_temp_train, df_temp_val_0, df_temp_val_1, df_temp_val_2]

def split_data_test_val_exp(df_join, val_solvent_indx, val_solute_indx, comp_list,seed):
    
    df_temp = df_join.copy()
    
    #reste index
    df_temp.reset_index(drop=True, inplace=True) 
    df_temp_val_0 = df_temp.loc[(df_temp.iloc[:,0].isin(comp_list.loc[val_solute_indx,'SMILE0'])) & (df_temp.iloc[:,1].isin(comp_list.loc[val_solvent_indx,'SMILE0']))]
    df_temp_val_1 = df_temp.loc[(df_temp.iloc[:,0].isin(comp_list.loc[val_solute_indx,'SMILE0'])) ^ (df_temp.iloc[:,1].isin(comp_list.loc[val_solvent_indx,'SMILE0']))]
    
    # df_temp is the dataframe that is not in the validation set
    df_temp_train = df_temp.loc[ ~df_temp.index.isin(df_temp_val_0.index) & ~df_temp.index.isin(df_temp_val_1.index)]
    
    #sample 10 % of the dataframe that is not in the validation set
    systems =  df_temp_train.groupby(['solvent','solute']).size().reset_index().rename(columns={0:'count'})
    systems_val_2 = systems.sample(frac=0.01, random_state=seed)
   
    bar = pb.ProgressBar(maxval=len(systems_val_2), widgets=['Checking integrety of val sets: ',pb.Timer(), pb.Bar('=', '[', ']'), pb.ETA()])
    bar.start()
    count = 0
    
    df_temp_val_2 = pd.DataFrame(columns=df_temp_train.columns)
    
    for i in systems_val_2.index:
        bar.update(count)
        count += 1	

        temp = systems_val_2.loc[i,['SMILES0','SMILES1']] == df_temp_train.loc[:,['SMILES0','SMILES1']]
        temp = temp[temp['SMILES0']]
        temp = temp[temp['SMILES1']]

        temp_2 = df_temp_train.drop(temp.index)

        if len(temp_2[temp_2['SMILES1'] == systems_val_2.loc[i,'SMILES1']]) == 0 and len(temp_2[temp_2['SMILES0'] == systems_val_2.loc[i,'SMILES0']]) == 0:
            df_temp_val_0 = df_temp_val_0.append(df_temp_train.loc[temp.index,:])
            df_temp_train = df_temp_train.drop(temp.index)

        elif len(temp_2[temp_2['SMILES1'] == systems_val_2.loc[i,'SMILES1']]) == 0 or len(temp_2[temp_2['SMILES0'] == systems_val_2.loc[i,'SMILES0']]) == 0:
            df_temp_val_1 = df_temp_val_1.append(df_temp_train.loc[temp.index,:])
            df_temp_train = df_temp_train.drop(temp.index)
        else:
            df_temp_val_2 = pd.concat([df_temp_val_2, df_temp_train.loc[temp.index,:]])

    bar.finish()

    df_temp_val_2.drop_duplicates(inplace=True)
    df_temp_val_1.drop_duplicates(inplace=True)
    df_temp_val_0.drop_duplicates(inplace=True)

    df_temp_train = df_temp_train.drop(df_temp_val_2.index)

    print("len of data: " + str(len(df_join)))
    print("len of val_0: " + str(len(df_temp_val_0)))
    print("len of val_1: " + str(len(df_temp_val_1)))
    print("len of val_2: " + str(len(df_temp_val_2)))
    print("len of train: " + str(len(df_temp_train)))
    print("len of val_0 + val_1 + val_2 + train: " + str(len(df_temp_val_0) + len(df_temp_val_1) + len(df_temp_val_2) + len(df_temp_train)) )

    # remove all nan form df_temp_train
    df_temp_train = df_temp_train.dropna()
    df_temp_train = df_temp_train.reset_index(drop=True)
    df_temp_val_0 = df_temp_val_0.reset_index(drop=True)
    df_temp_val_1 = df_temp_val_1.reset_index(drop=True)
    df_temp_val_2 = df_temp_val_2.reset_index(drop=True)

    return [df_temp_train, df_temp_val_0, df_temp_val_1, df_temp_val_2]

def split_data_test_val_single(df_join, comp_list, seed, frac):
    # function has only val_0 and train
    df_temp = df_join.copy()
    mixtures = df_temp.groupby(['SMILES0']).size().reset_index().rename(columns={0:'count'})
    mixtures_val_0 = mixtures.sample(frac=frac, random_state=seed)
    mixtures_val_0 = mixtures_val_0.reset_index(drop=True)
    
    df_temp_val_0 = df_temp.loc[df_temp.iloc[:,0].isin(mixtures_val_0.SMILES0)]
    df_temp_train = df_temp.loc[~df_temp.index.isin(df_temp_val_0.index)]

    df_temp_val_0 = df_temp_val_0.reset_index(drop=True)
    df_temp_train = df_temp_train.reset_index(drop=True)

    return [df_temp_train, df_temp_val_0]

def split_data_test_val_exp_n_in_noH2O(df_join, comp_list,seed, n):
    
    df_join = df_join[(df_join['SMILES1'] != 'O') & (df_join['SMILES0'] != 'O')]

    systems =  df_join.groupby(['SMILES1','SMILES0']).size().reset_index().rename(columns={0:'count'})
    systems_train = systems.sample(n=n, random_state=seed)

    df_temp_train = pd.DataFrame(columns=df_join.columns)
    df_temp_val_0 = df_join.copy()

    idx_solute = []
    idx_solvent = []

    for i in systems_train.index:
        temp = df_join[(df_join['SMILES1'] == systems_train.loc[i,'SMILES1']) & (df_join['SMILES0'] == systems_train.loc[i,'SMILES0'] )]
        to_train = temp.sample(n=1)
        to_drop = temp.drop(to_train.index)
        df_temp_train = pd.concat([df_temp_train, to_train])
        
        df_join = df_join.drop(temp.index)
        df_temp_val_0 = df_temp_val_0.drop(df_temp_val_0[df_temp_val_0['SMILES1'] == systems_train.loc[i,'SMILES1']].index)
        df_temp_val_0 = df_temp_val_0.drop(df_temp_val_0[df_temp_val_0['SMILES0'] == systems_train.loc[i,'SMILES0']].index)
        df_temp_val_0 = df_temp_val_0.drop(df_temp_val_0[df_temp_val_0['SMILES1'] == systems_train.loc[i,'SMILES0']].index)
        df_temp_val_0 = df_temp_val_0.drop(df_temp_val_0[df_temp_val_0['SMILES0'] == systems_train.loc[i,'SMILES1']].index)
        
    for i in systems_train.index:
        idx_solvent += df_join[df_join['SMILES1'] == systems_train.loc[i,'SMILES1']].index.tolist()
        idx_solute += df_join[df_join['SMILES0'] == systems_train.loc[i,'SMILES0']].index.tolist()

    # find the elements that are both in idx_solvent and idx_solute
    idx_solvent = list(set(idx_solvent) & set(idx_solute))
    df_temp_val_2 = df_join.loc[idx_solvent,:]

    df_temp_val_1 = df_join.drop(df_temp_val_2.index)
    df_temp_val_1 = df_temp_val_1.drop(df_temp_val_0.index)

    # drop water from val 0
    
    print("len of data: " + str(len(df_join)))
    print("len of val_0: " + str(len(df_temp_val_0)))
    print("len of val_1: " + str(len(df_temp_val_1)))
    print("len of val_2: " + str(len(df_temp_val_2)))
    print("len of train: " + str(len(df_temp_train)))
    print("len of val_0 + val_1 + val_2 + train: " + str(len(df_temp_val_0) + len(df_temp_val_1) + len(df_temp_val_2) + len(df_temp_train)) )

    # remove all nan form df_temp_train
    df_temp_train = df_temp_train.dropna()
    df_temp_train = df_temp_train.reset_index(drop=True)
    df_temp_val_0 = df_temp_val_0.reset_index(drop=True)
    df_temp_val_1 = df_temp_val_1.reset_index(drop=True)
    df_temp_val_2 = df_temp_val_2.reset_index(drop=True)

    return [df_temp_train, df_temp_val_0, df_temp_val_1, df_temp_val_2]

def split_data_test_val_exp_n_out(df_join, comp_list, systems, index):

    systems_val_0 = systems.iloc[index]

    df_temp_val_0 = pd.DataFrame(columns=df_join.columns)
    df_temp_train = df_join.copy()

    idx_solute = []
    idx_solvent = []

    for i in systems_val_0.index:
        to_val_0 = df_join[(df_join['SMILES1'] == systems_val_0.loc[i,'SMILES1']) & (df_join['SMILES0'] == systems_val_0.loc[i,'SMILES0'])]
        df_temp_val_0 = pd.concat([df_temp_val_0, to_val_0])
    
        df_temp_train = df_temp_train.drop(df_temp_train[df_temp_train['SMILES1'] == systems_val_0.loc[i,'SMILES1']].index)
        df_temp_train = df_temp_train.drop(df_temp_train[df_temp_train['SMILES0'] == systems_val_0.loc[i,'SMILES0']].index)
        df_temp_train = df_temp_train.drop(df_temp_train[df_temp_train['SMILES1'] == systems_val_0.loc[i,'SMILES0']].index)
        df_temp_train = df_temp_train.drop(df_temp_train[df_temp_train['SMILES0'] == systems_val_0.loc[i,'SMILES1']].index)

    for i in systems_val_0.index:
        idx_solvent += df_join[df_join['SMILES1'] == systems_val_0.loc[i,'SMILES1']].index.tolist()
        idx_solute += df_join[df_join['SMILES0'] == systems_val_0.loc[i,'SMILES0']].index.tolist()
        idx_solvent += df_join[df_join['SMILES1'] == systems_val_0.loc[i,'SMILES0']].index.tolist()
        idx_solute += df_join[df_join['SMILES0'] == systems_val_0.loc[i,'SMILES1']].index.tolist()

    idx_val_0 = set(df_temp_val_0.index)
    # find the elements that are either in idx_solvent and idx_solute but not in idx_val_0
    idx_solvent = list((set(idx_solvent) | set(idx_solute)) - idx_val_0)
    df_temp_val_1 = df_join.loc[idx_solvent,:]

    train_systems = df_temp_train.groupby(['SMILES1','SMILES0']).size().reset_index().rename(columns={0:'count'})
    
    val_2_systems = train_systems.sample(frac=0.05, random_state= index[0])

    df_temp_val_2 = pd.DataFrame(columns=df_join.columns)
    to_val = pd.DataFrame(columns=df_join.columns)

    for i in val_2_systems.index:
        temp_to_val = df_join[(df_join['SMILES1'] == val_2_systems.loc[i,'SMILES1']) & (df_join['SMILES0'] == val_2_systems.loc[i,'SMILES0'])]
        df_temp_train = df_temp_train.drop(temp_to_val.index)
        to_val = pd.concat([to_val, temp_to_val])
    
    train_solvents = set(df_temp_train['SMILES1'].unique().tolist())
    train_solutes = set(df_temp_train['SMILES0'].unique().tolist())

    to_val_solvents = set(to_val['SMILES1'].unique().tolist())
    to_val_solutes = set(to_val['SMILES0'].unique().tolist())

    not_train_solvents = to_val_solvents - train_solvents
    not_train_solutes = to_val_solutes - train_solutes

    to_val.reset_index(inplace=True, drop=True)

    
    temp = to_val[to_val['SMILES0'].isin(list(not_train_solutes)) & to_val['SMILES1'].isin(list(not_train_solvents))]
    df_temp_val_0 = pd.concat([df_temp_val_0, temp])
    to_val = to_val.drop(temp.index)

    temp = to_val[to_val['SMILES0'].isin(list(not_train_solutes)) | to_val['SMILES1'].isin(list(not_train_solvents))]
    df_temp_val_1 = pd.concat([df_temp_val_1, temp])
    to_val = to_val.drop(temp.index)

    df_temp_val_2 = pd.concat([df_temp_val_2, to_val])

    print("len of data: " + str(len(df_join)))
    print("len of val_0: " + str(len(df_temp_val_0)))
    print("len of val_1: " + str(len(df_temp_val_1)))
    print("len of val_2: " + str(len(df_temp_val_2)))
    print("len of train: " + str(len(df_temp_train)))
    print("len of val_0 + val_1 + val_2 + train: " + str(len(df_temp_val_0) + len(df_temp_val_1) + len(df_temp_val_2) + len(df_temp_train)) )

    # remove all nan form df_temp_train
    # remove index of df_temp_val_0
    df_temp_train = df_temp_train.dropna()
    df_temp_train = df_temp_train.reset_index(drop=True )
    df_temp_val_0 = df_temp_val_0.reset_index(drop=True)
    df_temp_val_1 = df_temp_val_1.reset_index(drop=True)
    df_temp_val_2 = df_temp_val_2.reset_index(drop=True)

    return [df_temp_train, df_temp_val_0, df_temp_val_1, df_temp_val_2]

def make_input_data(df, comp_list,batch_size):

    target = np.array(df.ioc[:,"y0"])

    emb  = np.array(df.emb[:])
    T = np.array(df['T'])
    x = np.array(df['x'])
    solute_index = np.array(df['SMILES0'])
    solvent_index = np.array(df['SMILES1'])

    target = target.reshape(target.shape[0],1)
    T = T.reshape(T.shape[0],1)
    x = x.reshape(x.shape[0],1)
    solute_index = solute_index.reshape(solute_index.shape[0],1)
    solvent_index = solvent_index.reshape(solvent_index.shape[0],1)
    # add temp, target, x and T to a nested np array
    data = np.concatenate((target, emb, x, T, solute_index, solvent_index), axis=1) 

    return data
                
def make_batches(data, batch_size):
    # make batches of the numpy array
    num_batches = int(data.shape[0]/batch_size)
    batches = []
    for i in range(num_batches):
        batches.append(data[i*batch_size:(i+1)*batch_size])
    #take care of the remaining data
    if data.shape[0] % batch_size != 0:
        batches.append(data[num_batches*batch_size:])
    return batches

def save_batches(batches, folder_path, type, ow):
    # save batches to csv for either train or val batches are numpy arrays
    bar = pb.ProgressBar(maxval=len(batches), widgets=['Save ' + type + ': ' ,pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    bar.start()

    # create folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # see if files exist in folder and delete them if overwrite is true
    if ow:
        #file ist without folders
        for file in os.listdir(folder_path):
            if os.path.isfile(folder_path + file) and file.startswith(type):
                os.remove(folder_path + file)
                
    # see how many files start with the type
    num_files = len([name for name in os.listdir(folder_path) if name.startswith(type)])

    for i, batch in enumerate(batches):
        bar.update(i)
        file_path = os.path.join(folder_path, type + '_' + str(num_files+i) + '.csv')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        batch.to_csv(file_path, index=False)
    bar.finish()

def get_paths(save_path, vocab_path):
    if os.environ.get('XPRUN_NAME') is not None:
        file_path = "/mnt/xprun/raw_data/" 
        file_out = "/mnt/xprun/data/" + save_path + "/"
        vocab_path = "/mnt/xprun/" + vocab_path + "/"
        alias_path = "/mnt/xprun/raw_data/alias/alias_dict.npy"
    else:
        file_path = "../raw_data/" 
        file_out = "../data/" + save_path + "/"
        vocab_path = "../" + vocab_path + "/"
        alias_path = '../raw_data/alias/alias_dict_brouwer.npy'
    return file_path, file_out, vocab_path, alias_path


def revert_vocab(df, comp_list):
    # look up the index in comp_list and return the smiles string first index is the solute second the solvent
    df['Solute'] = comp_list.loc[df['solute_index'].to_numpy().astype(int), 'SMILE0'].tolist()
    df['Solvent'] = comp_list.loc[df['solvent_index'].to_numpy().astype(int), 'SMILE0'].tolist()
    return df

def revert_vocab_index(df, comp_list):
    # look up the index in comp_list and return the smiles string first index is the solute second the solvent
    df['Solute'] = comp_list.loc[df['x_index'].to_numpy().astype(int), 'SMILE0'].tolist()
    df['Solvent'] = comp_list.loc[df['x_index'].to_numpy().astype(int), 'SMILE0'].tolist()
    return df

if __name__ == "__main__":
    main()
