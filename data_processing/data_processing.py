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

@click.option('--file_path', default="D", help='Location of raw data')
@click.option('--save_path', default="exp_D", help='Location of output data')
@click.option('--vocab_path', default="vocab", help='Location of vocab')
@click.option('--ow', default=True, help='overwirte exising files in the save folder or add to them ')

@click.option('--ul', default=np.inf, help='upper limit of gamma')
@click.option('--ll', default=-np.inf, help='lower limit of gamma')
@click.option('--frac', default=0.05, help='fraction of data to be used for testing and validation')

@click.option('--aug', default=True, help='augment the smile data')
@click.option('--seed', default=42, help='seed of the smile sampling for validation')

def main(file_path, save_path, vocab_path, ul, ll, frac, aug, seed, ow):
    processing(file_path, save_path, vocab_path, ul, ll, frac, aug, seed, ow)


def processing(file_path, save_path, vocab_path, ul, ll, frac, aug, seed, ow):
    
    if os.environ.get('XPRUN_NAME') is not None:
        file_path = "/mnt/xprun/raw_data/" + file_path + "/"
        file_out = "/mnt/xprun/data/" + save_path + "/"
        vocab_path = "/mnt/xprun/" + vocab_path + "/"
        alias_path = "/mnt/xprun/raw_data/alias/alias_dict.npy"
    else:
        file_path = "../raw_data/" + file_path + "/"
        file_out = "../data/" + save_path + "/"
        vocab_path = "../" + vocab_path + "/"
        alias_path = '../raw_data/alias/alias_dict.npy'

    # make os path

    print("Data Loading")
    
    vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')
    df_join, comp_list, solvent_indx, solute_indx  = load_exp_data(file_path) 
        
    if aug:
        comp_list = aug_data(comp_list, alias_path=alias_path)
    
    ## apply the vocab to the smiles
    comp_list = apply_vocab(comp_list, vocab_dict)
    
    val_solvent_indx, val_solute_indx = get_idx_test_val(solvent_indx, solute_indx, frac, seed)

    df_list = split_data_test_val_exp(df_join, val_solvent_indx, val_solute_indx, comp_list, seed)

    # make input data
    for i, df in enumerate(df_list):
        if not df.empty:
            data = make_input_data(df, comp_list)
            data_batches = make_batches(data, batch_size=100000)
            if i == 0:
                prefix = 'train'
            else:
                prefix = 'val_' + str(i-1)
            save_batches(data_batches, file_out, prefix, ow)

def load_exp_data(path):
    #load the data from the experiment 
    df = pd.DataFrame()
    # list all files in the folder
    files = os.listdir(path)
    # load all files into a panda
    for i, file in enumerate(files):
        print("Processing file: ", i, "/", len(files), end="\r")
        file_path = os.path.join(path, file)
        temp_df = pd.read_csv(file_path, sep=',', index_col=None)
        df = pd.concat([df, temp_df], ignore_index=True)

    # remove all nan rows
    df = df.dropna()

    solvent_list = df['solvent'].drop_duplicates()
    solute_list = df['solute'].drop_duplicates()

    complete_list = pd.concat([solute_list, solute_list])
    complete_list = complete_list.drop_duplicates()
    complete_list.reset_index(drop=True, inplace=True)
    complete_list = pd.DataFrame({'SMILE0':complete_list })

    # get the index of the solven_list and solute_list
    solvent_indx = complete_list.index[complete_list['SMILE0'].isin(solvent_list)]
    solute_indx = complete_list.index[complete_list['SMILE0'].isin(solute_list)]

    return df, complete_list, solvent_indx, solute_indx

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

    #add the alias to the new collumn alias to comp_list
    for i in range(len(comp_list)):
        for j, alias in enumerate(alias_dict[comp_list.loc[i,'SMILE0']]):
            comp_list.loc[i,'SMILE' +str(j+1)] = alias
    return comp_list

def apply_vocab(comp_list, vocab_dict):
    
    # add a collumn for each SMILE and alias with the embedding
    for j, columns in enumerate(comp_list.columns):
        for i in range(len(comp_list)):
            if i == 0:
                comp_list.insert(comp_list.shape[1],'emb'+str(j),np.nan)
                comp_list['emb' + str(j)] = comp_list['emb'+str(j)].astype(object, copy=True)
            if not pd.isnull(comp_list.loc[i,columns]):
                if comp_list.loc[i,columns] == 'O':
                    comp_list.at[i,'emb'+str(j)] = vocab_dict['H2O']
                else:
                    comp_list.at[i,'emb'+str(j)] = [vocab_dict [char] for char in comp_list.loc[i,columns]]            
    return comp_list

def get_idx_test_val(solvent_indx, solute_indx, frac, seed): 
    np.random.seed(seed)
    val_solvent_indx= np.random.choice(solvent_indx, int(frac*len(solvent_indx)), replace=False, )
    np.random.seed(seed)
    val_solute_indx= np.random.choice(solute_indx, int(frac*len(solute_indx)), replace=False, )

    return val_solvent_indx, val_solute_indx

def split_data_test_val_exp(df_join, val_solvent_indx, val_solute_indx, comp_list,seed):
    
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

def make_input_data(df, comp_list):

    bar = pb.ProgressBar(maxval=len(df), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    bar.start()
    data = []
    for i in range(len(df)):
        bar.update(i)
        # find number of aliases
        solvent = df.iloc[i,1]
        solute = df.iloc[i,0]
        
        solute_index = comp_list['SMILE0'].isin([solute])
        solvent_index = comp_list['SMILE0'].isin([solvent])

        num_alias_solute = int((~comp_list.loc[solute_index].isna()).sum().sum() / 2)
        num_alias_solvent = int((~comp_list.loc[solvent_index].isna()).sum().sum() / 2)

        for j in range(num_alias_solute):
            for k in range(num_alias_solvent):
                    
                    solute_emb = np.asarray(comp_list.loc[solute_index]['emb'+str(k)].values[0])
                    solvent_emb = np.asarray(comp_list.loc[solvent_index]['emb'+str(k)].values[0])
                    
                    value = np.zeros(131)
                    value[0] = np.array(df.loc[i,'lnGamma'])
                    value = np.array(df.loc[i,'lnGamma'])
                    value = np.append(value, 1)
                    value = np.append(value, solute_emb)
                    value = np.append(value, 2)
                    value = np.append(value, solvent_emb)
                    value = np.append(value, 3)
                    value = np.append(value, np.zeros(129 - len(value)))
                    value = np.append(value, df.loc[i,'x'])
                    value = np.append(value, df.loc[i,'T'])
                    value = np.append(value, solute_index)
                    value = np.append(value, solvent_index)

                
                    if i == 0:
                        data = np.array(value)
                    else:
                        data = np.vstack((data,value))
    pass
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
    bar = pb.ProgressBar(maxval=len(batches), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    bar.start()

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
        np.save(file_path, batch)


if __name__ == "__main__":
    main()
