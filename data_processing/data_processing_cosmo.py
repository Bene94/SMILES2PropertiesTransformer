from ntpath import join
import pandas as pd
import os
import csv
import progressbar as pb
import numpy as np
import click
import sys


from pandas.core.frame import DataFrame

@click.command()

@click.option('--file_path', default="inf_cosmo", help='Location of raw data')
@click.option('--save_path', default="data", help='Location of output data')
@click.option('--vocab_path', default="vocab", help='Location of vocab')
@click.option('--ul', default=np.inf, help='upper limit of gamma')
@click.option('--ll', default=-np.inf, help='lower limit of gamma')
@click.option('--frac', default=0.05, help='fraction of data to be used for testing and validation')
@click.option('--cosmo', default="exp", help='is loaded data from cosmo or form experiments')
@click.option('--aug', default=False, help='augment the smile data')
@click.option('--seed', default=42, help='seed of the smile sampling for validation')
@click.option('--ow', default=True, help='overwirte exising files in the save folder or add to them ')

def main(file_path, save_path, vocab_path, ul, ll, frac, cosmo, aug, seed, ow):
    processing(file_path, save_path, vocab_path, ul, ll, frac, cosmo, aug, seed, ow)


def processing(file_path, save_path, vocab_path, ul, ll, frac, cosmo, aug, seed, ow):
    
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
    
    if cosmo == "cosmo":

        list_smile0, list_smile1  = get_smiles(file_path)
        val_dict_0, val_dict_1 = get_smiles_test_val(list_smile0, list_smile1, frac, seed)

        df_train, df_val_0, df_val_1, df_val_2  = load_data_test_val(file_path, val_dict_0, val_dict_1)
        
        df_train_joined = join_input_data(df_train, vocab_dict)
        df_val_0_joined = join_input_data(df_val_0, vocab_dict)
        df_val_1_joined = join_input_data(df_val_1, vocab_dict)
        df_val_2_joined = join_input_data(df_val_2, vocab_dict)


        print("Length train: " + str(df_train_joined.shape[0]))
        print("Length val 0: " + str(df_val_0_joined.shape[0]))
        print("Length val 1: " + str(df_val_1_joined.shape[0]))
        print("Length val 2: " + str(df_val_2_joined.shape[0]))
        print("Data Loaded")

    else :

        solvent_list, solute_list, df_join = load_exp_data(file_path) 
         
        ok = True
        while ok:
        
            val_solvent, val_solute = get_smiles_test_val_exp(solvent_list, solute_list, frac, seed)
            df_train, df_val_0, df_val_1, df_val_2  = split_data_test_val_exp(df_join, val_solvent, val_solute, seed)
            if len(df_train) > 0 and len(df_val_0) > 0 and len(df_val_1) > 0 and len(df_val_2) > 0:
                ok = False
            else:
                seed = seed + 200



        # make batches of dfs
        df_train_batches = make_batches(df_train, 30000)
        df_val_0_batches = make_batches(df_val_0, 30000)
        df_val_1_batches = make_batches(df_val_1, 30000)
        df_val_2_batches = make_batches(df_val_2, 30000)

        for i in range(len(df_train_batches)):
            print("Processing batch train: ", i, "/", len(df_train_batches), end="\r")
            if aug:
                df_train_batches[i] = argument_data(df_train_batches[i], alias_path)
            df_train_batches[i] = join_input_data_exp(df_train_batches[i], vocab_dict)
            df_train_batches[i] = apply_vocab(df_train_batches[i], vocab_dict, ul, ll)
        save_batches(df_train_batches, file_out, "train", ow)
    
        for i in range(len(df_val_0_batches)):
            print("Processing batch val0: ", i, "/", len(df_val_0_batches), end="\r")
            if aug:
                df_val_0_batches[i] = argument_data(df_val_0_batches[i], alias_path)
            df_val_0_batches[i] = join_input_data_exp(df_val_0_batches[i], vocab_dict)
            df_val_0_batches[i] = apply_vocab(df_val_0_batches[i], vocab_dict, ul, ll)
        save_batches(df_val_0_batches, file_out, "val_0", ow)
        
        for i in range(len(df_val_1_batches)):
            print("Processing batch val1: ", i, "/", len(df_val_1_batches), end="\r")
            if aug:
                df_val_1_batches[i] = argument_data(df_val_1_batches[i], alias_path)
            df_val_1_batches[i] = join_input_data_exp(df_val_1_batches[i], vocab_dict)
            df_val_1_batches[i] = apply_vocab(df_val_1_batches[i], vocab_dict, ul, ll)
        save_batches(df_val_1_batches, file_out, "val_1", ow)
        
        for i in range(len(df_val_2_batches)):
            print("Processing batch val2: ", i, "/", len(df_val_2_batches), end="\r")
            if aug:
                df_val_2_batches[i] = argument_data(df_val_2_batches[i], alias_path)
            df_val_2_batches[i] = join_input_data_exp(df_val_2_batches[i], vocab_dict)
            df_val_2_batches[i] = apply_vocab(df_val_2_batches[i], vocab_dict, ul, ll)
        save_batches(df_val_2_batches, file_out, "val_2", ow)

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
        df = df.append(temp_df)

    # remove all nan rows
    df = df.dropna()

    solvent_list = df['solvent'].unique()
    solute_list = df['solute'].unique()

    solute_list = solute_list.astype(str)
    solvent_list = solvent_list.astype(str)    
    #remove duplicates from solvent and solute list
    solvent_list = np.unique(solvent_list)
    solute_list = np.unique(solute_list)

    return solvent_list, solute_list, df

def load_data(folder_path):
    # list all files in InputData
    files = os.listdir(folder_path)
    #load all files into a panda
    df = pd.DataFrame()
    num_smile0 = len(files)
    num_smile1 = 0
    for i, file in enumerate(files): 
        print("Processing file: ", i, "/", len(files), end="\r")
        file_path = os.path.join(folder_path, file)
        temp_df = pd.read_csv(file_path, sep=',', index_col=None)
        df = df.append(temp_df)
        num_smile1 = temp_df.shape[0]
    return df, num_smile0, num_smile1

def find_vocab(df):
    # find the vocab for the text
    vocab = set()
    for i in range(df.shape[0]):
        # add progress bar
        print("Processing row: ", i, "/", df.shape[0], end="\r")
        text = df.iloc[i][0]
        text = text+df.iloc[i][1]
        for char in text:
            vocab.add(char)
    return vocab

def create_vocab_dict(vocab):
    # create a dictionary for the vocab with the char being the key and the value being the index       
    vocab_dict = {}
    for i, char in enumerate(vocab):       
        vocab_dict[char] = i       
    return vocab_dict

def load_vocab(file_path,vocab_name):
    #load the vocab from a .cvs file and adds it to a dictoinary
    with open(file_path + vocab_name + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        vocab_dict = {}
        for row in csv_reader:
            vocab_dict[row[0]] = int(row[1])
        
    return vocab_dict

def get_smiles(file_path):
    # smile0 is the name of all files in file path
    list_smile0 = pd.DataFrame(os.listdir(file_path))
    # remove .txt from the end of entries in list_smile0
    list_smile0 = list_smile0[0].str.split('.',expand=True)[0]
    #read first csv in file path
    smile1 = pd.read_csv(file_path+'/'+list_smile0.iloc[0] + '.txt')
    #smile1 is the second column
    list_smile1 = pd.DataFrame(smile1.iloc[:,1])
    return list_smile0, list_smile1 

def get_smiles_test_val(list_smile0, list_smile1, frac, seed):
    # creates a dictorary where the key is a smile and the value is a bool if the smile is not in the test set
    val_dict_0 = {}
    val_dict_1 = {}
    np.random.seed(seed)
    list_smile0_test= np.random.choice(list_smile0, int(frac*len(list_smile0)), replace=False, )
    np.random.seed(os.setegid())
    list_smile1_test= np.random.choice(list_smile1, int(frac*len(list_smile1)), replace=False, )
    
    for i in range(list_smile0_test.shape[0]):
        val_dict_0[list_smile0_test.iloc[i]] = False
    for i in range(list_smile1_test.shape[0]):
        val_dict_1[list_smile1_test.iloc[i][0]] = False
    return val_dict_0, val_dict_1 

def get_smiles_test_val_exp(solvent_list, solute_list, frac, seed):
    
    val_dict_0 = {}
    val_dict_1 = {}
    np.random.seed(seed)
    list_smile0_test= np.random.choice(solvent_list, int(frac*len(solvent_list)), replace=False, )
    np.random.seed(seed)
    list_smile1_test= np.random.choice(solute_list, int(frac*len(solute_list)), replace=False, )

    for i in range(list_smile0_test.shape[0]):
        val_dict_0[list_smile0_test[i]] = False
    for i in range(list_smile1_test.shape[0]):
        val_dict_1[list_smile1_test[i]] = False
    return val_dict_0, val_dict_1
    
def load_data_test_val(folder_path, val_dict_0, val_dict_1):
    files = os.listdir(folder_path)
    #load all files into a panda
    df_train = pd.DataFrame()
    df_val_0 = pd.DataFrame()
    df_val_1 = pd.DataFrame()
    print("load datasets")
    bar = pb.ProgressBar(maxval=len(files), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    bar.start()
    for i, file in enumerate(files):
        bar.update(i)
        file_path = os.path.join(folder_path, file)
        temp_df = pd.read_csv(file_path, sep=',', index_col=None)
        #add to validation set if colum 0 or 1 is in val_dict else add to training set

        temp_df_val_0 = temp_df.loc[(temp_df.iloc[:,0].isin(val_dict_0.keys())) & (temp_df.iloc[:,1].isin(val_dict_1.keys()))]
        temp_df_val_1 = temp_df.loc[(temp_df.iloc[:,0].isin(val_dict_0.keys())) ^ (temp_df.iloc[:,1].isin(val_dict_1.keys()))]

        temp_df_train = temp_df.loc[ ~temp_df.index.isin(temp_df_val_0.index) & ~temp_df.index.isin(temp_df_val_1.index)]
        
        df_train = df_train.append(temp_df_train)
        df_val_0 = df_val_0.append(temp_df_val_0)
        df_val_1 = df_val_1.append(temp_df_val_1)
    
    return df_train, df_val_0, df_val_1

def split_data_test_val_exp(df_join, val_solvent, val_solute, seed):
    df_temp = df_join.copy()
    #reste index
    df_temp.reset_index(drop=True, inplace=True) 
    df_temp_val_0 = df_temp.loc[(df_temp.iloc[:,0].isin(val_solvent.keys())) & (df_temp.iloc[:,1].isin(val_solute.keys()))]
    df_temp_val_1 = df_temp.loc[(df_temp.iloc[:,0].isin(val_solvent.keys())) ^ (df_temp.iloc[:,1].isin(val_solute.keys()))]
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

    return df_temp_train, df_temp_val_0, df_temp_val_1, df_temp_val_2

def join_input_data(df, vocab_dict):
    #takes the smiles and adds them together wiht a space between them for the combination 0 ,1 collume 2 is the value otherwise collume 3
    df_joined = pd.DataFrame()

    temp_df = pd.DataFrame()
    temp_df['SMILES'] = list(vocab_dict.keys())[1] + df.iloc[:,0] + list(vocab_dict.keys())[2] + df.iloc[:,1] + list(vocab_dict.keys())[3]
    temp_df['gamma'] = df.iloc[:,2]

    df_joined = df_joined.append(temp_df)

    temp_df = pd.DataFrame()
    temp_df['SMILES'] = list(vocab_dict.keys())[1] + df.iloc[:,1] + list(vocab_dict.keys())[2] + df.iloc[:,0] + list(vocab_dict.keys())[3]
    temp_df['gamma'] = df.iloc[:,3]
    df_joined = df_joined.append(temp_df)
    return df_joined

def join_input_data_exp(df_val_1, vocab_dict):
    #takes the smiles and adds them together wiht a space between them for the combination 0 ,1 collume 2 is the value otherwise collume 3
    df_joined = pd.DataFrame()

    temp_df = pd.DataFrame()
    temp_df['SMILES'] = list(vocab_dict.keys())[1] + df_val_1.iloc[:,0] + list(vocab_dict.keys())[2] + df_val_1.iloc[:,1] + list(vocab_dict.keys())[3]
    temp_df['gamma'] = df_val_1.iloc[:,2]
    
    # check if field x or T exists and read that column
    if 'x' in df_val_1.columns:
        temp_df['x'] = df_val_1.iloc[:,3]
    elif 'T' in df_val_1.columns:
        temp_df['T'] = df_val_1.iloc[:,4]

    df_joined = df_joined.append(temp_df)

    return df_joined

def argument_data(df,alias_path):

    alias_dict = np.load(alias_path, allow_pickle=True).item()

    df_new = pd.DataFrame()

    #bar = pb.ProgressBar(maxval=len(df), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    #bar.start()
    for i in range(len(df)):
        #bar.update(i)
        solute_alias = alias_dict[df['solute'].iloc[i]]
        solvent_alias = alias_dict[df['solvent'].iloc[i]]
        temp_df = df.iloc[0:1+len(solute_alias)*len(solvent_alias)].copy()
        for j in range(len(solute_alias)):
            for k in range(len(solvent_alias)):
                # add all cominations of solute and solvent to the dataframe
                temp_df.iloc[j*len(solvent_alias) + k, 0] = list(solute_alias)[j]
                temp_df.iloc[j*len(solvent_alias) + k, 1] = list(solvent_alias)[k]
                temp_df.iloc[j*len(solvent_alias) + k, 2] = df['lnGamma'].iloc[i]
            
        df_new = df_new.append(temp_df)
    return df_new

def apply_vocab(df, vocab_dict, ul, ll):
    # apply the vocab to the dataframe and padd the data
    #bar = pb.ProgressBar(maxval=df.shape[0], widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    temp = np.zeros([df.shape[0], 128])
    remove_index = []
    padd_char = list(vocab_dict.keys())[0]
    #bar.start()
    for i in range(df.shape[0]):
        #bar.update(i)
        # if data longer than 128 chars then add to remove index
        if len(df.iloc[i][0]) > 128:
            remove_index.append(i)
        else:
            # padd to 128 with 0
            text = df.iloc[i,0].ljust(128, padd_char)
            temp[i,:] = [vocab_dict [char] for char in text]
    
    
    target = np.array(df.iloc[:,1])

    # check if field x or T exists
    if 'x' in df.columns:
        x = np.array(df['x'])
    else:
        x = np.zeros(df.shape[0])
    if 'T' in df.columns:
        T = np.array(df['T'])
    else:
        T = np.ones(df.shape[0]) * 298.15

    # make data an 2d array
    target = target.reshape(target.shape[0],1)

    # add temp, target, x and T to a nested np array
    data = np.concatenate((target, temp, x.reshape(x.shape[0],1), T.reshape(T.shape[0],1)), axis=1) 
    # remove the rows that are too long
    data = np.delete(data, remove_index, axis=0)
    # remove all data where gamma < -10 or > 10
    data = data[np.logical_and(data[:,0] > ll, data[:,0] < ul)]
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
        #formate first row float all other rows int
        fmt = '%.5f'+',%.0f'*(batch.shape[1]-1)
        #crate foler if not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(file_path, batch)


if __name__ == "__main__":
    main()
