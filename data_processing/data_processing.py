from ntpath import join
import pandas as pd
import os
import csv
import progressbar as pb
import numpy as np

from pandas.core.frame import DataFrame

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
    with open(file_path + '\\' +vocab_name + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        vocab_dict = {}
        for row in csv_reader:
            vocab_dict[row[0]] = int(row[1])
        
    return vocab_dict

def get_smiles(file_path):
    # smile0 is the name of all files in file path
    list_smile0 = pd.DataFrame(os.listdir(file_path))
    #read first csv in file path
    smile1 = pd.read_csv(file_path+'/'+list_smile0.iloc[0][0])
    #smile1 is the second column
    list_smile1 = pd.DataFrame(smile1.iloc[:,1])
    return list_smile0, list_smile1 

def get_smiles_test_val(list_smile0, list_smile1,frac):
    # creates a dictorary where the key is a smile and the value is a bool if the smile is not in the test set
    val_dict = {}
    list_smile0_test = list_smile0.sample(frac=frac, random_state=42)
    list_smile1_test = list_smile1.sample(frac=frac, random_state=42)

    for i in range(list_smile0_test.shape[0]):
        val_dict[list_smile0_test.iloc[i][0]] = False
    for i in range(list_smile1_test.shape[0]):
        val_dict[list_smile1_test.iloc[i][0]] = False
    return val_dict

def load_data_test_val(folder_path, val_dict):
    files = os.listdir(folder_path)
    #load all files into a panda
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    print("load datasets")
    bar = pb.ProgressBar(maxval=len(files), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    for i, file in enumerate(files):
        bar.update(i)
        file_path = os.path.join(folder_path, file)
        temp_df = pd.read_csv(file_path, sep=',', index_col=None)
        #add to validation set if colum 0 or 1 is in val_dict else add to training set
        temp_df_val = temp_df.loc[(temp_df.iloc[:,0].isin(val_dict.keys())) | (temp_df.iloc[:,1].isin(val_dict.keys()))]
        temp_df_train = temp_df.loc[~temp_df.index.isin(temp_df_val.index)]
        df_train = df_train.append(temp_df_train)
        df_val = df_val.append(temp_df_val)
    return df_train, df_val

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

def apply_vocab(df, vocab_dict):
    # apply the vocab to the dataframe and padd the data
    bar = pb.ProgressBar(maxval=df.shape[0], widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    temp = np.zeros([df.shape[0], 128])
    remove_index = []
    padd_char = list(vocab_dict.keys())[0]
    for i in range(df.shape[0]):
        bar.update(i)
        # if data longer than 128 chars then add to remove index
        if len(df.iloc[i][0]) > 128:
            remove_index.append(i)
        else:
            # padd to 128 with 0
            text = df.iloc[i,0].ljust(128, padd_char)
            temp[i,:] = [vocab_dict [char] for char in text]
    
    data = np.zeros([df.shape[0], 1])
    data = np.array(df.iloc[:,1])
    # make data an 2d array
    data = data.reshape(data.shape[0],1)
    data = -data
    # add data and temp to as single array
    data = np.concatenate((data, temp), axis=1)
    #negate data
    
    # remove the rows that are too long
    data = np.delete(data, remove_index, axis=0)
    # remove all data where gamma < -10 or > 10
    data = data[np.logical_and(data[:,0] > -10, data[:,0] < 10)]
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


def save_batches(batches, folder_path, type):
    # save batches to csv for either train or val batches are numpy arrays
    bar = pb.ProgressBar(maxval=len(batches), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
    for i, batch in enumerate(batches):
        bar.update(i)
        file_path = os.path.join(folder_path, type + '_' +str(i) + '.csv')
        #formate first row float all other rows int
        fmt = '%.5f'+',%.0f'*(batch.shape[1]-1)
        np.save(file_path, batch)


if __name__ == "__main__":

    file_path = "InputData/"
    file_out = "data"

    # make os path

    print("Data Loading")
    vocab_dict = load_vocab('Vocab','vocab_dict_full')
    list_smile0, list_smile1  = get_smiles(file_path)
    val_dict = get_smiles_test_val(list_smile0, list_smile1, 0.2)
    df_train,df_val = load_data_test_val(file_path, val_dict)
    df_train_joined = join_input_data(df_train, vocab_dict)
    df_val_joined = join_input_data(df_val, vocab_dict)
    print("Data Loaded")
    
    # apply vocab
    print("Applying Vocab")
    df_train_joined = apply_vocab(df_train_joined, vocab_dict)
    df_val_joined = apply_vocab(df_val_joined, vocab_dict)
    #save batches
    print("Saving Batches")
    batches = make_batches(df_train_joined, 100000)
    save_batches(batches, file_out, "train")
    batches = make_batches(df_val_joined, 100000)
    save_batches(batches, file_out, "val")
