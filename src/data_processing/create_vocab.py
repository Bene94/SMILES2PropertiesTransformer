import pandas as pd
import os
## Load the text files out of InputData into a pandas dataframe

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
    df = df.dropna()
    for i in range(df.shape[0]):
        # add progress bar
        print("Processing row: ", i, "/", df.shape[0], end="\r")
        text = df.iloc[i][0]
        text = text+df.iloc[i][1]
        #find unique characters in the text
        for char in text:
            vocab.add(char)
    return vocab

def create_vocab_dict(vocab):
    # create a dictionary for the vocab with the char being the key and the value being the index       
    vocab_dict = {}
    
    # add special characters to the vocab dictionary 
    vocab_dict['&'] = 0 # for padding
    vocab_dict['!'] = 1 # for start of sentence
    vocab_dict['?'] = 2 # for joining
    vocab_dict['*'] = 3 # for end of sentence

    for i, char in enumerate(vocab):       
        vocab_dict[char] = i + 4

    return vocab_dict

if __name__ == "__main__":
    #add consol output to show progress
    df, num_smile0, num_smile1 = load_data('../raw_data/exp/')
    print("Find Vocab")
    vocab = find_vocab(df)
    vocab = sorted(vocab)
    print("Create Vocab Dict")
    vocab_dict = create_vocab_dict(vocab)
    # save vocab_dict into TrainingData\
    with open('Vocab/vocab_dict_exp.csv', 'w') as f:
        for key, value in vocab_dict.items():
            f.write(key + ' ' + str(value) + '\n')
        

