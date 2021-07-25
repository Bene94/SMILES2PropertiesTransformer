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
    for i, char in enumerate(vocab):       
        vocab_dict[char] = i
    vocab_dict['%'] = len(vocab_dict)
    vocab_dict['&'] = len(vocab_dict)
    return vocab_dict

def convert_to_index(df, vocab_dict):
    # convert the text to index       
    df.iloc[:,0] = df.iloc[:,0].apply(lambda x: [vocab_dict[i] for i in x])       
    df.iloc[:,1] = df.iloc[:,1].apply(lambda x: [vocab_dict[i] for i in x])       
    return df

def get_test_val_train(df, num_smile1,num_smile2,split_ratio):
    
    smile0 = df['SMILES0'].drop_duplicates()
    smile1 = df['SMILES'].drop_duplicates()

    smile0_val = smile0.sample(frac=split_ratio, random_state=42)
    smile1_val = smile1.sample(frac=split_ratio, random_state=42)

    #create the val set by removing all elements from df that are equl to smile1_val and smile0_val
    val_df = df[~df['SMILES0'].isin(smile0_val)]
    val_df = val_df[~val_df['SMILES'].isin(smile1_val)]

    # remove all elements in the train set that are in the val set
    train_df = df.drop(val_df.index)
    return val_df, train_df

def make_batches(df, batch_size):
    # make batches of the dataframe
    num_batches = int(df.shape[0]/batch_size)
    batches = []
    for i in range(num_batches):
        batches.append(df.iloc[i*batch_size:(i+1)*batch_size])
    return batches

#add consol output to show progress
df, num_smile0, num_smile1 = load_data('InputData')
print("Find Vocab")
vocab = find_vocab(df)
print("Create Vocab Dict")
vocab_dict = create_vocab_dict(vocab)
# save vocab_dict into TrainingData\
with open('TrainingData/vocab_dict.csv', 'w') as f:
    for key, value in vocab_dict.items():
        f.write(key + ' ' + str(value) + '\n')

#print("Split Data")
#df_val, df_train = get_test_val_train(df,num_smile0,num_smile1,0.8)
#print("Make batches")
#batches_train = make_batches(df_train,1000)
#batches_val = make_batches(df_val,1000)
#print("Convert to Index for batches")
#for i, batch in enumerate(batches_train):
 #   print("Processing batch: ", i, "/", len(batches_train), end="\r")
#    batch = convert_to_index(batch, vocab_dict)
#print("\n")
#for i, batch in enumerate(batches_val):
 #   print("Processing batch: ", i, "/", len(batches_val), end="\r")
 #   batch = convert_to_index(batch, vocab_dict)
#print("\n")
#print("Save Data")

#save the batches and in new files into TrainingData\
""" for i, batch in enumerate(batches_train):
    print("Saving batch: ", i, "/", len(batches_train), end="\r")
    batch.to_csv("TrainingData\TrainBatch"+str(i)+".csv", sep=',', index=False)

for i, batch in enumerate(batches_val):
    print("Saving batch: ", i, "/", len(batches_val), end="\r")
    batch.to_csv("TrainingData\ValBatch"+str(i)+".csv", sep=',', index=False) """
    
print("Saved all batches")

# save vocab_dict into TrainingData\
with open('TrainingData/vocab_dict.csv', 'w') as f:
    for key, value in vocab_dict.items():
        f.write(key + ' ' + str(value) + '\n')

#print(df_train.shape)
#print(df_val.shape)
#print(df.shape)

print(vocab)


