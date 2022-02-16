from difflib import diff_bytes
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import progressbar as pb
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import data_processing as dp    


def load_data(file_path, val_type):

    target_list = []
    prediction_list = []
    mse_list = []
    input_list = []

    file_list = os.listdir(file_path)
    # sort
    file_list.sort()
    for files in file_list:
        if files.startswith('val_target_'+ val_type + '_'):
            target_list.append(np.load(file_path + files))
        if files.startswith('val_predction_'+ val_type + '_'):
            prediction_list.append(np.load(file_path + files))
        if files.startswith('val_input_'+ val_type + '_'):
            input_list.append(np.load(file_path + files))
    for i in range(0, len(target_list)):
        mse = np.mean(np.square(target_list[i] - prediction_list[i]))
        mse_list.append(mse)

    return target_list, prediction_list, mse_list, input_list

file_path = ["brouwer_exp_c"]
file_path = ["inf_cosmo"]
vocab_path = "vocab"
path_temp = '/home/bene/NNGamma/out_fine_tune/'
plot_path = '/home/bene/NNGamma/src/plot/'

name = 'f_t_211220-192228_220114-185541' # V2 run

data_path = path_temp + name + '/'



comp_list, systems, df_join = dp.get_comp_list(file_path, vocab_path)

vocab_path = "../vocab/"
vocab_dict = dp.load_vocab(vocab_path,'vocab_dict_aug')

solvent_list = df_join.solvent.values
solute_list = df_join.solute.values

# coung the occurence of each char in the solvents and solutes
solvent_count = np.zeros(len(vocab_dict))
solute_count = np.zeros(len(vocab_dict))


for i in range(len(solvent_list)):
    for j in range(len(solvent_list[i])):
        solvent_count[vocab_dict[solvent_list[i][j]]] += 1

for i in range(len(solute_list)):
    for j in range(len(solute_list[i])):
        solute_count[vocab_dict[solute_list[i][j]]] += 1


# plot the count of each char label with the corresponding char
plt.figure(figsize=(20,10))
plt.bar(np.arange(len(vocab_dict)), solvent_count, color='b', label='solvent')
plt.bar(np.arange(len(vocab_dict)), solute_count, color='r', label='solute')
plt.xticks(np.arange(len(vocab_dict)), list(vocab_dict.keys()))
plt.legend()
plt.show()
# print the occurence of each char in the solvents and solutes in a table
print('\n')
print('Solvent')
print(pd.DataFrame(list(vocab_dict.keys()), columns=['char']))
print(pd.DataFrame(solvent_count, columns=['count']))
print('\n')
print('Solute')
print(pd.DataFrame(list(vocab_dict.keys()), columns=['char']))
print(pd.DataFrame(solute_count, columns=['count']))
print('\n')


# save the figure
plt.savefig('char_occurance.png')

type_list = ['0', '1', '2']

val_target_0, val_predction_0, mse_list_0, val_input_0 = load_data(data_path, type_list[0])
val_target_1, val_predction_1, mse_list_1, val_input_1 = load_data(data_path, type_list[1])
val_target_2, val_predction_2, mse_list_2, val_input_2 = load_data(data_path, type_list[2])

val_target_0 = np.concatenate(val_target_0)
val_input_0 = np.concatenate(val_input_0)
val_predction_0 = np.concatenate(val_predction_0)

val_target_1 = np.concatenate(val_target_1)
val_input_1 = np.concatenate(val_input_1)
val_predction_1 = np.concatenate(val_predction_1)

val_target_2 = np.concatenate(val_target_2)
val_input_2 = np.concatenate(val_input_2)
val_predction_2 = np.concatenate(val_predction_2)

# match smiles in systems and input
results = pd.DataFrame(columns=['solvent', 'solute', 'mse'])
bar = pb.ProgressBar(maxval=len(val_input_0), widgets=['Gathering Results', pb.Bar('=', '[', ']'), ' ', pb.Percentage(), pb.ETA()])
bar.start()
for i, inpt in enumerate(val_input_0):
    bar.update(i)
    solvent = df_join.solvent[df_join.i == inpt].values[0]
    solute = df_join.solute[df_join.i == inpt].values[0]
    mse = np.square(val_target_0[i] - val_predction_0[i])
    results = results.append({'solvent': solvent, 'solute': solute, 'mse': mse}, ignore_index=True)

bar.finish()
# get the man error for each charater in solvent and solute

error = [[] for _ in range(len(vocab_dict))]

bar = pb.ProgressBar(maxval=len(results), widgets=['Gathering Errors', pb.Bar('=', '[', ']'), ' ', pb.Percentage(), pb.ETA()])
bar.start()
for i in range(len(results)):
    for j in range(len(vocab_dict)):
        if list(vocab_dict.keys())[j] in results.solvent[i] or list(vocab_dict.keys())[j] in results.solute[i]: 
            error[j].append(results.mse[i])

mean_error = [np.mean(error[i]) for i in range(len(vocab_dict))]


# plot the error for each char 


plt.figure(figsize=(20,10))
plt.bar(np.arange(len(vocab_dict)), mean_error, color='b')
plt.xticks(np.arange(len(vocab_dict)), list(vocab_dict.keys()))
plt.show()
plt.savefig('char_error.png')




