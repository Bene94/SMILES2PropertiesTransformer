import os
import sys

# add current path to sys.path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd


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


name = '211209-214402'
name = '211223-032657' # modle with aug
name = '211229-043706' # modle with leave n out
name = '211231-031659' # modle with leave n out no water
name = 'f_t_211220-192228_220112-105727'

path_temp = '/home/bene/NNGamma/out_fine_tune/'
plot_path = '/home/bene/NNGamma/src/'
save_path = '/home/bene/NNGamma/out_fine_tune/' 
data_path = path_temp + name + '/'



#path_temp = '../temp/'

group = True

#load data

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

if group:
    val_0 = pd.DataFrame({'input':val_input_0.squeeze(),'prediction': val_predction_0.squeeze(),'target': val_target_0.squeeze()})
    val_0 = val_0.groupby(['input']).mean()
    val_0 = val_0.reset_index()

    val_1 = pd.DataFrame({'input':val_input_1.squeeze(),'prediction': val_predction_1.squeeze(),'target': val_target_1.squeeze()})
    val_1 = val_1.groupby(['input']).mean()
    val_1 = val_1.reset_index()

    val_2 = pd.DataFrame({'input':val_input_2.squeeze(),'prediction': val_predction_2.squeeze(),'target': val_target_2.squeeze()})
    val_2 = val_2.groupby(['input']).mean()
    val_2 = val_2.reset_index()

    val_predction_0 = val_0['prediction'].to_numpy()
    val_predction_1 = val_1['prediction'].to_numpy()
    val_predction_2 = val_2['prediction'].to_numpy()

    val_target_0 = val_0['target'].to_numpy()
    val_target_1 = val_1['target'].to_numpy()
    val_target_2 = val_2['target'].to_numpy()

    val_input_0 = val_0['input'].to_numpy()
    val_input_1 = val_1['input'].to_numpy()
    val_input_2 = val_2['input'].to_numpy()


val_all_pred = np.concatenate((val_predction_0, val_predction_1, val_predction_2), axis=0)
val_all_target = np.concatenate((val_target_0, val_target_1, val_target_2), axis=0)
val_all_input = np.concatenate((val_input_0, val_input_1, val_input_2), axis=0)


print('data loaded')
print('length val0: ' + str(len(val_predction_0)))
print('length val1: ' + str(len(val_predction_1)))
print('length val2: ' + str(len(val_predction_2)))

# turn np arrays into pandas dataframes for train and val

train_df_2 = pd.DataFrame(data={  'out':val_predction_2, 'target':val_target_2, 'x_index':val_input_2})

train_df_1 = pd.DataFrame(data={  'out':val_predction_1, 'target':val_target_1, 'x_index':val_input_1})

train_df_0 = pd.DataFrame(data={  'out':val_predction_0, 'target':val_target_0, 'x_index':val_input_0})


print("Data Loading")

# save all to excel file

train_df_0.to_excel(save_path + 'train_df_0.xlsx')
train_df_1.to_excel(save_path + 'train_df_1.xlsx')
train_df_2.to_excel(save_path + 'train_df_2.xlsx')
# save high error data to excel
#df = df[ np.abs(df['out']-df['target']) > 0.5]
#df.to_excel(save_path + name + '_high_error.xlsx', index=False)


