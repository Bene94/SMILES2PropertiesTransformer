import numpy as np
import plot_results as pr
import pandas as pd
from plot_line_tuning_mult import load_data


name = '211209-214402'
name = '211223-032657' # modle with aug
name = '211229-043706' # modle with leave n out

path_temp = '/home/bene/NNGamma/out_fine_tuen/'
plot_path = '/home/bene/NNGamma/src/'
save_path = '/home/bene/NNGamma/out_fine_tuen/' 
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

train_df = pd.DataFrame(data={  'out':val_predction_2, 'target':val_target_2, 'x_index':val_input_2})

train_df = pd.DataFrame(data={  'out':val_predction_1, 'target':val_target_1, 'x_index':val_input_1})


print("Data Loading")

train_df.to_excel(save_path + name + '.xlsx', index=False)
# save high error data to excel
#df = df[ np.abs(df['out']-df['target']) > 0.5]
#df.to_excel(save_path + name + '_high_error.xlsx', index=False)


