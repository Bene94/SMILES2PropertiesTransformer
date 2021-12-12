import numpy as np
import plot_results as pr
import pandas as pd


name = '211209-214402'
path_temp = '/home/bene/NNGamma/out_fine_tuen/'
plot_path = '/home/bene/NNGamma/src/'
save_path = '/home/bene/NNGamma/out_fine_tuen/' 
#path_temp = '../temp/'

group = True

#load data
val_predction_0 = np.load(path_temp + name + '/val_predction_0.npy')
val_predction_1 = np.load(path_temp + name + '/val_predction_1.npy')
val_predction_2 = np.load(path_temp + name + '/val_predction_2.npy')

val_target_0 = np.load(path_temp + name + '/val_target_0.npy')
val_target_1 = np.load(path_temp + name + '/val_target_1.npy')
val_target_2 = np.load(path_temp + name + '/val_target_2.npy')

val_input_0 = np.load(path_temp + name + '/val_input_0.npy')
val_input_1 = np.load(path_temp + name + '/val_input_1.npy')
val_input_2 = np.load(path_temp + name + '/val_input_2.npy')

# claculate MSE and MAE for each model
mse_0 = np.mean(np.square(val_target_0 - val_predction_0))
mse_1 = np.mean(np.square(val_target_1 - val_predction_1))
mse_2 = np.mean(np.square(val_target_2 - val_predction_2))

mea_0 = np.mean(np.abs(val_target_0 - val_predction_0))
mea_1 = np.mean(np.abs(val_target_1 - val_predction_1))
mea_2 = np.mean(np.abs(val_target_2 - val_predction_2))

# print results
print('MSE_0:', '{:.2f}'.format(np.mean(mse_0)))
print('MAE_0:', '{:.2f}'.format(np.mean(mea_0)))

print('MSE_1:', '{:.2f}'.format(np.mean(mse_1)))
print('MAE_1:', '{:.2f}'.format(np.mean(mea_1)))

print('MSE_2:', '{:.2f}'.format(np.mean(mse_2)))
print('MAE_2:', '{:.2f}'.format(np.mean(mea_2)))

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


print("Data Loading")

train_df.to_excel(save_path + name + '.xlsx', index=False)
# save high error data to excel
#df = df[ np.abs(df['out']-df['target']) > 0.5]
#df.to_excel(save_path + name + '_high_error.xlsx', index=False)


