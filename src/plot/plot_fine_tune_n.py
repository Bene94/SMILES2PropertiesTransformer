import numpy as np
import plot_results as pr
import pandas as pd


path_temp = '/home/bene/NNGamma/out_fine_tuen/'
plot_path = '/home/bene/NNGamma/src/'

group = True


n_list = [10, 50, 100, 500, 1000]

mse_list_0 = []
mse_list_1 = []
mse_list_2 = []

mea_list_0 = []
mea_list_1 = []
mea_list_2 = []

for n in n_list:
    #load data

    name = 'n_f_' + str(n)

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

        mse_0 = np.square(val_target_0 - val_predction_0)
        mse_1 = np.square(val_target_1 - val_predction_1)
        mse_2 = np.square(val_target_2 - val_predction_2)

        mea_0 = np.abs(val_target_0 - val_predction_0)
        mea_1 = np.abs(val_target_1 - val_predction_1)
        mea_2 = np.abs(val_target_2 - val_predction_2)

        mse_list_0.append(mse_0)
        mse_list_1.append(mse_1)
        mse_list_2.append(mse_2)

        mea_list_0.append(mea_0)
        mea_list_1.append(mea_1)
        mea_list_2.append(mea_2)


## Plot boxplots

pr.plot_boxplot(n_list, mse_list_0, mse_list_1, mse_list_2, 'MSE', path=plot_path, save=True)
pr.plot_boxplot(n_list, mea_list_0, mea_list_1, mea_list_2, 'MAE', path=plot_path, save=True)

