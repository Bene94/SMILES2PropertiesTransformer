import numpy as np
import plot_results as pr
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path, type):

    target_list = []
    prediction_list = []
    mse_list = []
    input_list = []

    for i in range(0, 80):
        target_list.append(np.load(file_path + 'val_target_'+ type + '_' + str(i) + '.npy'))
        prediction_list.append(np.load(file_path + 'val_predction_'+ type + '_' + str(i) + '.npy'))
        input_list.append(np.load(file_path + 'val_input_'+ type + '_' + str(i) + '.npy'))
        mse = np.mean(np.square(target_list[i] - prediction_list[i]))
        mse_list.append(mse)

    return target_list, prediction_list, mse_list, input_list

if __name__ == '__main__':

    name = '211209-214402'
    name = '211214-125306' # model without aug
    name = '211223-032657' # modle with aug
    name = '211229-043706' # modle with leave n out

    group = False

    path_temp = '/home/bene/NNGamma/out_fine_tuen/'
    plot_path = '/home/bene/NNGamma/src/'
    data_path = path_temp + name + '/'
    #path_temp = '../temp/'

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

    mse_0 = np.mean(mse_list_0)
    mea_0 = np.mean(np.abs(val_target_0-val_predction_0))

    mse_1 = np.mean(mse_list_1)
    mea_1 = np.mean(np.abs(val_target_1-val_predction_1))

    mse_2 = np.mean(mse_list_2)
    mea_2 = np.mean(np.abs(val_target_2-val_predction_2))

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

    print('data loaded')
    print('length val0: ' + str(len(val_predction_0)))
    print('length val1: ' + str(len(val_predction_1)))
    print('length val2: ' + str(len(val_predction_2)))

    pr.make_heatmap(val_predction_0, val_target_0, 'val_0_fine' , path = plot_path, save=True)
    pr.make_heatmap(val_predction_1, val_target_1, 'val_1_fine' , path = plot_path, save=True)
    pr.make_heatmap(val_predction_2, val_target_2, 'val_2_fine' , path = plot_path, save=True)

    print('heatmaps plotted')

    pr.make_historgam_delta(val_predction_0, val_target_0, 'val_0_fine' , path = plot_path, save=True)
    pr.make_historgam_delta(val_predction_1, val_target_1, 'val_1_fine' , path = plot_path, save=True)
    pr.make_historgam_delta(val_predction_2, val_target_2, 'val_2_fine' , path = plot_path, save=True)

    max = int(2e5)

    pr.make_scatter(val_predction_0, val_target_0, name = 'val_0', path = plot_path, save=True)
    print('val_0_fine done')
    pr.make_scatter(val_predction_1[0:max], val_target_1[:max], name = 'val_1', path = plot_path, save=True)
    print('val_1_fine done')
    pr.make_scatter(val_predction_2[:max], val_target_2[:max], name = 'val_2', path = plot_path, save=True)
    print('val_2_fine done')
    print('scatter plots made')

    # load cvs with data from COSMO

    cosmo_data = pd.read_csv(path_temp + 'BROUWER-COSMO-OUT.csv', sep=';')
    cosmo_data = cosmo_data.dropna()

    cosmo_data_target = cosmo_data['lnGamma_exp'].to_numpy(dtype=np.float64)
    cosmo_data_prediction = cosmo_data['lnGamma'].to_numpy(dtype=np.float64)

    prediction_list = [cosmo_data_prediction, val_predction_0, val_predction_1, val_predction_2]
    target_list = [cosmo_data_target, val_target_0, val_target_1, val_target_2]
    name_list = ['cosmo', 'val_0', 'val_1', 'val_2']

    pr.make_historgam_delta_mult(prediction_list, target_list, name_list, path = '', save=False)



    
