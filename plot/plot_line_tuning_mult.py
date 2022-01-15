import numpy as np
import plot_results as pr
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('data_processing/')

from data_processing import get_comp_list 

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

if __name__ == '__main__':

    name = '211209-214402'
    name = '211214-125306' # model without aug
    name = '211223-032657' # modle with aug
    name = '211231-031659' # modle with leave n out no water 
    name = 'f_t_211220-192228_220112-105727'
    name = 'f_t_211220-192228_220114-185541' # V2 run

    group = True
    scatter = True

    comp_list_path = '/home/bene/NNGamma/data/data_exp_noH2O_1000/0/comp_list.csv'
    comp_lsit_path = '/home/bene/NNGamma/data/data_exp_onlyH2O_1000_V2/0/comp_list.csv'
    path_temp = '/home/bene/NNGamma/out_fine_tune/'
    plot_path = '/home/bene/NNGamma/src/plot/'
    data_path = path_temp + name + '/'

    file_path = ["brouwer_exp_c"]
    vocab_path = "vocab"

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
        
        val_input_0 = val_0['input'].to_numpy()
        val_input_1 = val_1['input'].to_numpy()
        val_input_2 = val_2['input'].to_numpy()

    print('data loaded')
    print('length val0: ' + str(len(val_predction_0)))
    print('length val1: ' + str(len(val_predction_1)))
    print('length val2: ' + str(len(val_predction_2)))

    pr.make_heatmap(val_predction_0, val_target_0, r'val0' , path = plot_path, save=True)
    pr.make_heatmap(val_predction_1, val_target_1, r'val1' , path = plot_path, save=True)
    pr.make_heatmap(val_predction_2, val_target_2, r'val2' , path = plot_path, save=True)

    print('heatmaps plotted')

    pr.make_historgam_delta(val_predction_0, val_target_0, 'val_0_fine' , path = plot_path, save=True)
    pr.make_historgam_delta(val_predction_1, val_target_1, 'val_1_fine' , path = plot_path, save=True)
    pr.make_historgam_delta(val_predction_2, val_target_2, 'val_2_fine' , path = plot_path, save=True)

    pr.plot_err_sorted(val_predction_0, val_target_0, r'val_0_fine' , path = plot_path, save=True)
    pr.plot_err_sorted(val_predction_1, val_target_1, r'val_1_fine' , path = plot_path, save=True)
    pr.plot_err_sorted(val_predction_2, val_target_2, r'val_2_fine' , path = plot_path, save=True)


    comp_list, systems, df_join = get_comp_list(file_path, vocab_path)
    h2o_index = df_join[(df_join.solvent == 'O') | (df_join.solute == 'O')].i
    
    ## seperate prediction and target data
    # chekc if val_input_0 continas h2o_index

    h2o_val_0_index = np.where(np.in1d(val_input_0, h2o_index))[0]
    h2o_val_1_index = np.where(np.in1d(val_input_1, h2o_index))[0]
    h2o_val_2_index = np.where(np.in1d(val_input_2, h2o_index))[0]

    h2o_val_0_pred = val_predction_0[h2o_val_0_index]
    h2o_val_0_target = val_target_0[h2o_val_0_index]

    h2o_val_1_pred = val_predction_1[h2o_val_1_index]
    h2o_val_1_target = val_target_1[h2o_val_1_index]

    h2o_val_2_pred = val_predction_2[h2o_val_2_index]
    h2o_val_2_target = val_target_2[h2o_val_2_index]

    no_h2o_val_0_index = np.where(np.in1d(val_input_0, h2o_index, invert=True))[0]
    no_h2o_val_1_index = np.where(np.in1d(val_input_1, h2o_index, invert=True))[0]
    no_h2o_val_2_index = np.where(np.in1d(val_input_2, h2o_index, invert=True))[0]

    no_h2o_val_0_pred = val_predction_0[no_h2o_val_0_index]
    no_h2o_val_0_target = val_target_0[no_h2o_val_0_index]
    
    no_h2o_val_1_pred = val_predction_1[no_h2o_val_1_index]
    no_h2o_val_1_target = val_target_1[no_h2o_val_1_index]

    no_h2o_val_2_pred = val_predction_2[no_h2o_val_2_index]
    no_h2o_val_2_target = val_target_2[no_h2o_val_2_index]    

    max = int(2e5)
    if scatter:
        pr.make_scatter(val_predction_0, val_target_0, name = 'val_0', path = plot_path, save=True)
        print('val_0_fine done')
        pr.make_scatter(val_predction_1[0:max], val_target_1[:max], name = 'val_1', path = plot_path, save=True)
        print('val_1_fine done')
        pr.make_scatter(val_predction_2[:max], val_target_2[:max], name = 'val_2', path = plot_path, save=True)
        print('val_2_fine done')
        if h2o_val_0_target != []:
            pr.make_scatter(h2o_val_0_pred, h2o_val_0_target, name = 'h2o_val_0', path = plot_path, save=True)
        print('h2o_val_0_fine done')
        pr.make_scatter(h2o_val_1_pred[0:max], h2o_val_1_target[:max], name = 'h2o_val_1', path = plot_path, save=True)
        print('h2o_val_1_fine done')
        pr.make_scatter(h2o_val_2_pred[:max], h2o_val_2_target[:max], name = 'h2o_val_2', path = plot_path, save=True)
        print('h2o_val_2_fine done')
        pr.make_scatter(no_h2o_val_0_pred, no_h2o_val_0_target, name = 'no_h2o_val_0', path = plot_path, save=True)
        print('no_h2o_val_0_fine done')
        pr.make_scatter(no_h2o_val_1_pred[0:max], no_h2o_val_1_target[:max], name = 'no_h2o_val_1', path = plot_path, save=True)
        print('no_h2o_val_1_fine done')
        pr.make_scatter(no_h2o_val_2_pred[:max], no_h2o_val_2_target[:max], name = 'no_h2o_val_2', path = plot_path, save=True)
        print('no_h2o_val_2_fine done')
        print('scatter plots made')
  


    # load cvs with data from COSMO


    cosmo_data = pd.read_csv(path_temp + 'BROUWER-COSMO-OUT.csv', sep=';')
    cosmo_data = cosmo_data.dropna()

    cosmo_sac_data = pd.read_csv(path_temp + 'COSMO-SAC.csv', sep=';')
    cosmo_sac_data = cosmo_sac_data.dropna()

    UNIFAC_data = pd.read_csv(path_temp + 'UNIFAC_out.csv', sep=';')
    UNIFAC_data = UNIFAC_data.dropna()
    
    damay_data = pd.read_csv(path_temp + 'data_Damay_et.al.csv', sep=';')
    #here we have to fake the Damay data with the same format as the COSMO data
    damay_data = damay_data.to_numpy()
    damay_data = damay_data * 1000
    damay_data_target = np.zeros(1000)
    damay_data_prediction = np.zeros(1000)
    for i in range(len(damay_data)):
        value = 0.2 * i - 1.6
        damay_data_target[int(np.sum(damay_data[0:i])):int(np.sum(damay_data[0:i+1]))] = value
    damay_data_target[int(np.sum(damay_data)):] = 9999

    cosmo_data_target = cosmo_data['lnGamma_exp'].to_numpy(dtype=np.float64)
    cosmo_data_prediction = cosmo_data['lnGamma'].to_numpy(dtype=np.float64)
    pr.make_scatter(cosmo_data_prediction, cosmo_data_target, name = 'cosmo', path = plot_path, save=True)

    i_val_0 = set(val_0['input'].to_numpy())
    i_val_1 = set(val_1['input'].to_numpy())
    i_val_2 = set(val_2['input'].to_numpy())
    i_cosmo = set(cosmo_data['i'].to_numpy())
    i_cosmo_sac = set(cosmo_sac_data['i'].to_numpy())
    i_UNIFAC = set(UNIFAC_data['i'].to_numpy())

    # find common i's
    i_common = i_UNIFAC.intersection(i_val_0).intersection(i_val_1).intersection(i_val_2).intersection(i_cosmo).intersection(i_cosmo_sac)

    # filter data
    cosmo_data_target = cosmo_data[cosmo_data['i'].isin(i_common)]['lnGamma_exp'].to_numpy(dtype=np.float64)
    cosmo_data_prediction = cosmo_data[cosmo_data['i'].isin(i_common)]['lnGamma'].to_numpy(dtype=np.float64)
    
    UNIFAC_data_target = UNIFAC_data[UNIFAC_data['i'].isin(i_common)]['lnGamma_exp'].to_numpy(dtype=np.float64)
    UNIFAC_data_prediction = UNIFAC_data[UNIFAC_data['i'].isin(i_common)]['lnGamma_UNIFAC'].to_numpy(dtype=np.float64)

    cosmo_sac_data_target = cosmo_sac_data[cosmo_sac_data['i'].isin(i_common)]['lnGamma_exp'].to_numpy(dtype=np.float64)
    cosmo_sac_data_prediction1 = cosmo_sac_data[cosmo_sac_data['i'].isin(i_common)]['lnGamma_SAC'].to_numpy(dtype=np.float64)
    cosmo_sac_data_prediction3 = cosmo_sac_data[cosmo_sac_data['i'].isin(i_common)]['lnGamma_SAC3'].to_numpy(dtype=np.float64)

    val_predction_0 = val_0[val_0['input'].isin(i_common)]['prediction'].to_numpy(dtype=np.float64)
    val_predction_1 = val_1[val_1['input'].isin(i_common)]['prediction'].to_numpy(dtype=np.float64)
    val_predction_2 = val_2[val_2['input'].isin(i_common)]['prediction'].to_numpy(dtype=np.float64)

    val_target_0 = val_0[val_0['input'].isin(i_common)]['target'].to_numpy(dtype=np.float64)
    val_target_1 = val_1[val_1['input'].isin(i_common)]['target'].to_numpy(dtype=np.float64)
    val_target_2 = val_2[val_2['input'].isin(i_common)]['target'].to_numpy(dtype=np.float64)

    prediction_list = [cosmo_sac_data_prediction1, cosmo_sac_data_prediction3, cosmo_data_prediction, UNIFAC_data_prediction, damay_data_prediction, val_predction_0, val_predction_1, val_predction_2]
    target_list = [cosmo_sac_data_target, cosmo_sac_data_target, cosmo_data_target, UNIFAC_data_target, damay_data_target,val_target_0, val_target_1, val_target_2]
    
    print('data filtered')
    print('length cosmo_data_prediction: ' + str(len(cosmo_data_prediction)))
    print('length cosmo_sac_data_prediction1: ' + str(len(cosmo_sac_data_prediction1)))
    print('length cosmo_sac_data_prediction3: ' + str(len(cosmo_sac_data_prediction3)))
    print('length val_predction_0: ' + str(len(val_predction_0)))
    print('length val_predction_1: ' + str(len(val_predction_1)))
    print('length val_predction_2: ' + str(len(val_predction_2)))
    
    color_list = ['lightcoral', 'indianred', 'brown', 'red', 'coral','lightsteelblue', 'cornflowerblue', 'royalblue']
    name_list = ['COSMO-SAC$_{2002}$', 'COSMO-SAC$_{dsp}$', 'COSMO-RS$_{TZVDP-F}$', 'UNIFAC$_{Dortmund}$', '\emph{Damay et al. 2021*}','SMILE2P$_{val_0}$', 'SMILE2P$_{val_1}$', 'SMILE2P$_{val_2}$']
    pr.make_historgam_delta_mult(prediction_list, target_list, name_list, path = plot_path, save=False, color_list = color_list)



    
