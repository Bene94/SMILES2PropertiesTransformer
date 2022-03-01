from operator import truediv
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

    name = 'f_t_211220-192228_220224-010259' # Sundmacher run

    group = True
    scatter = False 

    comp_list_path = '/home/bene/NNGamma/data/data_sund_sund200/0/comp_list.csv'
    path_temp = '/home/bene/NNGamma/out_fine_tune/'
    plot_path = '/home/bene/NNGamma/src/plot/'
    data_path = path_temp + name + '/'

    file_path = ["sund"]
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
        pr.make_scatter(val_predction_0, val_target_0, name = 'val_0', title = '$val_\mathrm{ext}$', path = plot_path, save=True)
        print('val_0_fine done')
        pr.make_scatter(val_predction_1[0:max], val_target_1[:max], name = 'val_1',  title = '$val_\mathrm{edge}$', path = plot_path, save=True)
        print('val_1_fine done')
        pr.make_scatter(val_predction_2[:max], val_target_2[:max], name = 'val_2', title = '$val_\mathrm{int}$', path = plot_path, save=True)
        print('val_2_fine done')
        if h2o_val_0_target != []:
            pr.make_scatter(h2o_val_0_pred, h2o_val_0_target, name = 'h2o_val_0', title = '$val_\mathrm{ext}$', path = plot_path, save=True)
        print('h2o_val_0_fine done')
        pr.make_scatter(h2o_val_1_pred[0:max], h2o_val_1_target[:max], name = 'h2o_val_1', title = '$val_\mathrm{edge}$', path = plot_path, save=True)
        print('h2o_val_1_fine done')
        pr.make_scatter(h2o_val_2_pred[:max], h2o_val_2_target[:max], name = 'h2o_val_2', title = '$val_\mathrm{int}$', path = plot_path, save=True)
        print('h2o_val_2_fine done')
        pr.make_scatter(no_h2o_val_0_pred, no_h2o_val_0_target, name = 'no_h2o_val_0', title = '$val_\mathrm{ext}$', path = plot_path, save=True)
        print('no_h2o_val_0_fine done')
        pr.make_scatter(no_h2o_val_1_pred[0:max], no_h2o_val_1_target[:max], name = 'no_h2o_val_1', title = '$val_\mathrm{edge}$', path = plot_path, save=True)
        print('no_h2o_val_1_fine done')
        pr.make_scatter(no_h2o_val_2_pred[:max], no_h2o_val_2_target[:max], name = 'no_h2o_val_2', title = '$val_\mathrm{int}$', path = plot_path, save=True)
        print('no_h2o_val_2_fine done')
        print('scatter plots made')
  
    
        # load sundmacher data
    n_e = 200

    for i in range(n_e):
        print(i)
        df_val0 = pd.read_csv('~/GNN_IAC/Ensemble_'+str(i)+'/Val0.csv')
        df_val1 = pd.read_csv('~/GNN_IAC/Ensemble_'+str(i)+'/Val1.csv')
        df_val2 = pd.read_csv('~/GNN_IAC/Ensemble_'+str(i)+'/Val2.csv')

        train = pd.read_csv('~/GNN_IAC/Ensemble_'+str(i)+'/Training.csv')
        best_epoch_V0 = np.argmin(np.array(train['Valid_0_loss']))
        best_epoch_V1 = np.argmin(np.array(train['Valid_1_loss']))
        best_epoch_V2 = np.argmin(np.array(train['Valid_2_loss']))

        best_epoch_V0 = 17
        best_epoch_V1 = 17
        best_epoch_V2 = 48

        if i == 0:
            df_val0_all = df_val0[['i','Literature',str(best_epoch_V0)]]
            df_val1_all = df_val1[['i','Literature',str(best_epoch_V1)]]
            df_val2_all = df_val2[['i','Literature',str(best_epoch_V2)]]
            # rename best epoch to prediction
            df_val0_all.rename(columns={str(best_epoch_V0):'Prediction'}, inplace=True)
            df_val1_all.rename(columns={str(best_epoch_V1):'Prediction'}, inplace=True)
            df_val2_all.rename(columns={str(best_epoch_V2):'Prediction'}, inplace=True)
        else:

            df_val0.rename(columns={str(best_epoch_V0):'Prediction'}, inplace=True)
            df_val1.rename(columns={str(best_epoch_V1):'Prediction'}, inplace=True)
            df_val2.rename(columns={str(best_epoch_V2):'Prediction'}, inplace=True)

            df_val0_all = pd.concat([df_val0_all, df_val0[['i','Literature','Prediction']]], axis=0, ignore_index=True)
            df_val1_all = pd.concat([df_val1_all, df_val1[['i','Literature','Prediction']]], axis=0, ignore_index=True)
            df_val2_all = pd.concat([df_val2_all, df_val2[['i','Literature','Prediction']]], axis=0, ignore_index=True)
            # rename best epoch to prediction

    df_val0_all = df_val0_all.groupby(['i','Literature']).mean().reset_index()
    df_val1_all = df_val1_all.groupby(['i','Literature']).mean().reset_index()
    df_val2_all = df_val2_all.groupby(['i','Literature']).mean().reset_index()

    IAC_data = pd.read_csv('~/GNN_IAC/Data/database_IAC_ln_clean.csv')

    cosmo_data = IAC_data[['Literature','COSMO_RS']]
    cosmo_data = cosmo_data.dropna()

    unifac_data = IAC_data[['Literature','mod_UNIFAC_Do']]
    unifac_data = unifac_data.dropna()

    cosmo_data_target = cosmo_data['Literature'].to_numpy(dtype=np.float64)
    cosmo_data_prediction = cosmo_data['COSMO_RS'].to_numpy(dtype=np.float64)

    unifac_data_target = unifac_data['Literature'].to_numpy(dtype=np.float64)
    unifac_data_prediction = unifac_data['mod_UNIFAC_Do'].to_numpy(dtype=np.float64)

    i_val_0 = set(val_0['input'].to_numpy())
    i_val_1 = set(val_1['input'].to_numpy())
    i_val_2 = set(val_2['input'].to_numpy())
    i_s_val_0 = set(df_val0_all['i'].to_numpy())
    i_s_val_1 = set(df_val1_all['i'].to_numpy())
    i_s_val_2 = set(df_val2_all['i'].to_numpy())
    i_cosmo = set(cosmo_data.index)
    i_UNIFAC = set(unifac_data.index)

    i_common = i_val_0.intersection(i_val_1).intersection(i_val_2).intersection(i_s_val_0).intersection(i_s_val_1).intersection(i_s_val_2).intersection(i_cosmo).intersection(i_UNIFAC)
    i_common = np.array(list(i_common), dtype=np.int64)

    val_predction_0 = val_0[val_0['input'].isin(i_common)]['prediction'].to_numpy(dtype=np.float64)
    val_predction_1 = val_1[val_1['input'].isin(i_common)]['prediction'].to_numpy(dtype=np.float64)
    val_predction_2 = val_2[val_2['input'].isin(i_common)]['prediction'].to_numpy(dtype=np.float64)

    val_target_0 = val_0[val_0['input'].isin(i_common)]['target'].to_numpy(dtype=np.float64)
    val_target_1 = val_1[val_1['input'].isin(i_common)]['target'].to_numpy(dtype=np.float64)
    val_target_2 = val_2[val_2['input'].isin(i_common)]['target'].to_numpy(dtype=np.float64)

    sund_pred_0 = df_val0_all[df_val0_all['i'].isin(i_common)]['Prediction'].to_numpy(dtype=np.float64)
    sund_pred_1 = df_val1_all[df_val1_all['i'].isin(i_common)]['Prediction'].to_numpy(dtype=np.float64)
    sund_pred_2 = df_val2_all[df_val2_all['i'].isin(i_common)]['Prediction'].to_numpy(dtype=np.float64)

    sund_target_0 = df_val0_all[df_val0_all['i'].isin(i_common)]['Literature'].to_numpy(dtype=np.float64)
    sund_target_1 = df_val1_all[df_val1_all['i'].isin(i_common)]['Literature'].to_numpy(dtype=np.float64)
    sund_target_2 = df_val2_all[df_val2_all['i'].isin(i_common)]['Literature'].to_numpy(dtype=np.float64)

    unifac_data_target = unifac_data.loc[i_common]['Literature'].to_numpy(dtype=np.float64)
    unifac_data_prediction = unifac_data.loc[i_common]['mod_UNIFAC_Do'].to_numpy(dtype=np.float64)

    cosmo_data_target = cosmo_data.loc[i_common]['Literature'].to_numpy(dtype=np.float64)
    cosmo_data_prediction = cosmo_data.loc[i_common]['COSMO_RS'].to_numpy(dtype=np.float64)

    # print MSE und MEA for sundmacher und val cosmo data
    print('MSE for sundmacher:')
    print('Val0:', np.mean((sund_target_0 - sund_pred_0)**2))
    print('Val1:', np.mean((sund_target_1 - sund_pred_1)**2))
    print('Val2:', np.mean((sund_target_2 - sund_pred_2)**2))
    print('MSE for Winter:')
    print('Val0:', np.mean((val_predction_0 - val_target_0)**2))
    print('Val1:', np.mean((val_predction_1 - val_target_1)**2))
    print('Val2:', np.mean((val_predction_2 - val_target_2)**2))
    print('MSE for cosmo data:')
    print('Val0:', np.mean((cosmo_data_target - cosmo_data_prediction)**2))
    print('MSE for unifac data:')
    print('Val0:', np.mean((unifac_data_target - unifac_data_prediction)**2))
    print('MEA for sundmacher:')
    print('Val0:', np.mean(np.abs(sund_target_0 - sund_pred_0)))
    print('Val1:', np.mean(np.abs(sund_target_1 - sund_pred_1)))
    print('Val2:', np.mean(np.abs(sund_target_2 - sund_pred_2)))
    print('MEA for Winter:')
    print('Val0:', np.mean(np.abs(val_target_0 - val_predction_0)))
    print('Val1:', np.mean(np.abs(val_target_1 - val_predction_1)))
    print('Val2:', np.mean(np.abs(val_target_2 - val_predction_2)))
    print('MEA for cosmo data:')
    print('Val0:', np.mean(np.abs(cosmo_data_target - cosmo_data_prediction)))
    print('MEA for unifac data:')
    print('Val0:', np.mean(np.abs(unifac_data_target - unifac_data_prediction)))

    # load cvs with data from COSMO
    prediction_list = [cosmo_data_target, unifac_data_target, sund_pred_0, sund_pred_1, sund_pred_2, val_predction_0, val_predction_1, val_predction_2]
    target_list = [cosmo_data_prediction, unifac_data_prediction, sund_target_0, sund_target_1, sund_target_2, val_target_0, val_target_1, val_target_2]


    color_list = ['grey', 'lightgrey','green', 'orange', 'blue','green', 'orange', 'blue']
    color_list = ['tab:purple', 'tab:purple','tab:green', 'tab:orange', 'tab:blue','tab:green', 'tab:orange', 'tab:blue']
    name_list = ['COSMO-RS', 'UNIFAC','Medina$_\mathrm{ext}$','Medina$_\mathrm{edge}$','Medina$_\mathrm{int}$','SPT$_\mathrm{ext}$', 'SPT$_\mathrm{edge}$', 'SPT$_\mathrm{int}$']
    line_style = ['dashdot','dashed',':', ':', ':', '-', '-', '-']

    pr.plot_err_curve_mult_sund(prediction_list, target_list, name_list, color_list, line_style, name = 'sund', path = plot_path, save=True)
