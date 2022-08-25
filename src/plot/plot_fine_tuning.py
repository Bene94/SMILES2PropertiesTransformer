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

    name = 'f_t_512_T_base_220412-152241'

    group = True 
    scatter = True 

    path_temp = '../out_fine_tune/'
    plot_path = '../src/plot/'
    data_path = path_temp + name + '/'

    type_list = ['0', '1', '2']

    val_target_0, val_predction_0, mse_list_0, val_input_0 = load_data(data_path, type_list[0])
    val_target_1, val_predction_1, mse_list_1, val_input_1 = load_data(data_path, type_list[1])
    val_target_2, val_predction_2, mse_list_2, val_input_2 = load_data(data_path, type_list[2])

    print('data loaded')
    print('length val0: ' + str(len(val_predction_0)))
    print('length val1: ' + str(len(val_predction_1)))
    print('length val2: ' + str(len(val_predction_2)))

    if len(val_target_0) > 0:
        val_target_0 = np.concatenate(val_target_0)
        val_input_0 = np.concatenate(val_input_0)
        val_predction_0 = np.concatenate(val_predction_0)
        mse_0 = np.mean(mse_list_0)
        mea_0 = np.mean(np.abs(val_target_0-val_predction_0))
        print('MSE_0:', '{:.2f}'.format(np.mean(mse_0)))
        print('MAE_0:', '{:.2f}'.format(np.mean(mea_0)))
   
    if len(val_target_1) > 0:
        val_target_1 = np.concatenate(val_target_1)
        val_input_1 = np.concatenate(val_input_1)
        val_predction_1 = np.concatenate(val_predction_1)
        mse_1 = np.mean(mse_list_1)
        mea_1 = np.mean(np.abs(val_target_1-val_predction_1))
        print('MSE_1:', '{:.2f}'.format(np.mean(mse_1)))
        print('MAE_1:', '{:.2f}'.format(np.mean(mea_1)))
    
    if len(val_target_2) > 0:
        val_target_2 = np.concatenate(val_target_2)
        val_input_2 = np.concatenate(val_input_2)
        val_predction_2 = np.concatenate(val_predction_2)
        mse_2 = np.mean(mse_list_2)
        mea_2 = np.mean(np.abs(val_target_2-val_predction_2))
        print('MSE_2:', '{:.2f}'.format(np.mean(mse_2)))
        print('MAE_2:', '{:.2f}'.format(np.mean(mea_2)))
        
    if group:

        if len(val_input_0) > 0:        
            val_0 = pd.DataFrame({'input':val_input_0.squeeze(),'prediction': val_predction_0.squeeze(),'target': val_target_0.squeeze()})
            val_0 = val_0.groupby(['input']).mean()
            val_0 = val_0.reset_index()
            val_predction_0 = val_0['prediction'].to_numpy()
            val_target_0 = val_0['target'].to_numpy()
            val_input_0 = val_0['input'].to_numpy()

        if len(val_input_1) > 0:
            val_1 = pd.DataFrame({'input':val_input_1.squeeze(),'prediction': val_predction_1.squeeze(),'target': val_target_1.squeeze()})
            val_1 = val_1.groupby(['input']).mean()
            val_1 = val_1.reset_index()
            val_predction_1 = val_1['prediction'].to_numpy()
            val_target_1 = val_1['target'].to_numpy()
            val_input_1 = val_1['input'].to_numpy()
        
        if len(val_input_2) > 0:
            val_2 = pd.DataFrame({'input':val_input_2.squeeze(),'prediction': val_predction_2.squeeze(),'target': val_target_2.squeeze()})
            val_2 = val_2.groupby(['input']).mean()
            val_2 = val_2.reset_index()
            val_predction_2 = val_2['prediction'].to_numpy()
            val_target_2 = val_2['target'].to_numpy()
            val_input_2 = val_2['input'].to_numpy()

    if len(val_predction_0) > 0:
        pr.make_heatmap(val_predction_0, val_target_0, r'val0' , path = plot_path, save=True)
        pr.make_historgam_delta(val_predction_0, val_target_0, 'val_0_fine' , path = plot_path, save=True)
    if len(val_predction_1) > 0:
        pr.make_heatmap(val_predction_1, val_target_1, r'val1' , path = plot_path, save=True)
        pr.make_historgam_delta(val_predction_1, val_target_1, 'val_1_fine' , path = plot_path, save=True)
    if len(val_predction_2) > 0:
        pr.make_heatmap(val_predction_2, val_target_2, r'val2' , path = plot_path, save=True)
        pr.make_historgam_delta(val_predction_2, val_target_2, 'val_2_fine' , path = plot_path, save=True)

    print('heatmaps plotted')

    max = int(2e5)
    if scatter:
        if len(val_predction_0) > 0:
            pr.make_scatter(val_predction_0[0:max], val_target_0[0:max], name = 'val_0', title = '$val_\mathrm{ext}$', path = plot_path, save=True)
            print('val_0_fine done')
        if len(val_predction_1) > 0:
            pr.make_scatter(val_predction_1[0:max], val_target_1[0:max], name = 'val_1', title = '$val_\mathrm{int}$', path = plot_path, save=True)
            print('val_1_fine done')
        if len(val_predction_2) > 0:
            pr.make_scatter(val_predction_2[0:max], val_target_2[0:max], name = 'val_2', title = '$val_\mathrm{int}$', path = plot_path, save=True)
            print('val_2_fine done')
        print('scatter plots plotted')
        