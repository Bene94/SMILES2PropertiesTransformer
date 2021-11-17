import numpy as np
import plot_results as pr

name = '211116-085945'
path_temp = '../out_fine_tuen/'
#path_temp = '../temp/'


#load data
val_predction_0 = np.load(path_temp + 'val_prediction_array_0_' + name + '.npy')
val_target_0 = np.load(path_temp + 'val_target_array_0_' + name + '.npy')

val_predction_1 = np.load(path_temp + 'val_prediction_array_1_' + name + '.npy')
val_target_1 = np.load(path_temp + 'val_target_array_1_' + name + '.npy')

val_predction_2 = np.load(path_temp + 'val_prediction_array_2_' + name + '.npy')
val_target_2 = np.load(path_temp + 'val_target_array_2_' + name + '.npy')


pr.make_scatter(val_predction_0, val_target_0, name = 'val_0', path = '', save=True)
pr.make_scatter(val_predction_1, val_target_1, name = 'val_1', path = '', save=True)
pr.make_scatter(val_predction_2, val_target_2, name = 'val_2', path = '', save=True)