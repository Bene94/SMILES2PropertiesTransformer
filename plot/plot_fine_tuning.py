
import numpy as np

from plot_results import *

# load data

name = '211116-164532'

val_loss_array = np.load('../out_fine_tuen/'+ 'val_loss_array_' + name + '.npy', allow_pickle=True)
val_prediction_array = np.load('../out_fine_tuen/'+ 'val_prediction_array_' + name + '.npy', allow_pickle=True)
val_target_array = np.load('../out_fine_tuen/'+ 'val_target_array_' + name + '.npy', allow_pickle=True)

mean_loss = np.mean(val_loss_array, axis=0)

print(mean_loss)

val_prediction_array = val_prediction_array[:,-1]
val_target_array = val_target_array[:,-1]

make_scatter(val_prediction_array, val_target_array, 'EXP', save=True)
make_historgam_delta(val_prediction_array, val_target_array, 'EXP', save=True)