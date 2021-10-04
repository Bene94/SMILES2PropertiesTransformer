
import numpy as np

from plot_results import *

# load data

name = '211003-111953'

val_loss_array = np.load('../temp/'+ 'val_loss_array_' + name + '.npy')
val_prediction_array = np.load('../temp/'+ 'val_prediction_array_' + name + '.npy')
val_target_array = np.load('../temp/'+ 'val_target_array_' + name + '.npy')

val_prediction_array = val_prediction_array.squeeze()
val_target_array = val_target_array.squeeze()

make_scatter(val_prediction_array, val_target_array, '', save=True)