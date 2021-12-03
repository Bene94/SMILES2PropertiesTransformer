import numpy as np
import plot_results as pr
import pandas as pd

name = '211201-125751'
path_temp = '../out_fine_tuen/'
#path_temp = '../temp/'


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

pr.make_heatmap(val_predction_0, val_target_0, 'val_0_fine' , path = '', save=True)
pr.make_heatmap(val_predction_1, val_target_1, 'val_1_fine' , path = '', save=True)
pr.make_heatmap(val_predction_2, val_target_2, 'val_2_fine' , path = '', save=True)

print('heatmaps made')

max = int(2e5)

pr.make_scatter(val_predction_0, val_target_0, name = 'val_0', path = '', save=True)
print('val_0_fine done')
pr.make_scatter(val_predction_1[0:max], val_target_1[:max], name = 'val_1', path = '', save=True)
print('val_1_fine done')
pr.make_scatter(val_predction_2[:max], val_target_2[:max], name = 'val_2', path = '', save=True)
print('val_2_fine done')
print('scatter plots made')