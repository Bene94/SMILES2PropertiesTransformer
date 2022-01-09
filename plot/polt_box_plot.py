
import numpy as np
from matplotlib import pyplot as plt

import plot_results as pr


def load_data(n_list, file_path, type):

    target_list = []
    prediction_list = []
    mse_list = []

    for n in n_list:
        path = file_path + str(n) + '/'
        temp_target_list = []
        temp_prediction_list = []
        temp_mse_list = []
        for i in range(0, 200):
            temp_target_list.append(np.load(path + 'val_target_'+ type + '_' + str(i) + '.npy'))
            temp_prediction_list.append(np.load(path + 'val_predction_'+ type + '_' + str(i) + '.npy'))
            mse = np.mean(np.square(temp_target_list[i] - temp_prediction_list[i]))
            temp_mse_list.append(mse)

        target_list.append(temp_target_list)
        prediction_list.append(temp_prediction_list)
        mse_list.append(temp_mse_list)

    
    return target_list, prediction_list, mse_list



n_list = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000] #, 2000] # , 100, 500, 1000]

type_list = ['0', '1','2']

if False:
    data_path = "/home/bene/NNGamma/out_fine_tuen/n_"
    target_list_0, prediction_list_0, mse_list_0 = load_data(n_list, data_path, type_list[0])
    target_list_1, prediction_list_1, mse_list_1 = load_data(n_list, data_path, type_list[1])
    target_list_2, prediction_list_2, mse_list_2 = load_data(n_list, data_path, type_list[2])

if False:
    data_path = "/home/bene/NNGamma/out_fine_tuen/n_ut_"
    target_list_ut_0, prediction_list_ut_0, mse_list_ut_0 = load_data(n_list, data_path, type_list[0])
    target_list_ut_1, prediction_list_ut_1, mse_list_ut_1 = load_data(n_list, data_path, type_list[1])
    target_list_ut_2, prediction_list_ut_2, mse_list_ut_2 = load_data(n_list, data_path, type_list[2])
if True:
    data_path = "/home/bene/NNGamma/out_fine_tuen/n_f_aug_"
    target_list_f_aug_0, prediction_list_f_aug_0, mse_list_f_aug_0 = load_data(n_list, data_path, type_list[0])
    target_list_f_aug_1, prediction_list_f_aug_1, mse_list_f_aug_1 = load_data(n_list, data_path, type_list[1])
    target_list_f_aug_2, prediction_list_f_aug_2, mse_list_f_aug_2 = load_data(n_list, data_path, type_list[2])
if False:
    data_path = "/home/bene/NNGamma/out_fine_tuen/n_ut_aug_"
    target_list_ut_aug_0, prediction_list_ut_aug_0, mse_list_ut_aug_0 = load_data(n_list, data_path, type_list[0])
    target_list_ut_aug_1, prediction_list_ut_aug_1, mse_list_ut_aug_1 = load_data(n_list, data_path, type_list[1])
    target_list_ut_aug_2, prediction_list_ut_aug_2, mse_list_ut_aug_2 = load_data(n_list, data_path, type_list[2])

# plot the mean mse for each n in a log log plot
fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('n')
ax.set_ylabel('loss')
ax.set_title('Median loss as funciton of training data')
#mean_mse = [np.median(mse) for mse in mse_list_0]
#mean_mse_ut = [np.median(mse) for mse in mse_list_ut_0]
mean_mse_ft_aug = [np.median(mse) for mse in mse_list_f_aug_0]
#mean_mse_ut_aug = [np.median(mse) for mse in mse_list_ut_aug_0]


#ax.plot(n_list, mean_mse, label='ft', linestyle='--', marker='o')
#ax.plot(n_list, mean_mse_ut, label='ut', linestyle='--', marker='o')
#ax.plot(n_list, mean_mse_ft_aug, label='ft aug', linestyle='--', marker='o')
#ax.plot(n_list, mean_mse_ut_aug, label='ut aug', linestyle='--', marker='o')

# add horizontal line at 0.35
ax.axhline(y=0.35, color='k', linestyle='--', label='before fine tune')
ax.axhline(y=0.16, color='k', linestyle='dashdot', label='after fine tune')



# make y axis limit between 0.1 and 1 and make labeling not scientific
ax.set_ylim(0.1, 5)
# add a legend
ax.legend(loc='upper right')
# decrease ledgend size
ax.legend(loc='upper right', prop={'size': 6})
plt.show
plt.savefig('plot/boxplot/mean_mse_val_0.png')


