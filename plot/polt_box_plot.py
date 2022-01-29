
import matplotlib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tabulate import tabulate


import plot_results as pr
import os


def load_data_n(n_list, file_path, val_type):

    target_list_n = []
    prediction_list_n = []
    mse_list_n = []
    input_list_n = []

    for n in n_list:
        path = file_path + str(n) + '/'

        target_list, prediction_list, mse_list, input_list = load_data(path, val_type)

        target_list_n.append(target_list)
        prediction_list_n.append(prediction_list)
        mse_list_n.append(mse_list)
        input_list_n.append(input_list)
    
    return target_list_n, prediction_list_n, mse_list_n, input_list_n

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




n_list = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000] #, 2000] # , 100, 500, 1000]

n_list = [10, 50, 100, 500, 999, 1000, 2000, 3000, 4000, 5000]
type_list = ['0', '1','2']

if False:
    data_path = "/home/bene/NNGamma/out_fine_tune/n_"
    target_list_0, prediction_list_0, mse_list_0 = load_data(n_list, data_path, type_list[0])
    target_list_1, prediction_list_1, mse_list_1 = load_data(n_list, data_path, type_list[1])
    target_list_2, prediction_list_2, mse_list_2 = load_data(n_list, data_path, type_list[2])

if False:
    data_path = "/home/bene/NNGamma/out_fine_tune/n_ut_"
    target_list_ut_0, prediction_list_ut_0, mse_list_ut_0 = load_data(n_list, data_path, type_list[0])
    target_list_ut_1, prediction_list_ut_1, mse_list_ut_1 = load_data(n_list, data_path, type_list[1])
    target_list_ut_2, prediction_list_ut_2, mse_list_ut_2 = load_data(n_list, data_path, type_list[2])
if True:
    data_path = "/home/bene/NNGamma/out_fine_tune/n_f_aug_"
    target_list_f_aug_0, prediction_list_f_aug_0, mse_list_f_aug_0, input_list_f_aug_0 = load_data_n(n_list, data_path, type_list[0])
    target_list_f_aug_1, prediction_list_f_aug_1, mse_list_f_aug_1, input_list_f_aug_1 = load_data_n(n_list, data_path, type_list[1])
    target_list_f_aug_2, prediction_list_f_aug_2, mse_list_f_aug_2, input_list_f_aug_2 = load_data_n(n_list, data_path, type_list[2])
if False:
    data_path = "/home/bene/NNGamma/out_fine_tune/n_ut_aug_"
    target_list_ut_aug_0, prediction_list_ut_aug_0, mse_list_ut_aug_0 = load_data(n_list, data_path, type_list[0])
    target_list_ut_aug_1, prediction_list_ut_aug_1, mse_list_ut_aug_1 = load_data(n_list, data_path, type_list[1])
    target_list_ut_aug_2, prediction_list_ut_aug_2, mse_list_ut_aug_2 = load_data(n_list, data_path, type_list[2])

# get input_list_f_aug_0[:][3][:] into single array
input_list_f_aug_0_number = []
for i in range(0, len(input_list_f_aug_0)):
    temp_i_flat = [item for sublist in input_list_f_aug_0[i] for item in sublist]
    input_list_f_aug_0_number.append(len(np.unique(temp_i_flat)))
input_list_f_aug_1_number = []
for i in range(0, len(input_list_f_aug_1)):
    temp_i_flat = [item for sublist in input_list_f_aug_1[i] for item in sublist]
    input_list_f_aug_1_number.append(len(np.unique(temp_i_flat)))
input_list_f_aug_2_number = []
for i in range(0, len(input_list_f_aug_2)):
    temp_i_flat = [item for sublist in input_list_f_aug_2[i] for item in sublist]
    input_list_f_aug_2_number.append(len(np.unique(temp_i_flat)))

# print input_list_f_aug_x_number as a table with n_list as header using tabulate
print(tabulate([input_list_f_aug_0_number, input_list_f_aug_1_number, input_list_f_aug_2_number], headers=n_list, tablefmt="fancy_grid", showindex=True))


# plot the mean mse for each n in a log log plot
fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('n')
ax.set_ylabel('loss')
ax.set_title('Mean loss as funciton of training data')
#mean_mse = [np.median(mse) for mse in mse_list_0]
#mean_mse_ut = [np.median(mse) for mse in mse_list_ut_0
mse_ft_aug_0 = []
mse_ft_aug_1 = []
mse_ft_aug_2 = []

for i in range(len(target_list_f_aug_0)):
    temp_mse_0 = []
    temp_mse_1 = []
    temp_mse_2 = []
    for j in range(len(target_list_f_aug_0[i])):
        temp_mse_0.append(np.nanmean(np.square(target_list_f_aug_0[i][j] - prediction_list_f_aug_0[i][j])))
        temp_mse_1.append(np.nanmean(np.square(target_list_f_aug_1[i][j] - prediction_list_f_aug_1[i][j])))
        temp_mse_2.append(np.nanmean(np.square(target_list_f_aug_2[i][j] - prediction_list_f_aug_2[i][j])))
    mse_ft_aug_0.append(temp_mse_0)
    mse_ft_aug_1.append(temp_mse_1)
    mse_ft_aug_2.append(temp_mse_2)

mean_mse_ft_aug_0 = [np.nanmean(mse) for mse in mse_ft_aug_0]
mean_mse_ft_aug_1 = [np.nanmean(mse) for mse in mse_ft_aug_1]
mean_mse_ft_aug_2 = [np.nanmean(mse) for mse in mse_ft_aug_2]

ax.plot(n_list, mean_mse_ft_aug_0, label='ft aug 0', linestyle='--', marker='o')
ax.plot(n_list, mean_mse_ft_aug_1, label='ft aug 1', linestyle='--', marker='*')
ax.plot(n_list, mean_mse_ft_aug_2, label='ft aug 2', linestyle='--', marker='v')

# add horizontal line at 0.35
ax.axhline(y=0.35, color='k', linestyle='--', label='before fine tune')
ax.axhline(y=0.13, color='k', linestyle='dashdot', label='after fine tune Val 0', c='0.85')
ax.axhline(y=0.08, color='k', linestyle='dashdot', label='after fine tune Val 1', c='0.85')
ax.axhline(y=0.04, color='k', linestyle='dashdot', label='after fine tune Val 2', c='0.85')

# make y axis limit between 0.05 and 0.4 and make labeling not scientific
ax.set_ylim(0.03, 0.4)
ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
ax.set_yticklabels(['0.05', '0.1', '0.2', '0.3', '0.4'])
# add a legend
ax.legend(loc='upper right')
# decrease ledgend size
ax.legend(loc='upper right', prop={'size': 6})
plt.show
plt.savefig('plot/boxplot/mean_mse_val_0.png')


