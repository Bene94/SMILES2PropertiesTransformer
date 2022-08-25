
import matplotlib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tabulate import tabulate
from scipy.optimize import curve_fit
from torch import std 
import progressbar as pb

import plot_results as pr
import os
import pandas as pd


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




n_list = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 3000, 4000, 5000]

type_list = ['0', '1','2']

data_path = "/local/home/bewinter/Paper_SPT/SPT/out_fine_tune/n_f_aug_"
target_list_f_aug_0, prediction_list_f_aug_0, mse_list_f_aug_0, input_list_f_aug_0 = load_data_n(n_list, data_path, type_list[0])
target_list_f_aug_1, prediction_list_f_aug_1, mse_list_f_aug_1, input_list_f_aug_1 = load_data_n(n_list, data_path, type_list[1])
target_list_f_aug_2, prediction_list_f_aug_2, mse_list_f_aug_2, input_list_f_aug_2 = load_data_n(n_list, data_path, type_list[2])

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

# reduce the validation set to a consistent seti

cutoff_val_0 = 15
cutoff_val_1 = 21
cutoff_val_2 = 12

if False:
    input_set_aug_0 = np.array([item for sublist in input_list_f_aug_0[cutoff_val_0] for item in sublist])
    input_set_aug_1 = np.array([item for sublist in input_list_f_aug_1[cutoff_val_1] for item in sublist])
    bar = pb.ProgressBar(max_value=len(target_list_f_aug_0)*len(target_list_f_aug_0[0]), widgets=[pb.Timer(), pb.Bar(), pb.ETA()])
    bar.start()
    # check where input_set_aug_0[i] contains input_set_aug_0
    for i in range(0, len(input_list_f_aug_0)):
        for j in range(0, len(input_list_f_aug_0[i])):  
            bar.update(i * len(input_list_f_aug_0[i]) + j)
            temp_input = np.array(input_list_f_aug_0[i][j])
            # check where temp_input does not contain input_set_aug_0
            remove_index = [ k for k in range(0, len(temp_input)) if temp_input[k] not in input_set_aug_0]
            input_list_f_aug_0[i][j] = np.delete(input_list_f_aug_0[i][j], remove_index)
            target_list_f_aug_0[i][j] = np.delete(target_list_f_aug_0[i][j], remove_index)
            prediction_list_f_aug_0[i][j] = np.delete(prediction_list_f_aug_0[i][j], remove_index)
    bar.finish()
    bar = pb.ProgressBar(max_value=len(target_list_f_aug_1)*len(target_list_f_aug_1[0]), widgets=[pb.Timer(), pb.Bar(), pb.ETA()])
    bar.start()
    for i in range(0, len(input_list_f_aug_1)):
        for j in range(0, len(input_list_f_aug_1[i])):
            bar.update(i * len(input_list_f_aug_1[i]) + j)
            temp_input = np.array(input_list_f_aug_1[i][j])
            # check where temp_input does not contain input_set_aug_0
            remove_index = [ k for k in range(0, len(temp_input)) if temp_input[k] not in input_set_aug_1]
            input_list_f_aug_1[i][j] = np.delete(input_list_f_aug_1[i][j], remove_index)
            target_list_f_aug_1[i][j] = np.delete(target_list_f_aug_1[i][j], remove_index)
            prediction_list_f_aug_1[i][j] = np.delete(prediction_list_f_aug_1[i][j], remove_index)
    bar.finish()
else:
    prediction_list_f_aug_0 = prediction_list_f_aug_0[0:cutoff_val_0]
    target_list_f_aug_0 = target_list_f_aug_0[0:cutoff_val_0]
    input_list_f_aug_0 = input_list_f_aug_0[0:cutoff_val_0]
    mse_list_f_aug_0 = mse_list_f_aug_0[0:cutoff_val_0]
    prediction_list_f_aug_1 = prediction_list_f_aug_1[0:cutoff_val_1]
    target_list_f_aug_1 = target_list_f_aug_1[0:cutoff_val_1]
    input_list_f_aug_1 = input_list_f_aug_1[0:cutoff_val_1]
    mse_list_f_aug_1 = mse_list_f_aug_1[0:cutoff_val_1]
    prediction_list_f_aug_2 = prediction_list_f_aug_2[cutoff_val_2:]
    target_list_f_aug_2 = target_list_f_aug_2[cutoff_val_2:]
    input_list_f_aug_2 = input_list_f_aug_2[cutoff_val_2:]


# plot the mean mse for each n in a log log plot
fig, ax = plt.subplots(1, 1)
plt.rc('text', usetex=True)

#mean_mse = [np.median(mse) for mse in mse_list_0]
#mean_mse_ut = [np.median(mse) for mse in mse_list_ut_0
mse_ft_aug_0 = []
mse_ft_aug_1 = []
mse_ft_aug_2 = []

for i in range(len(target_list_f_aug_0)):
    temp_mse_0 = []
    for j in range(len(target_list_f_aug_0[i])):
        #temp_mse_0.append(np.nanmean(np.square(target_list_f_aug_0[i][j] - prediction_list_f_aug_0[i][j])))
        temp_mse_0.append(np.nanmean(np.abs(target_list_f_aug_0[i][j] - prediction_list_f_aug_0[i][j])))
    mse_ft_aug_0.append(temp_mse_0)

for i in range(len(target_list_f_aug_1)):
    temp_mse_1 = []
    for j in range(len(target_list_f_aug_1[i])):
        #temp_mse_1.append(np.nanmean(np.square(target_list_f_aug_1[i][j] - prediction_list_f_aug_1[i][j])))
        temp_mse_1.append(np.nanmean(np.abs(target_list_f_aug_1[i][j] - prediction_list_f_aug_1[i][j])))
    mse_ft_aug_1.append(temp_mse_1)

for i in range(len(target_list_f_aug_2)):
    temp_mse_2 = []
    for j in range(len(target_list_f_aug_2[i])):
        #temp_mse_2.append(np.nanmean(np.square(target_list_f_aug_2[i][j] - prediction_list_f_aug_2[i][j])))
        temp_mse_2.append(np.nanmean(np.abs(target_list_f_aug_2[i][j] - prediction_list_f_aug_2[i][j])))
    mse_ft_aug_2.append(temp_mse_2)

mean_mse_ft_aug_0 = [np.nanmean(mse) for mse in mse_ft_aug_0]
mean_mse_ft_aug_1 = [np.nanmean(mse) for mse in mse_ft_aug_1]
mean_mse_ft_aug_2 = [np.nanmean(mse) for mse in mse_ft_aug_2]

ax.plot(n_list[cutoff_val_2:] , mean_mse_ft_aug_2, label='Val$_\mathrm{int}$' , linestyle='', marker='v', color='#1f77b4')
ax.plot(n_list[0:cutoff_val_1], mean_mse_ft_aug_1, label='Val$_\mathrm{edge}$', linestyle='', marker='*', color='#ff7f0e')
ax.plot(n_list[0:cutoff_val_0], mean_mse_ft_aug_0, label='Val$_\mathrm{ext}$' , linestyle='', marker='o', color='#2ca02c')

# calculate the upper and lower 95% confident intervall of the mean mse
mse_ft_aug_0_ci = np.array([np.nanpercentile(mse, [15, 85]) for mse in mse_ft_aug_0])
mse_ft_aug_1_ci = np.array([np.nanpercentile(mse, [15, 85]) for mse in mse_ft_aug_1])
mse_ft_aug_2_ci = np.array([np.nanpercentile(mse, [15, 85]) for mse in mse_ft_aug_2])

#ax.fill_between(n_list, mse_ft_aug_2_ci[:,0], mse_ft_aug_2_ci[:,1], color='#1f77b4', alpha=0.2)
#ax.fill_between(n_list[0:cutoff_val_1], mse_ft_aug_1_ci[0:cutoff_val_1,0], mse_ft_aug_1_ci[0:cutoff_val_1,1], color='#ff7f0e', alpha=0.2)
#ax.fill_between(n_list[0:cutoff_val_0], mse_ft_aug_0_ci[0:cutoff_val_0,0], mse_ft_aug_0_ci[0:cutoff_val_0,1], color='#2ca02c', alpha=0.2)

# add horizontal line at 0.35
ax.axhline(y=0.39, color='k', linestyle='--', label='pre finetune')
#ax.axhline(y=0.11, color='#1f77b4', linestyle=':', label='limit $val_\mathrm{int}$', alpha=0.2)
#ax.axhline(y=0.13, color='#ff7f0e', linestyle=':', label='limit $val_\mathrm{edge}$', alpha=0.2)
#ax.axhline(y=0.17, color='#2ca02c', linestyle=':', label='limit $val_\mathrm{ext}$',  alpha=0.2)

lb = 2
ub = 5000

# fit a exponential regression into the data and plot it
exp_fuc = lambda x, a, b: a * x**b
x = np.array(n_list[0:cutoff_val_0])
y = np.array(mean_mse_ft_aug_0[0:cutoff_val_0])
p0 = [1, -0.0001]
popt, pcov = curve_fit(exp_fuc, x, y, p0)
x_fit = np.linspace(lb, ub, 200)
ax.plot(x_fit, exp_fuc(x_fit, *popt), linestyle='--',  color='#2ca02c', alpha=0.4)
# write the fit parameters into the plot next to the line
#ax.text(0.75, 0.85, 'Val$_\mathrm{ext}$\na = %.2f b = %.2f' % tuple(popt), transform=ax.transAxes)
x = np.array(n_list[0:cutoff_val_1])
y = np.array(mean_mse_ft_aug_1[0:cutoff_val_1])
p0 = [1, -0.0001]
popt, pcov = curve_fit(exp_fuc, x, y, p0)
x_fit = np.linspace(lb, ub, 200)
ax.plot(x_fit, exp_fuc(x_fit, *popt), linestyle='--',  color='#ff7f0e', alpha=0.4)
# write the fit parameters into the plot
#ax.text(0.75, 0.75, 'Val$_\mathrm{edge}$\na = %.2f b = %.2f' % tuple(popt), transform=ax.transAxes)
x = np.array(n_list[cutoff_val_2:])
y = np.array(mean_mse_ft_aug_2)
p0 = [1, -0.0001] 
popt, pcov = curve_fit(exp_fuc, x, y, p0)
x_fit = np.linspace(lb, ub, 200)
ax.plot(x_fit, exp_fuc(x_fit, *popt), linestyle='--',  color='#1f77b4', alpha=0.4)
# write the fit parameters into the plot
plt.rcParams.update({'font.size': 16})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel('# of training mixtures', fontsize=16)
ax.set_ylabel('average MAE', fontsize=16)
ax.set_ylim(0.1, 0.4)
ax.set_yticks([0.1, 0.2, 0.3, 0.4])
ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4'])
ax.set_xlim(lb, ub)
ax.set_xticks([2, 10, 100, 1000, 5000])
ax.set_xticklabels(['2','10', '100', '1000', '5000'])
ax.legend(loc='lower left')
plt.tight_layout()
plt.show
plt.savefig('plot/boxplot/mean_mse_val_0.png', dpi=900)
plt.savefig('plot/boxplot/mean_mse_val_0.pdf')