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



data_path = "/home/bene/NNGamma/out_fine_tuen/n_"
n_list = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000] # , 100, 500, 1000]

type_list = ['0', '1','2']

target_list_0, prediction_list_0, mse_list_0 = load_data(n_list, data_path, type_list[0])
target_list_1, prediction_list_1, mse_list_1 = load_data(n_list, data_path, type_list[1])
target_list_2, prediction_list_2, mse_list_2 = load_data(n_list, data_path, type_list[2])

# make a figure with three subplots sharing both x axes with the mse list as boxplot

fig, ax = plt.subplots(3, 1)
#  y axis limit between 0 and 1
ax[0].set_ylim(0, 1)
ax[1].set_ylim(0, 1)
ax[2].set_ylim(0, 1)

ax[0].boxplot(mse_list_0)
ax[1].boxplot(mse_list_1)
ax[2].boxplot(mse_list_2)

ax[0].set_title('val 0')
ax[1].set_title('val 1')
ax[2].set_title('val 2')

ax[0].set_xlabel('n')
ax[1].set_xlabel('n')
ax[2].set_xlabel('n')

ax[0].set_ylabel('mse')
ax[1].set_ylabel('mse')
ax[2].set_ylabel('mse')
# set the x ticks to n_list
ax[0].set_xticklabels(n_list)



plt.show()
# save the figure
plt.savefig('plot/boxplot_mse.png')

# make one figure with just val 0
fig, ax = plt.subplots(1, 1)
ax.boxplot(mse_list_0)
ax.plot(range(1,len(n_list)+1), [np.median(mse) for mse in mse_list_0], label='val 0', linestyle='--', marker='o')
ax.set_title('val 0')
ax.set_xlabel('n')
ax.set_ylabel('mse')
ax.set_xticklabels(n_list)
ax.set_ylim(0, 1)
# draw a line at 0.35 and 0.22
ax.axhline(y=0.35, color='r', linestyle='--')
ax.axhline(y=0.22, color='r', linestyle='--')
# lable the line as "before fine tune" and "after fine tune" on the right side of the plot
ax.text(len(n_list) + 1, 0.35, 'before fine tune', rotation=90, verticalalignment='center')
ax.text(len(n_list) + 0.7, 0.22, 'after fine tune', rotation=90, verticalalignment='center')

plt.show()
plt.savefig('plot/boxplot_mse_val0.png')

# make one figure with just val 1
fig, ax = plt.subplots(1, 1)
ax.boxplot(mse_list_1)
ax.set_title('val 1')
ax.set_xlabel('n')
ax.set_ylabel('mse')
ax.set_xticklabels(n_list)
ax.set_ylim(0, 1)
# add line with mean mse
# draw a line at 0.35 and 0.22

ax.axhline(y=0.22, color='r', linestyle='--')
# lable the line as "before fine tune" and "after fine tune" on the right side of the plot
ax.text(len(n_list) + 1, 0.35, 'before fine tune', rotation=90, verticalalignment='center')
ax.text(len(n_list) + 0.7, 0.22, 'after fine tune', rotation=90, verticalalignment='center')

plt.show()
plt.savefig('plot/boxplot_mse_val1.png')


# make one figure with just val 2
fig, ax = plt.subplots(1, 1)
ax.boxplot(mse_list_2)
ax.set_title('val 2')
ax.set_xlabel('n')
ax.set_ylabel('mse')
ax.set_xticklabels(n_list)
ax.set_ylim(0, 1)
# draw a line at 0.35 and 0.22
ax.axhline(y=0.35, color='r', linestyle='--')
ax.axhline(y=0.22, color='r', linestyle='--')
# lable the line as "before fine tune" and "after fine tune" on the right side of the plot
ax.text(len(n_list) + 1, 0.35, 'before fine tune', rotation=90, verticalalignment='center')
ax.text(len(n_list) + 0.7, 0.22, 'after fine tune', rotation=90, verticalalignment='center')

plt.show()
plt.savefig('plot/boxplot_mse_val2.png')

index = 3

# plot the mean mse for each n in a log log plot
fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('n')
ax.set_ylabel('loss')
ax.set_title('Median loss as funciton of training data')
ax.plot(n_list, [np.median(mse) for mse in mse_list_0], label='val 0', linestyle='--', marker='o')
# make y axis limit between 0.1 and 1 and make labeling not scientific
ax.set_ylim(0.1, 1)
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])

plt.show
plt.savefig('plot/mean_mse_val0.png')


