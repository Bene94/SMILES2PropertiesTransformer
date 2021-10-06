import numpy as np
from matplotlib import pyplot as plt

data_path = "../temp/val_loss_array_211003-111953.npy"
data_path_20_50 = "../temp/val_loss_array_step_211003-111953_20_50.npy"

# load data
data = np.load(data_path, allow_pickle=True)
n_10 = data[0]
n_100 = data[1]
n_500 = data[2]
n_1000 = data[3]
data = np.load(data_path_20_50, allow_pickle=True)
n__20 = data[0]
n__50 = data[1]

n_0_loss = 0.12
n_10_loss = np.zeros([len(n_10),len(n_10[0])])
n_20_loss = np.zeros([len(n__20),len(n__20[0])])
n_50_loss = np.zeros([len(n__50),len(n__50[0])])
n_100_loss = np.zeros([len(n_100),len(n_100[0])])
n_500_loss = np.zeros([len(n_500),len(n_500[0])])
n_1000_loss = np.zeros([len(n_1000),len(n_1000[0])])
n_one_out_loss = 0.04

for j in range(len(n_10)):
    for i in range(len(n_10[j])):
        n_10_loss[j,i] = n_10[j][i]

for j in range(len(n__20)):
    for i in range(len(n__20[j])):
        n_20_loss[j,i] = n__20[j][i]

for j in range(len(n__50)):
    for i in range(len(n__50[j])):
        n_50_loss[j,i] = n__50[j][i]

for j in range(len(n_100)):
    for i in range(len(n_100[j])):
        n_100_loss[j,i] = n_100[j][i]

for j in range(len(n_500)):
    for i in range(len(n_500[j])):
        n_500_loss[j,i] = n_500[j][i]

for j in range(len(n_1000)):
    for i in range(len(n_1000[j])):
        n_1000_loss[j,i] = n_1000[j][i]


mean_n_10 = np.mean(n_10_loss, axis=0)
mean_n_20 = np.mean(n_20_loss, axis=0)
mean_n_50 = np.mean(n_50_loss, axis=0)
mean_n_100 = np.mean(n_100_loss, axis=0)
mean_n_500 = np.mean(n_500_loss, axis=0)
mean_n_1000 = np.mean(n_1000_loss, axis=0)

# find index of the minimum loss
min_loss_index = np.argmin(mean_n_10)
n_10_loss = n_10_loss[:,min_loss_index]
min_loss_index  = np.argmin(mean_n_20)
n_20_loss = n_20_loss[:,min_loss_index]
min_loss_index  = np.argmin(mean_n_50)
n_50_loss = n_50_loss[:,min_loss_index]
min_loss_index = np.argmin(mean_n_100)
n_100_loss = n_100_loss[:,min_loss_index]
min_loss_index = np.argmin(mean_n_500)
n_500_loss = n_500_loss[:,min_loss_index]
min_loss_index = np.argmin(mean_n_1000)
n_1000_loss = n_1000_loss[:,min_loss_index]

# take mean over ax 2

# make box plot of the data
plt.boxplot([n_0_loss,n_10_loss, n_20_loss, n_50_loss, n_100_loss, n_500_loss, n_1000_loss, n_one_out_loss], labels=['0','10', '20', '50', '100', '500', '1000','2000'])
plt.title("Box Plot of Validation Loss")
plt.ylabel("Validation Loss")
plt.xlabel("# Experiment Data")
plt.show()
# connect the mean of the losses with a line
#plt.plot(np.mean(n_10_loss), np.mean(n_100_loss), np.mean(n_500_loss), np.mean(n_1000_loss), np.mean(n_one_out_loss))

# save to polt/box_plot.png
plt.savefig("plot/box_plot.png")