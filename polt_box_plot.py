import numpy as np
from matplotlib import pyplot as plt

data_path = "../temp/val_loss_array_step_211003-111953_20_50.npy"

# load data
data = np.load(data_path, allow_pickle=True)

sample_list = []
plot_list = []

for sample in data:
    sample_list.append(np.zeros((len(sample),len(sample[0]))))
    for i in range(len(sample)):
        sample_list[-1][i] = sample[i]
    mean = np.mean(sample_list[-1], axis=0)
    min_loss_index = np.argmin(mean)
    print(min_loss_index)
    plot_list.append(sample_list[-1][:,min_loss_index])

# take mean over ax 2
temp = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000]
labels = [str(i) for i in temp]
# make box plot of the data
plt.boxplot(plot_list, labels=labels)
plt.title("Box Plot of Validation Loss")
plt.ylabel("Validation Loss")
plt.xlabel("# Experiment Data")
plt.show()
# connect the mean of the losses with a line
#plt.plot(np.mean(n_10_loss), np.mean(n_100_loss), np.mean(n_500_loss), np.mean(n_1000_loss), np.mean(n_one_out_loss))

# save to polt/box_plot.png
plt.savefig("plot/box_plot.png")