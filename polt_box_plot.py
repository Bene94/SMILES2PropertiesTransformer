import numpy as np
from matplotlib import pyplot as plt

data_path = "../temp/val_loss_array_211003-111953.npy"

# load data
data = np.load(data_path, allow_pickle=True)
n_10 = data[0]
n_100 = data[1]
n_500 = data[2]
n_1000 = data[3]

n_0_loss = 0.12
n_10_loss = np.array([])
n_100_loss = np.array([])
n_500_loss = np.array([])
n_1000_loss = np.array([])
n_one_out_loss = 0.04
for run in n_10:
    n_10_loss = np.append(n_10_loss, run[-1])

for run in n_100:
    n_100_loss = np.append(n_100_loss, run[-1])

for run in n_500:
    n_500_loss = np.append(n_500_loss, run[-1])  

for run in n_1000:
    n_1000_loss = np.append(n_1000_loss, run[-1])



# take mean over ax 2

# make box plot of the data
plt.boxplot([n_0_loss,n_10_loss, n_100_loss, n_500_loss, n_1000_loss,n_one_out_loss], labels=["0", "10", "100", "500", "1000", "one out"])
plt.title("Box Plot of Validation Loss")
plt.ylabel("Validation Loss")
plt.xlabel("# Experiment Data")
plt.show()
# connect the mean of the losses with a line
plt.plot(np.mean(n_10_loss), np.mean(n_100_loss), np.mean(n_500_loss), np.mean(n_1000_loss), np.mean(n_one_out_loss))

# save to polt/box_plot.png
plt.savefig("plot/box_plot.png")