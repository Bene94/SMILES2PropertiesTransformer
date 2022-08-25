
from dataclasses import dataclass
import matplotlib.pyplot as plt
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from transprop.nn_dataloader import *

@dataclass()
class Config:
    data_path: str
    data_path: str
    shift: int
    mode: str = 'reg'
    batch_size: int = 1024
    test = False



## function to compare the data from experimental and simulation data

path_data = os.path.join('../inf_cosmo/')

config = Config(data_path=path_data, shift=0)

data = gamma_dataset(path_data, '', config)

print("Length of data: " +  str(len(data)))

target = data.train_target
xT = data.xT
xT = xT.numpy()
x = xT[:,0]
T = xT[:,1]
target = torch.squeeze(target)
target = target.numpy()
# plot target over x, x is in range 0,1
#plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots()
ax.plot(x, target[:,0], '*b', label='target')
ax.plot(1-x, target[:,1], '*r', label='target')
ax.set_xlabel('x')
ax.set_ylabel('ln $\gamma_\infty$')
ax.set_title('Target over x')
ax.legend()
plt.show()


#save in the plot/ folder
plt.savefig('plot/analysis/target_over_x.png')

print('Plotted target over x')
# plot target over T


fig, ax = plt.subplots()
ax.plot(T, target, '*b', label='target')
ax.set_xlabel('T')
ax.set_ylabel('ln $\gamma_\infty$')
ax.set_title('Target over T')
ax.legend()
plt.show()

#save in the plot/ folder
plt.savefig('plot/analysis/target_over_T.png')

# plot the distribution of the target

plt.clf()
fig, ax = plt.subplots()
ax.hist(target, bins=201)
ax.set_xlabel('ln $\gamma_\infty$')
ax.set_ylabel('counts')
ax.set_title('Target distribution')
plt.show()

plt.savefig('plot/analysis/target_distribution.png')

# plot the distribution over T


plt.clf()
fig, ax = plt.subplots()
ax.hist(T, bins=51)
ax.set_xlabel('T')
ax.set_ylabel('counts')
ax.set_title('T distribution')
plt.show()

plt.savefig('plot/analysis/T_distribution.png')

plt.clf()
fig, ax = plt.subplots()
ax.hist(x, bins=51)
ax.set_xlabel('x')
ax.set_ylabel('counts')
ax.set_title('x distribution')
plt.show()

plt.savefig('plot/analysis/x_distribution.png')