# %% 
# This is a simple script to evaluate the modle for custom data pairs
import os
import sys

# add current path to sys.path
sys.path.append(os.getcwd())

# %% Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from transprop.trainer import *
from transprop.nn_dataloader import *
from transprop.load_model import *
from simple_evaluation_utils import *

# %% Parameters
model_path = '/home/bene/NNGamma/temp/'
model_name = 'local_test_fine'
model_name =  '220118-040417'
device = 'cuda'
#device = 'cpu'

solvent = "0"
solute = "C=O"

T = 298.15

x = np.linspace(0, 1, num=20)
T = np.linspace(T, T, 1)

data_loader_solute = smile2input_NRTL(solute, solvent, x ,T)
# %% Load model

model, config = load_model(model_path,model_name)
config.device = device
model = model.to(config.device)
criterion = nn.MSELoss()
# time evaluation
start = time.time()
__, gamma_solute, __, __ = evaluate(model, data_loader_solute, criterion, config)
end = time.time()
print("Evaluation time: ", end - start)
gamma_solute = np.reshape(gamma_solute, (int(len(gamma_solute)/2),2))

# plot train out over x 
fig, ax = plt.subplots()
# use dashed lines
ax.plot(x, gamma_solute[:,0], 'r--', label='solute')
ax.plot(x, gamma_solute[:,1], 'b--', label='solvent')


ax.set_title(str(T[0]) + " K")
ax.set_xlabel('x ' + solvent )
ax.set_ylabel('D')
ax.set_xlim([0,1])

plt.show()
plt.savefig('plot/simple_NRTL.png')