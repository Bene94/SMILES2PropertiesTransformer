# %% 
# This is a simple script to evaluate the modle for custom data pairs
import os
import sys
from matplotlib import markers

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
model_path = '/local/home/bewinter/SPT/temp/'
model_path = '/local/home/bewinter/SPT/Models/'
model_name = 'local_test_fine'
model_name = '220331-141545' # NRTL-T
model_name = '220331-141603' # NRTL
#model_name = '220127-180116' #reg
device = 'cuda'
#device = 'cpu'

solvent = "N=CCCCO"
solute = "C=O"
#solute = "CCCCCCCC"
T = 298.15

x = np.linspace(0, 1, num=100)
T = np.linspace(T, T, 1)

# %% Load model

model, config = load_model(model_path,model_name)
config.device = device

if config.mode == 'NRTL' or config.mode == 'NRTL-T':
    data_loader_solute = smile2input_NRTL(solute, solvent, x ,T)
else:   
    data_loader_solute = smile2input(solute, solvent, x ,T)
    data_loder_solvent = smile2input(solvent, solute, x ,T) 

model = model.to(config.device)
criterion = nn.MSELoss()
# time evaluation
start = time.time()
__, gamma_solute, __, __ = evaluate(model, data_loader_solute, criterion, config)
if config.mode != 'NRTL' and config.mode != 'NRTL-T':
    __, gamma_solvent, __, __ = evaluate(model, data_loder_solvent, criterion, config)
end = time.time()

print("Evaluation time: ", end - start)
if config.mode == 'NRTL' or config.mode == 'NRTL-T':
    gamma_solute = np.reshape(gamma_solute, (int(len(gamma_solute)/2),2))
else:
    gamma_solvent = np.flip(gamma_solvent, axis=0)
    gamma_solute = np.stack((gamma_solute, gamma_solvent), axis=1)

# plot train out over x 
fig, ax = plt.subplots()
# use dashed lines
ax.plot(x, gamma_solute[:,0], 'r*', label='solute')
ax.plot(x, gamma_solute[:,1], 'b*', label='solvent')


ax.set_title(str(T[0]) + " K")
ax.set_xlabel('x ' + solvent )
ax.set_ylabel('ln gamma')
ax.set_xlim([0,1])

plt.show()
plt.savefig('plot/simple_NRTL.png')
