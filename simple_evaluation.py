# %% 
# This is a simple script to evaluate the modle for custom data pairs


# %% Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from trainer import *
from nn_dataloader import *
from nn_model import * 
from load_model import *
from simple_evaluation_utils import *

# %% Parameters
model_path = '/home/bene/NNGamma/Models/'
model_name = '211101-151855'
device = 'cuda'
#device = 'cpu'

solvent = "(F)C(F)(F)(F)"
solute = "CCCCCC"

x = np.linspace(0, 1, num=1000)
T = np.linspace(298.15, 298.15, 1)

data_loader_solute = smile2input(solute, solvent, x ,T)
data_loader_solvent = smile2input(solvent, solute, x ,T)
# %% Load model

model, config = load_model(model_path,model_name)
config.device = device
model = model.to(config.device)
criterion = nn.MSELoss()
# time evaluation
start = time.time()
__,gamma_solute, __ = evaluate(model, data_loader_solute, criterion, config)
__,gamma_solvent, __ = evaluate(model, data_loader_solvent, criterion, config)
end = time.time()
print("Evaluation time: ", end - start)

# %%

# plot train out over x 
fig, ax = plt.subplots()
ax.plot(x, gamma_solute)
ax.plot(1-x, gamma_solvent)
ax.set_xlabel('x')
ax.set_ylabel('ln gamma')
ax.set_xlim([0,1])
ax.legend([solute, solvent])
plt.show()

plt.savefig('plot/simple_.png')