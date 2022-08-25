# %% 
# This is a simple script to evaluate the modle for custom data pairs
import os
import sys
import numpy as np
import pandas as pd
# add current path to sys.path
sys.path.append(os.getcwd())
# %% Imports
from transprop.trainer import *
from transprop.nn_dataloader import *
from transprop.load_model import *
from simple_evaluation_utils import *

# %% Parameters
model_path = '../temp/'
model_path = '../Models/'

model_name = 'model_512_brouwer'
file_name = 'test_file.csv'

from_file = False

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('='*50)
print('Device:', device)
print('='*50)

print('preparing data...')

# make input file alternativly load input here

if from_file:
    input_file = pd.read_csv('../data/' + file_name)
else:
    solute_list = ["CCCCC"]
    solvent_list = ["FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"]
    T = np.linspace(298.15, 298.15, len(solvent_list))
    input_file = pd.DataFrame(columns=['SMILES0','SMILES1','T'])
    input_file = pd.concat([input_file, pd.DataFrame({'SMILES0':solute_list, 'SMILES1':solvent_list, 'T':T})])

# create dataloader for input file
data_loader = smile2input(input_file.copy()) 

# %% Load model
print('='*50)
print('Loading model...')
model, config = load_model(model_path,model_name)
config.device = device
model = model.to(config.device)
criterion = nn.MSELoss()
print('='*50)
# time evaluation
print('Evaluating model...')

start = time.time()
__, gamma_solute, __, __ = evaluate(model, data_loader, criterion, config)
end = time.time()

print("Evaluation time: ", end - start)

# add results to input file
input_file['lnGamma'] = gamma_solute

# print results
print('='*50)
print('Results:')
print('='*50)
print(input_file)
print('='*50)

