# SPT

This git contains the code for the paper "A SMILE is all you need: Predicting limiting activity coefficients from SMILES with natural language processing"

# How to run the code

In the following how to run the code is described. This is separated into two sections.

  1) Data processing
  2) Training

# Data processing

The folder src/data_processig contains multiple function for data processing. These are used to convert .csv input files that contain SMILES of the species into input for the neural network in a memory efficient form so that even large datasets > 10 million fit into RAM. Files to be converted should be in a subfolder of raw_data and conform to the outline of the example files.

## Make alias list

The data processing requires a list that contains all SMILES and their aliases.

The function src/make_alias_list.py creates a dictionary that creates these alternative representations. To update the list, add the folder name of your data to the folder name list. If your data has different amounts of columns, they must be loaded separated as otherwise all data gets dropped due to NaNs. The code currently loads two sets of data separately.

## COSMO data preprocessing

To process large datasets for pretraining use data_processing.py this function will convert the data and split x% of data for Val_ext. The following flags are important:

  -p: path of the input in raw_data can pass multiple times
  
  -s: name of the save folder in data
  
  --frac: fraction of data to val_ext
  
  --h2o: exclude h2o from val_ext
  
E.g. $ python data_processing/data_processing.py -p t_cosmo -p inf_cosmo -s data
  
## Experimental data pre-processing

For limited experimental data n-fold cross validation should be used. To create the datasets the function mult_split_n_out.py can be used. The data gets split according to the methodology discussed in the paper.

## Other data processing

The function mult_split_n.py splits the data in x datasets containing n unique mixtures in the training set. This function creates datasets to examine scaling. 
  
# Training the model

The training on the model was conducted on a server running xprun for scheduling GPUs using the provided .ron files. However, xprun is not yet available. However, training can be conducted by calling the python functions called by xprun directly. Runs are logged in wandb.ai to run the code create a wandb.ai account and follow instruction to set account details in python. https://wandb.ai

## Pretraining

To create a new model and pretrain it uses the function mai_script.py. The function takes multiple flags to set hyperparameters for training and model architecture. The trained model is saved in Models.

## Fine-Tuning

For fine-tuning using n-fold cross validation use fine_tune_mult_n_fold.py set the folder path the top folder containing the splits. Output of the evaluation is saved to out_fine_tune.

## Scaling

To evaluate the scaling of the model us the function run_fine_tune_step_n.py.

## Authors note

The very dyslexic author apologizes for any inconveniences caused by misspelled variables. 
