# SPT

This git contains the code for the paper "A SMILE is all you need"

# How to run

In the follwoing how to run the code is discribed. This is seperated into two sections.

  1) Data processing
  2) Training

# Data processing

The folder src/data_processig containts multible function for data processing. These are used to convert .csv input files that contain SMILES of the species into input for the neural network in a memorry efficent form so that even large datasets > 10 milion fits into RAM. Files to be converted should be in a subfolder of raw_data and confrom to the outline of the example files.

## Make alisas list

The dataprossing requiers a list that contains all SMILES and their aliases.

The function src/make_alias_list.py creates a dictonary that creates these alternative representations. To update the list add the folder name of your data to the folder name list. If your data has diffrent amounts of collumns they have to be loaded seperated as otherwise all data gets droped due to NaNs. The code currently loads two sets of data seperattly.

## COSMO data preprocessing

To process large datasets for pretraining use data_processing.py this function will convert the data and split x% of data for Val_ext. The following flags are important:

  -p: path of the input in raw_data can pass multible
  
  -s: name of the save folder in data
  
  --frac: fraction of data to val_ext
  
  --h2o: exclude h2o from val_ext
  
## Experimental data prepossing

For limited experimental data n-fold corss validation should be used. To create the datasets the function mult_split_n_out.py can be used. The data gets split acording to our methodology discussed in the paper.

## Other data processing

The function mult_split_n.py splits the data in x datasets containing n unique mixtures in the training set. This function is used to created datasets to look at scaling. 
  
  
# Training the model

The training on the model was conducted on a server running xprun for sceduling GPUs using the provided .ron files. However, this program is not yet avaialbe. However training can b conducted by siply calling the python functions called by xprun directly. Runs are loged in wandb.ai to run the code create a wandb.ai account and follow instruction to set account details in python. 

## Pretraining

To create a new model and pretrain it use the function mai_script.py. The function takes multible flags to set hyperparameters for training and model architecutre. The trained model is saved in Models 

## Fine Tuning

For fine tuning using n-fold cross validation use fine_tune_mult_n_fold.py set the folder path the the top folder containing the splits. Output of the evaluation is saved to out_fine_tune

## Scaling

To evaluate the scaling of the model us the function run_fine_tune_step_n.py 




