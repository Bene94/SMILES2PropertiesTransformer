# SPT

This git contains the code for the paper "A SMILE is all you need: Predicting limiting activity coefficients from SMILES with natural language processing"

# Paper reproduction

## Pull git

`git clone git@github.com:Bene94/SMILES2PropertiesTransformer.git`

`cd SMILES2PropertiesTransformer/`

## Set up Environment

`conda create -n'SPT_env' python=3.9`

`conda activate SPT_env`

`pip install -r requirements_pip.txt`

Install a suitable torch version:

https://pytorch.org/

## Get raw data

Data update: Due to concerns about rights to the data the raw data and models are no longer available.

Get the content of the raw_data folder from https://polybox.ethz.ch/index.php/s/kyVOt3pwHW26PP4 and put it into the raw_data folder.

## Run training and fine-tuning

In the following, it is described how to reproduce the training and validation of the model from "A SMILE is all you need":

`cd src/`

Generate the alias list of SMILES. Warnings can be ignored:

`python data_processing/make_alias_list.py`
 
Generate the training data:

`python data_processing/data_processing.py -p inf_cosmo -p t_cosmo -s inf_t_cosmo`

Run the pretraining:

`python main_script.py`

Alternatively the pretrained model available in https://polybox.ethz.ch/index.php/s/kyVOt3pwHW26PP4 can be used.

Generate the validation sets:

`python data_processing/mult_split_n.py`

To run the validation, set the model name in `run_fine_tune_n_fold.py` to your pre-trained model name. The outputs are saved into SPT/out_fine_tune/. See wandb.ai for the name of the output folder (xp_name):

`python run_fine_tune_n_fold.py`

Run the evaluation script, set 'name' to the output name of the fine_tune:

`python plot/plot_results.py`

## Evaluating mixtures

To use the models to predict limiting activity coefficients use the script `misc/simple_evaluation.py`. Mixtures can either be put in manualy or loaded from an input file.

## Fine-tuning a model on new data

Alternative data for fine-tuning the model should be dropped in the raw_data folder as a tab-separated .csv. See the example files in https://polybox.ethz.ch/index.php/s/kyVOt3pwHW26PP4 for the format. To fine-tune a model, the new data has first to be processed.

`python data_processing/data_processing.py -p path_to_raw_data -s save_path`

Optional: It might be required to run `data_processing/make_alias_list.py` again to update the alias_list with the new data. 
After the data is processed, the fine-tuning can be executed:

`python fine_tune.py --model_name=model_name --data_path=input_data_path`

The trained modle will then be saved to: “SPT/Models/model_name_fine.*”

If the flag `--no_val` is set also the validation data is used for training.

# How to run the code more general

The code contains more function than used for the paper that can be used more flexible. In the following, we describe how to run the code, separated into two sections:

  1) Data processing
  2) Training


# Data processing

The folder `src/data_processing` contains multiple functions for data processing. These functions are used to convert `.csv` input files that contain SMILES of the species into input for the neural network in a memory efficient form so that even large datasets > 10 million fit into RAM. The files to be converted should be in a subfolder of `raw_data` and conform to the outline of the example files.

## Make alias list

The data processing requires a list that contains all SMILES and their aliases.

The function `src/make_alias_list.py` creates a dictionary that creates these alternative representations. To update the list, add the folder name of your data to the folder name list. If your data has different amounts of columns, they must be loaded separately. Otherwise all data gets dropped due to NaNs. The code currently loads two sets of data separately.

## COSMO data pre-processing

To process large datasets for pretraining use `data_processing.py`. This function will convert the data and split x% of data for Val_ext. The following flags are important:

  -p: path of the input in `raw_data`, can pass multiple times
  
  -s: name of the save folder in `data`
  
  --frac: fraction of data to val_ext
  
  --h2o: exclude water from val_ext
  
e.g., `$ python data_processing/data_processing.py -p t_cosmo -p inf_cosmo -s data`
  
## Experimental data pre-processing

For limited experimental data, n-fold cross validation should be used. To create the datasets, the function `mult_split_n_out.py` can be used. The data is split according to the methodology discussed in the paper.

## Other data processing

The function `mult_split_n.py` splits the data into x datasets containing n unique mixtures in the training set. This function creates datasets to examine scaling. 
  
# Training the model

The training of the model was conducted on a server running xprun for scheduling GPUs using the provided `.ron` files. However, xprun is not yet available. Still, training can be conducted by calling the python functions called by xprun directly. Runs are logged in `wandb.ai`. To run the code, create a `wandb.ai` account and follow the instruction to set account details in python. https://wandb.ai

## Pre-training

To create a new model and pretrain it, use the function `main_script.py`. The function takes multiple flags to set hyperparameters for training and model architecture. The trained model is saved in `Models`.

## Fine-Tuning

For fine-tuning using n-fold cross validation, use `fine_tune_mult_n_fold.py`. Set the folder path to the top folder containing the splits. Output of the evaluation is saved to `out_fine_tune`.

## Scaling

To evaluate the scaling of the model, use the function `run_fine_tune_step_n.py`.

# Using the model

To use the model, use the function `misc\simple_evaluation.py` and the trained models provided in the git.

# Data availability 

Data is no longer available due to licensing issues.

# Authors note

The very dyslexic author apologizes for any inconveniences caused by misspelled variables.
