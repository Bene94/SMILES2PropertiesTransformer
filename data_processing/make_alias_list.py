import rdkit as rd  
import numpy as np


from SmilesEnumerator import SmilesEnumerator
import data_processing_cosmo as dp


## script to make a list of aliases for the molecules in the database


def main():
    file_path = '../raw_data/cosmo_002/'
    solvent_list, solute_list, df_join = dp.load_exp_data(file_path) 
    smile_list = np.append(solvent_list, solute_list)
    smile_list = np.unique(smile_list)
    alias_dict = augment_smile(smile_list)
    np.save('../raw_data/alias/alias_dict.npy', alias_dict)



def augment_smile(solvent_list):
    # creates alternitive representations of smiles
    aug_fac = 5
    sme = SmilesEnumerator()
    
    alias_dict = {}

    #argument the data by a factor of aug_fac
    for i in range(len(solvent_list)):
        alias_list = [solvent_list[i]]
        for j in range(aug_fac):
            try:
                alias_list.append(sme.randomize_smiles(solvent_list[i]))
            except:
                pass
        alias_list = set(alias_list)
        alias_dict[solvent_list[i]] = alias_list
    return alias_dict

if __name__ == '__main__':
    main()
