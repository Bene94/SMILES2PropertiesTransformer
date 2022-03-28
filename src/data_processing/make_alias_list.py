from progressbar.widgets import ETA
import rdkit as rd  
import numpy as np

import progressbar as pb

from SmilesEnumerator import SmilesEnumerator
import data_processing as dp


## script to make a list of aliases for the molecules in the database


def main():
    file_path = '../raw_data/'
    folder_names = ['t_cosmo']
    #folder_names = ['t_cosmo', 'brouwer_exp_c'] 
    df, complete_list1, __ = dp.load_exp_data(file_path, folder_names)
    #folder_names = ['elect']
    #df, complete_list2, __ = dp.load_exp_data(file_path, folder_names)

    #complete_list = complete_list1.append(complete_list2, ignore_index=True)

    complete_list = complete_list1

    alias_dict = augment_smile(complete_list.SMILE0.to_list())
    np.save('../raw_data/alias/alias_dict_brouwer.npy', alias_dict)



def augment_smile(solvent_list):
    # creates alternitive representations of smiles
    aug_fac = 8
    sme = SmilesEnumerator()
    
    alias_dict = {}

    bar = pb.ProgressBar(max_value=len(solvent_list), widgets=['Making alias: ', pb.Bar('=', '[', ']'), ' ', pb.Percentage(), pb.ETA()])
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
