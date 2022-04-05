from progressbar.widgets import ETA
import numpy as np
import csv 
import progressbar as pb

import data_processing as dp


## script to to extned a current vocab dict with new chars

def main():
    
    file_path, file_out, vocab_path, alias_path  =  dp.get_paths('', 'vocab') 

    org_vocab_dict = dp.load_vocab(vocab_path,'vocab_dict_tox21')
   
    folder_names = ['t_cosmo', 'brouwer']
    df, complete_list1, __ = dp.load_exp_data(file_path, folder_names)
    folder_names = ['elect']
    df, complete_list2, __ = dp.load_exp_data(file_path, folder_names)
    folder_names = ['ddb']
    df, complete_list3, __ = dp.load_exp_data(file_path, folder_names)
    complete_list = complete_list1.append(complete_list2.append(complete_list3, ignore_index=True), ignore_index=True)
    
    char_list = set(complete_list['SMILE0'].sum())

    # add new chars to the vocab dict
    for char in char_list:
        if char not in org_vocab_dict:
            org_vocab_dict[char] = len(org_vocab_dict)

    vocab_save_name = 'vocab_dict_ddb'
    # save the new vocab dict dilimiter is ' '
    with open(vocab_path + vocab_save_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        for key, value in org_vocab_dict.items():
            writer.writerow([key, value])

if __name__ == '__main__':
    main()
