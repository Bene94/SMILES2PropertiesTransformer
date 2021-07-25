## reate a large dataset where the token are all equal to the key

import numpy as np
import os


def create_data(length,max_key,padding):
    data = np.ones((max_key*length,length+1))
    data = data * padding
    
    for i in range(max_key):
        for j in range(length-1):
            data[i*length+j,0:2+j] = i

    return data


def save_batches(batches, folder_path, type):
    # save batches to csv for either train or val batches are numpy arrays
    for i, batch in enumerate(batches):
        file_path = os.path.join(folder_path, type + '_' +str(i) + '.csv')
        #formate first row float all other rows int
        fmt = '%.5f'+',%.0f'*(batch.shape[1]-1)
        np.save(file_path, batch)


data = create_data(256,37,36)
save_batches([data], 'TraningData_dummy/', 'train')
save_batches([data], 'TraningData_dummy/', 'val')

print(data)