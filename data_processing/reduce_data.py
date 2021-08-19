import os
import progressbar
## list all file names in a directory then delete all files that contain Br, S, N, I, P, i, n, s, A in the name

def reduce_data(dir_path):
    for file in os.listdir(dir_path):

        if 'Br' in file:
            os.remove(dir_path + file)
        elif 'S' in file:
            os.remove(dir_path + file)
        elif 'N' in file:
            os.remove(dir_path + file)
        elif 'I' in file:
            os.remove(dir_path + file)
        elif 'P' in file:
            os.remove(dir_path + file)
        elif 'i' in file:
            os.remove(dir_path + file)
        elif 'n' in file:
            os.remove(dir_path + file)
        elif 's' in file:
            os.remove(dir_path + file)
        elif 'A' in file:
            os.remove(dir_path + file)
        elif 'F' in file:
            os.remove(dir_path + file)
        elif 'B' in file:
            os.remove(dir_path + file)
        elif 'Cl' in file:
            os.remove(dir_path + file)
        #if the name does not include C or c delte the file
        elif not ('C' in file or 'c' in file):
            os.remove(dir_path + file)

# datapath Input_Reduced

if __name__ == '__main__':
    datapath = 'Input_Reduced/'
    reduce_data(datapath)
        