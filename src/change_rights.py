# © 2022-2023, ETH Zurich, EPSE, Benedikt Winter
import os  
import shutil
path = "/mnt/xprun"
#path = "/home/bene/NNGamma"

def recursive_chown(path, owner, group):
    for dirpath, dirnames, filenames in os.walk(path):
        shutil.chown(dirpath, owner, group)
        for filename in filenames:
            shutil.chown(os.path.join(dirpath, filename), owner, group)

recursive_chown(path, 527563, 527563)