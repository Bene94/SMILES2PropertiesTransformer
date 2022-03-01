import os

import os  
path = "/NNgamma/"  
for root, dirs, files in os.walk(path):  
  for momo in dirs:  
    os.chown(os.path.join(root, momo), "bene", -1)
  for momo in files:
    os.chown(os.path.join(root, momo), "bene", -1)