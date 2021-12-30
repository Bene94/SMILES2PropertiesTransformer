import os  
path = "/mnt/xprun"
for root, dirs, files in os.walk(path):  
  for momo in dirs:  
    os.chown(os.path.join(root, momo), 1001, 1001)
  for momo in files:
    os.chown(os.path.join(root, momo), 1001, 1001)