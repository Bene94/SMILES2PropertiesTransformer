import os
import datetime

import fine_tune_mult as ft

now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d%H:%M:%S")
now_str = now_str.replace(" ", "_")
now_str = now.strftime("%Y%m%d-%H%M%S")[2:]

model = '211220-192228'

aug = 1 # 0: no aug, 1: aug
lval_int = 0 # sets the interval of runs that validate during training validating often will increase training time, however not a large issue for small val datasets
lval = 10
xprun = True
n_split = 1000
epo = int(50)

path = 'data_exp_noH2O_' + str(n_split) + '_V2'
save_path = 'f_t_'
save_path = save_path + model + '_' + now_str
group = str(n_split) + '_' + now_str 

comand_s = 'xp run /home/bene/SPT/src/xprun_fine_mult.ron -p 2 --include-dirty' 
comand_e = ' -- -m ' + model + ' -l 1e-4 -x 200 '     

for i in range(0,n_split):
    if lval_int != 0:
        if i % lval_int == 0:
            lval = 1
        else:
            lval = 0
    now_str = now.strftime("%Y%m%d-%H%M%S")[2:]
    xp_name = save_path + '_' + str(i)
    comand = comand_s + ' --name=' + xp_name + comand_e + '-p ' + path + ' -ow ' + str(1) + ' -e '
    comand = comand + str(epo) + ' -n ' + save_path + ' -g ' + group + ' -s ' + str(i) + ' -lval ' + str(lval)
    comand = comand + ' -w GNN_FT' + ' -b 256'
    if xprun:
        os.system(comand)
    else: 
        ft.fine_tune(model, path, xp_name, 256, 50, 1e-5, 0, True, i, 1, lval, group, i, "GNN_FT")