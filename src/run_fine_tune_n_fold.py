import os
import datetime

import fine_tune_mult as ft

now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d%H:%M:%S")
now_str = now_str.replace(" ", "_")
now_str = now.strftime("%Y%m%d-%H%M%S")[2:]

model = 'model_512_cosmo' # name of the pretrained model
lval_int = 1 # sets the interval of runs that validate during training validating often will increase training time, however not a large issue for small val datasets
wandb_project = 'FT_Paper' # name of the wandb project
xp_path = os.getcwd() + 'xprun_fine_mult.ron' # path to the xprun file
path = 'data_exp_noH2O_1000' # path to the data
#path = 'data_exp_sund_200' # path to the data
lval = 10
xprun = False
n_split = 2000
epo = int(50)

save_path = 'f_t_'
save_path = save_path + model + '_' + now_str
group = str(n_split) + '_' + now_str 

comand_s = 'xp run ' + xp_path + ' -p 1 --include-dirty' 
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
    comand = comand + ' -w ' + wandb_project + ' -b 256'
    if xprun:
        os.system(comand)
    else: 
        ft.fine_tune(model, path, xp_name, 256, 50, 1e-5, 0, True, i, 1, lval, group, i, "GNN_FT")