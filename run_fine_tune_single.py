import os
import datetime
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d%H:%M:%S")
# remove spaces from now_str
now_str = now_str.replace(" ", "_")

model = '211220-192228'
model = 'untrained_model'
model = '211220-192228'

aug = 1 # 0: no aug, 1: aug
lval_int = 0
lval = 1

comand_s = 'xp run /home/bene/NNGamma/src/xprun_fine_mult.ron -p 3 --include-dirty' 
comand_e = ' -- -m ' + model + ' -l 1e-4 -x 200 '

epo = int(50)
now_str = now.strftime("%Y%m%d-%H%M%S")[2:]


n_split = 200

path = 'data_exp_noH2O_' + str(n_split) + '_V2'

save_path = 'f_t_'

save_path = save_path + model + '_' + now_str
group = str(n_split) + '_' + now_str
        

for i in range(0,200):
    if lval_int != 0:
        if i % lval_int == 0:
            lval = 1
    now_str = now.strftime("%Y%m%d-%H%M%S")[2:]
    xp_name = save_path + '_' + str(i)
    comand = comand_s + ' --name=' + xp_name + comand_e + '-p ' + path + ' -ow ' + str(1) + ' -e '
    comand = comand + str(epo) + ' -n ' + save_path + ' -g ' + group + ' -s ' + str(i) + ' -lval ' + str(lval)
    comand = comand + ' -w GNN_FT' + ' -b 128'
    os.system(comand)