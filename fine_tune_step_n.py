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


comand_s = 'xp run /home/bene/NNGamma/src/xprun_fine_mult.ron -p 3 --include-dirty' 
comand_e = ' -- -m ' + model + ' -l 1e-4 -x 200 '

n_list = [10, 20, 30, 40, 50 , 100, 200, 300, 400, 500, 600, 700, 800, 1000] # , 2000, 3000, 4000, 5000]
#n_list = [2000, 3000, 4000, 5000]



for n in n_list:
    epo = int(200)
    path = 'data_exp_n/n_' + str(n)

    if model == 'untrained_model':
        save_path = 'n_ut_'
    else:
        save_path = 'n_f_'
    if aug == 1:
        save_path = save_path + 'aug_'
    else:
        save_path = save_path + ''

    xp_name = save_path + str(n) + '_' + now_str
    
    comand = comand_s + ' --name=' + xp_name + comand_e + '-p ' + path + ' -e ' + str(epo) + ' -n ' + save_path
    os.system(comand)