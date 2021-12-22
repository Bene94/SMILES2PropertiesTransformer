import os
import datetime
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d%H:%M:%S")
# remove spaces from now_str
now_str = now_str.replace(" ", "_")

model =  '211220-192228'
model = 'untrained_model'

comand_s = 'xp run /home/bene/NNGamma/src/xprun_fine_mult.ron -p 1 --include-dirty' 
comand_e = ' -- -m ' + model + ' -l 1e-4 -x 200 '

n_list = [10, 20, 30, 40, 50 , 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, 3000, 4000, 5000]
#n_list = [2000, 3000, 4000, 5000]

for n in n_list:
    epo = int(5000 * 20 / n)
    xp_name = 'n_f_aug_' + str(n) + '_' + now_str
    path = 'data_exp_n/n_' + str(n)
    comand = comand_s + ' --name=' + xp_name + comand_e + '-p ' + path + ' -e ' + str(epo) + ' -n ' + 'n_ut_' + str(n)
    os.system(comand)