import os
import datetime
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d%H:%M:%S")
# remove spaces from now_str
now_str = now_str.replace(" ", "_")

model = '220512-142153' # name of the pretrained model

aug = 1 # 0: no aug, 1: aug
lval_int = 0 # sets the interval of runs that validate during training validating often will increase training time, large issue for small n in this function!!!!!!!
lval = 0

comand_s = 'xp run /local/home/bewinter/Paper_SPT/SPT/src/xprun_fine_mult.ron -p 3 --include-dirty' 
comand_e = ' -- -m ' + model + ' -l 1e-4 -x 200 '

n_list = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 3000, 4000, 5000]

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

    save_path = save_path + str(n)
    group = save_path

    for i in range(0,200):
        if lval_int != 0:
            if i % lval_int == 0:
                lval = 1

        now_str = now.strftime("%Y%m%d-%H%M%S")[2:]
        xp_name = save_path + '_' + str(i) + '_' + now_str
        comand = comand_s + ' --name=' + xp_name + str(i) + comand_e + '-p ' + path + ' -ow ' + str(1) + ' -e ' + str(epo) + ' -n ' + save_path + ' -g ' + group + ' -s ' + str(i) + ' -lval ' + str(lval)
        os.system(comand)