import os

comand_s = 'xp run /home/bene/NNGamma/src/xprun_fine_mult.ron -p 1 --include-dirty' 
comand_e = ' -- -m 211126-160520 -l 1e-4 -x 200 '

n_list = [10, 50 , 100, 500, 1000, 5000]

n_list = [3, 5, 20 , 30, 40]


for n in n_list:
    epo = int(5000 * 20 / n)
    xp_name = 'n_f_' + str(n)
    path = 'data_exp_n/n_' + str(n)
    comand = comand_s + ' --name=' + xp_name + comand_e + '-p ' + path + ' -e ' + str(epo)
    os.system(comand)