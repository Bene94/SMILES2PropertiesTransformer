## load the pytorch transformer model and the cofiguration file
import torch.nn as nn
import os
import sys

sys.path.append(os.getcwd())

import click

from plot.plot_results import *
from plot.plot_results import *

from transprop.trainer import *
from transprop.load_model import *
from transprop.nn_dataloader import *



@click.command()

@click.option('--name','-n', default='211220-192228', help='Name of the modle')
@click.option('--data','-d', default='data', help='Path to the data if empty use datapath from modle config')

@click.option('--calc','-c', default=True, help='Calculate results and eval')
@click.option('--plot','-p', default=True, help='Plot results')
@click.option('--save','-s', default=True, help='Save results')


def main(name,data,calc,plot,save):
    
    if os.environ.get('XPRUN_NAME') is not None:
        path_model = "/mnt/xprun/out/"
        data_path = "/mnt/xprun/data/" + data + "/"
        save_path = "/mnt/xprun/out/" + name + "/"
    else:
        path_model = '../Models/'
        data_path = '../data/' + data + '/'
        save_path = '../out/' + name +  '/'
        plot_path = 'plot/'

    model, config = load_model(path_model,name)

    if calc:

        #model to devide
        print(config.data_path)
        model = model.to('cuda')
        criterion = nn.MSELoss()

        print('-' * 89)
        print('Loading Data...')
        print('-' * 89)

        train_dataset = gamma_dataset(data_path, 'train', config, aug = False)
        val_0_dataset = gamma_dataset(data_path, 'val_0', config, aug = False)
        val_1_dataset = gamma_dataset(data_path, 'val_1', config, aug = False)
        val_2_dataset = gamma_dataset(data_path, 'val_2', config, aug = False)

        training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        val_0_data = DataLoader(val_0_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        val_1_data = DataLoader(val_1_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        val_2_data = DataLoader(val_2_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        print('-' * 89)
        print('Calculating Traning...')
        print('-' * 89)

        train_loss, train_out, train_target, train_in = evaluate(model, training_data, criterion, config)

        print('-' * 89)
        print('Calculating Validation...')
        print('-' * 89)

        val_0_loss, val_0_out, val_0_target, val_0_in = evaluate(model, val_0_data, criterion, config)
        val_1_loss, val_1_out, val_1_target, val_1_in = evaluate(model, val_1_data, criterion, config)
        val_2_loss, val_2_out, val_2_target, val_2_in = evaluate(model, val_2_data, criterion, config)

        train_target = train_target.squeeze()
        train_out = train_out.squeeze()

        val_0_target = val_0_target.squeeze()
        val_0_out = val_0_out.squeeze()
        val_1_target = val_1_target.squeeze()
        val_1_out = val_1_out.squeeze()
        val_2_target = val_2_target.squeeze()
        val_2_out = val_2_out.squeeze()

        print("Training loss: ", train_loss)
        print("Validation loss: ", val_0_loss)
        print("Validation loss: ", val_1_loss)
        print("Validation loss: ", val_2_loss)

        if save:
            # check if save path exists
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # save the results to a file
            np.save(save_path + 'train_out.npy', train_out)
            np.save(save_path + 'train_target.npy', train_target)
            np.save(save_path + 'train_smile.npy', train_in[0])
            np.save(save_path + 'train_xT.npy', train_in[1])
            np.save(save_path + 'train_x.npy', train_in[2])

            np.save(save_path + 'val_0_out.npy', val_0_out)
            np.save(save_path + 'val_0_target.npy', val_0_target)
            np.save(save_path + 'val_0_smile.npy', val_0_in[0])
            np.save(save_path + 'val_0_xT.npy', val_0_in[1])
            np.save(save_path + 'val_0_x.npy', val_0_in[2])

            np.save(save_path + 'val_1_out.npy', val_1_out)
            np.save(save_path + 'val_1_target.npy', val_1_target)
            np.save(save_path + 'val_1_smile.npy', val_1_in[0])
            np.save(save_path + 'val_1_xT.npy', val_1_in[1])
            np.save(save_path + 'val_1_x.npy', val_1_in[2])

            np.save(save_path + 'val_2_out.npy', val_2_out)
            np.save(save_path + 'val_2_target.npy', val_2_target)
            np.save(save_path + 'val_2_smile.npy', val_2_in[0])
            np.save(save_path + 'val_2_xT.npy', val_2_in[1])
            np.save(save_path + 'val_2_x.npy', val_2_in[2])

    else:

        train_out = np.load(save_path + 'train_out.npy')
        train_target = np.load(save_path + 'train_target.npy')
        val_0_out = np.load(save_path + 'val_0_out.npy')
        val_0_target = np.load(save_path + 'val_0_target.npy')
        val_1_out = np.load(save_path + 'val_1_out.npy')
        val_1_target = np.load(save_path + 'val_1_target.npy')
        val_2_out = np.load(save_path + 'val_2_out.npy')
        val_2_target = np.load(save_path + 'val_2_target.npy')


    if plot:
        make_MSE_x(train_out, train_target, name = "train", save = True, path=plot_path)
        make_MSE_x(val_0_out, val_0_target, name = "val_0", save = True, path=plot_path)
        make_MSE_x(val_1_out, val_1_target, name = "val_1", save = True, path=plot_path)
        make_MSE_x(val_2_out, val_2_target, name = "val_2", save = True, path=plot_path)

        make_heatmap(train_out, train_target, name = "train", title='$val_\mathrm{train}$', save = True, path=plot_path)
        make_heatmap(val_0_out, val_0_target, name = "val_0", title='$val_\mathrm{ext}$', save = True, path=plot_path)
        make_heatmap(val_1_out, val_1_target, name = "val_1", title='$val_\mathrm{edge}$', save = True, path=plot_path)
        make_heatmap(val_2_out, val_2_target, name = "val_2", title='$val_\mathrm{int}$', save = True, path=plot_path)

        make_historgam_delta(train_out, train_target, name = "train", save = True, path=plot_path)
        make_historgam_delta(val_0_out, val_0_target, name = "val_0", save = True, path=plot_path)
        make_historgam_delta(val_1_out, val_1_target, name = "val_1", save = True, path=plot_path)
        make_historgam_delta(val_2_out, val_2_target, name = "val_2", save = True, path=plot_path)

        if len(train_out) < 30000:
            print('-' * 89)
            print('Make Scatter...')
            print('-' * 89)
            train_out = np.concatenate((train_out, val_0_out, val_1_out, val_2_out), axis=0)
            train_target = np.concatenate((train_target, val_0_target, val_1_target, val_2_target), axis=0)
            
            make_scatter(train_out, train_target, name = "train", title="$pretrained$",save = True, path=plot_path)

if __name__ == '__main__':
    main()
