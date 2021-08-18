
import matplotlib.pyplot as plt

import wandb
import torch.nn as nn

from nn_dataloader import *
from nn_model import *

#Function to compare the results

def make_histogram(prediciton, target, name, path):
    # make histogram of the output, use a normalised histogram constant bin width
    plt.clf()
    plt.hist(prediciton, bins=100, alpha=0.5, label='prediciton')
    plt.hist(target, bins=100, alpha=0.5, label='target', range=(max(prediciton), min(prediciton)))
    plt.legend(loc='upper right')
    plt.ylabel('count')
    plt.title(name)
    plt.savefig(path + 'hist_' + name)

def make_heatmap(prediciton, target, name, path):
    heatmap, xedges, yedges = np.histogram2d(target.squeeze(), prediciton.squeeze(), bins=500)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.savefig(path + 'heat_' + name)

def make_scatter(prediciton, target, name, path):
    plt.clf()
    plt.scatter(prediciton, target)
    plt.xlabel('predicted value')
    plt.ylabel('ground truth')
    plt.savefig(path + 'scat_' + name)

if __name__ == '__main__':
    config = wandb.config
    data_path = '/home/bene/TrainingData_red/'

    device = 'cuda'
    emb = 512
    hid = 1024
    nlay = 2
    nhead = 4
    drp = 0.1


    batch_size = 1024
    max_btch = 512


    config.embed_size = emb
    config.hidden_size = hid
    config.num_layers = nlay
    config.num_heads = nhead
    config.max_btch = max_btch
    config.dropout = drp
    config.padding_idx = 22
    config.ntokens =  24
    config.device = device
    config.data_path = data_path
    config.batch_size = batch_size

    model = TransformerModel(config).to(config.device)
    #model.load_state_dict(torch.load('../Models/202108151834trans_512_1024_2_4_1e-01_1e-01_1e-04_1024_5000.pth'))
    model.load_state_dict(torch.load('../Models/overfitt_test.pth'))
    model.eval()


    criterion = nn.MSELoss() 
    train_dataset = gamma_dataset(config.data_path, 'train')
    val_dataset = gamma_dataset(config.data_path, 'val')

    train_dataset.train_data = train_dataset.train_data[0:500]
    train_dataset.train_target = train_dataset.train_target[0:500]

    val_dataset.train_data = val_dataset.train_data[0:10]
    val_dataset.train_target = val_dataset.train_target[0:10]

    training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    val_data = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # load the pytorch model form ../models/


    # calculate the results for the training data and val data  
    total_loss_val , output_val = evaluate(model, val_data, criterion, config)
    total_loss_train , output_train = evaluate(model, training_data, criterion, config)


