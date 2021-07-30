from nn_model import * 
from nn_dataloader import *
import torch
import torch.nn as nn
from dataclasses import dataclass
import wandb


import click

@click.command()
@click.option('--emb', default=128, help='Embeding size')
@click.option('--hid', default=256, help='Hidden layer size')
@click.option('--nlay', default=6, help='Number of transfprmer layers')
@click.option('--nhead', default=2, help='Number of heads')
@click.option('--drp', default=0.2, help='Dropout rate')
@click.option('--lr', default=0.0001, help='learning rate')
@click.option('--epo', default=100, help='Number of epochs')
@click.option('--btch', default=256, help='Batchsize')
@click.option('--set', default='TrainingData_test/', help='Location of dataset')

def main(emb, hid, nlay, nhead, drp, lr, epo, btch, set):
    
    wandb.init(project= 'gamma', entity='bene94')

    config = wandb.config

    config.device = torch.device('cuda')
    config.criterion = nn.MSELoss()
    
    config.padding_idx = 36
    config.ntokens =  37

    config.embed_size = emb
    config.hidden_size = hid
    config.num_layers = nlay
    config.num_heads = nhead
    config.dropout =  drp
    config.lr = lr
    config.epoch =  epo
    config.batch_size  = btch

    model = TransformerModel(config).to(config.device)
    wandb.watch(model)


    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.epoch, eta_min=config.lr/100)

    best_val_loss = float("inf")
    best_model = None

    # load training and validation data

    train_dataset = gamma_dataset(set, 'train')
    val_dataset = gamma_dataset(set, 'val')

    # train_dataset.train_data = train_dataset.train_data[0:16]
    # train_dataset.train_target = train_dataset.train_target[0:16]

    # val_dataset.train_data = val_dataset.train_data[0:16]
    # val_dataset.train_target = val_dataset.train_target[0:16]

    training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_data = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        train(model, criterion, optimizer, training_data, scheduler, epoch, wandb )
        val_loss = evaluate(model, val_data, criterion, config)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            .format(epoch, (time.time() - epoch_start_time),
                                        val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()


if __name__ == '__main__': 
    main()