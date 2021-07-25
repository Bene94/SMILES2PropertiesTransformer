from nn_model import * 
from nn_main import *
import torch
import torch.nn as nn
from dataclasses import dataclass
import wandb




if __name__ == '__main__': 

    wandb.init(project= 'gamma', entity='bene94')


    config = wandb.config

    config.device = torch.device('cuda')
    config.ntokens =  37
    config.embed_size = 256
    config.hidden_size = 512
    config.num_layers = 8
    config.num_heads = 4
    config.dropout = 0.2
    config.criterion = nn.MSELoss()
    config.lr = 0.0001
    config.epoch = 50
    config.batch_size  = 128

    model = TransformerModel(config).to(config.device)

    wandb.watch(model)


    criterion = nn.MSELoss() # I guess i want something else
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    best_model = None

    # load training and validation data

    train_dataset = gamma_dataset('TrainingData_test/', 'train')
    val_dataset = gamma_dataset('TrainingData_test/', 'val')

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

        #scheduler.step()