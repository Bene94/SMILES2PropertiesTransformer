from nn_model import * 
from nn_dataloader import *
import torch
import torch.nn as nn
from dataclasses import dataclass
import wandb
import datetime


import click

@click.command()
@click.option('--emb', default=512, help='Embedding size')
@click.option('--hid', default=1024, help='Hidden layer size')
@click.option('--nlay', default=2, help='Number of transformer layers')
@click.option('--nhead', default=4, help='Number of heads')
@click.option('--drp', default=0.1, help='Dropout rate')
@click.option('--lr', default= 0.0001, help='Learning rate')
@click.option('--epo', default=0, help='Number of epochs')
@click.option('--btch', default=1024, help='Batchsize')
@click.option('--set', default='TrainingData_red/', help='Location of dataset')
@click.option('--wdecay', default=0.1, help='Weight decay')
@click.option('--local' , default=False, help='Using training data from local folder')
@click.option('--max_btch', default=128, help='Maximum batch size')
@click.option('--cuda', default=True, help='Using GPU')



def main(emb, hid, nlay, nhead, drp, lr, epo, btch, set, wdecay, local, max_btch, cuda):
    
    name = 'trans_' + str(emb) + '_' + str(hid) + '_' + str(nlay) + '_' + str(nhead) + '_' + str(drp) + '_' + str(wdecay) + '_' + str(lr) +  '_' + str(btch) + '_' + str(epo)
    
    wandb.init(project= 'gamma', entity='bene94', name=name)

    config = wandb.config

    config.device = torch.device('cuda')
    config.criterion = nn.MSELoss()
    


    config.padding_idx = 22
    config.ntokens =  24

    config.embed_size = emb
    config.hidden_size = hid
    config.num_layers = nlay
    config.num_heads = nhead
    config.dropout =  drp
    config.lr = lr
    config.epoch =  epo
    config.batch_size  = btch
    config.data_path = set
    config.weight_decay = wdecay
    config.max_btch = max_btch

    model = TransformerModel(config).to(config.device)
    wandb.watch(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = pytorch_total_params - emb *config.ntokens

    config.params = pytorch_total_params


    criterion = nn.MSELoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.epoch, eta_min=config.lr/10)

    best_val_loss = float("inf")
    best_model = None

    # load training and validation data

    if local:
        data_path = os.path.join(config.data_path)
    else:
        data_path = os.path.join('/mnt/xprun/data/')

    train_dataset = gamma_dataset(data_path, 'train')
    val_dataset = gamma_dataset(data_path, 'val')

    # train_dataset.train_data = train_dataset.train_data[0:16]
    # train_dataset.train_target = train_dataset.train_target[0:16]

    # val_dataset.train_data = val_dataset.train_data[0:16]
    # val_dataset.train_target = val_dataset.train_target[0:16]

    training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_data = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    overall_start_time = time.time()

    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        train(model, criterion, optimizer, training_data, scheduler, epoch, wandb )
        torch.cuda.empty_cache()
        val_loss = evaluate(model, val_data, criterion, config)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            .format(epoch, (time.time() - epoch_start_time),
                                        val_loss))
        print('-' * 89)
        wandb.log({"val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    print('-' * 89)
    print('| End of training | time: {:5.2f}s |'.format((time.time() - overall_start_time)))
    print('-' * 89)
    print("Best validation loss {:.4f}".format(best_val_loss))
    model = best_model
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    torch.save(model.state_dict(), '../Models/'+ date + name +'.pth')

if __name__ == '__main__': 
    main()
