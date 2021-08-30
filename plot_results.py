
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib import cm
import numpy as np

from scipy.stats import gaussian_kde as kde

import wandb
import torch.nn as nn

from nn_dataloader import *
from nn_model import *

#Function to compare the results

def make_histogram(prediciton, target, name, path):
    # make histogram of the output, use a normalised histogram constant bin width
    plt.clf()
    plt.hist(target, bins=500, alpha=0.5, label='target', range=(min(target), max(target)))
    plt.hist(prediciton, bins=500, alpha=0.5, label='prediciton', range=(min(target), max(target)))
    plt.legend(loc='upper right')
    plt.ylabel('count')
    plt.xlim(-10,10)
    plt.title(name)
    plt.savefig(path + 'hist_' + name)

def make_heatmap(prediciton, target, name = '', path = '', save=False):
    # make histogram of the output, use a normalised histogram constant bin width
    plt.clf()
    plt.hist2d(target, prediciton, bins=200, norm=LogNorm())
    heatmap, xedges, yedges = np.histogram2d(target.squeeze(), prediciton.squeeze(), bins=2000)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.colorbar()
    # set title to MSE of the prediction scientific notation two decimal places
    MSE = np.around(np.mean( (target - prediciton)**2 ),2)
    plt.title('MSE: ' + str(MSE))
    plt.ylabel('predicted value')
    plt.xlabel('ground truth')
    plt.savefig(path + 'heat_' + name)

def make_scatter(prediciton, target, name = '', path = '', save=False):
    plt.clf()
    vals = []
    vals.append(target)
    vals.append(prediciton)
    colors = makeColours(vals)

    plt.scatter(target, prediciton, color=colors)
    plt.ylabel('predicted value')
    plt.xlabel('ground truth')
    if save:
        plt.savefig(path + 'scatter_' + name)
    else:
        plt.show()

def make_MSE_x(prediciton, target, name = '', path = '', save=False):
    # devide data in bins with bound -20,20
    bins = np.linspace(-20,20,100)
    # calculate the relative mean squared error for each bin
    MSE = np.zeros(len(bins)-1)
    RMSE = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        MSE[i] = np.mean( (target[(target>bins[i]) & (target<bins[i+1])] - prediciton[(target>bins[i]) & (target<bins[i+1])])**2 )
        RMSE[i] = np.mean( ((target[(target>bins[i]) & (target<bins[i+1])] - prediciton[(target>bins[i]) & (target<bins[i+1])]) / (np.mean((target[(target>bins[i]) & (target<bins[i+1])]))) **2))
    # plot the mean squared error
    plt.clf()

    plt.plot(bins[:-1], MSE)
    plt.ylabel('MSE')
    plt.xlabel('bin')
    plt.xlim(-20,20)
    plt.ylim(0,max(MSE))

    #add second axis with a histogram of the target
    plt.twinx()
    plt.hist(target, bins=100, alpha=0.5, label='target', range=(-20,20))
    plt.legend(loc='upper right')
    plt.ylabel('count')
    
    # set title to MSE of the prediction scientific notation two decimal places
    MSE = np.around(np.mean( (target - prediciton)**2 ),2)
    plt.title('MSE: ' + str(MSE))
    # add a third axis with the relative mean squared error per bin
    ax2 = plt.twinx()
    ax2.plot(bins[:-1], RMSE, color='r')
    ax2.set_ylabel('RMSE', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim(0,max(RMSE))
    

    if save:
        plt.savefig(path + 'MSE_' + name)
 
def makeColours( vals ):

    # sample 100000 points from vals 
    #vals[0] = np.random.choice(vals[0], size=1000, replace=True)
    #vals[1] = np.random.choice(vals[1], size=1000, replace=True)
    densObj = kde( vals )    
    vals = densObj.evaluate( vals )

    colours = np.zeros( (len(vals),3) )
    norm = Normalize( vmin=vals.min(), vmax=vals.max() )

    #Can put any colormap you like here.
    colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

    return colours

