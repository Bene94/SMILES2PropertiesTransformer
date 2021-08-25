
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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

def make_heatmap(prediciton, target, name, path):
    heatmap, xedges, yedges = np.histogram2d(target.squeeze(), prediciton.squeeze(), bins=2000)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
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
 
def makeColours( vals ):
    N = 10000
    mean = [np.mean(vals[0]),np.mean(vals[1])]
    cov = [[2,2],[0,2]]

    samples = np.random.multivariate_normal(mean,cov,N).T   
    densObj = kde( vals )
    
    vals = densObj.evaluate( vals )

    colours = np.zeros( (len(vals),3) )
    norm = Normalize( vmin=vals.min(), vmax=vals.max() )

    #Can put any colormap you like here.
    colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

    return colours

