
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize, LogNorm
from matplotlib import cm
from scipy.stats import gaussian_kde as kde


#Function to compare the results

def make_histogram(prediciton, target, name, path=''):
    # make histogram of the output, use a normalised histogram constant bin width
    plt.clf()
    plt.hist(target, bins=500, alpha=0.5, label='target', range=(min(target), max(target)))
    plt.hist(prediciton, bins=500, alpha=0.5, label='prediciton', range=(min(target), max(target)))
    plt.legend(loc='upper right')
    plt.ylabel('count')
    plt.xlim(-10,10)
    plt.title(name)
    plt.savefig(path + 'hist/hist_' + name)

def make_historgam_delta(prediciton, target, name = '', path = '', save=False):

    MSE = np.around(np.mean( (target - prediciton)**2 ),2)
    MAE = np.around(np.mean( np.abs(target - prediciton) ),2)

    perc_data = np.around(np.sum( (target - prediciton)**2 < 0.3**2 ) / len(target) * 100,2)

    delta = target - prediciton
    weights = np.ones_like(delta)/float(len(delta))
    plt.clf()
    plt.hist(delta, bins=21, alpha=0.5, label='delta', range=(-2,2), weights=weights)
    plt.ylabel('count')
    plt.xlabel('delta ln g')
    plt.xlim(-2,2)
    plt.title('MSE: ' + str(MSE) + ' MAE: ' + str(MAE) + ' perc_data: ' + str(perc_data))
    plt.savefig(path + 'hist/hist_delta_' + name)

# funciton that makes a histogram of the diff but for multiple data sets in a singel plot
def make_historgam_delta_mult(prediction_list, target_list, name_list, path = '', save=False):
    
    # make histogram of the output, use a normalised histogram constant bin width
    delta_list = []
    perc_data_list = []
    weights_list = []
    for i in range(len(prediction_list)):
        # bin and normalise the data
        delta = target_list[i] - prediction_list[i]
        weights = np.ones_like(delta)/float(len(delta))
        
        delta_list.append(delta)
        weights_list.append(weights)
        perc_data_list.append(np.sum( (target_list[i] - prediction_list[i])**2 < 0.3**2 ) / len(target_list[i]) * 100)

    plt.clf()
    plt.hist(delta_list, bins=21, alpha=0.5, range=(-2,2), weights=weights_list)
    plt.legend(name_list)
    plt.ylabel('percentage')
    plt.xlabel('delta ln gamma')
    plt.xlim(-2,2)
    # only two significant digits
    plt.title('perc_data: ' + str(np.around(perc_data_list,0)))
    plt.savefig(path + 'hist/hist_delta_mult')


def make_heatmap(prediciton, target, name = '', path = '', save=False):
    # make histogram of the output, use a normalised histogram constant bin width
    plt.clf()
    plt.hist2d(target, prediciton, bins=101, norm=LogNorm())
    heatmap, xedges, yedges = np.histogram2d(target.squeeze(), prediciton.squeeze(), bins=2000)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    plt.xlim(xedges[0],xedges[-1])
    plt.ylim(yedges[0],yedges[-1])
    plt.colorbar()

    MSE = np.around(np.mean( (target - prediciton)**2 ),2)
    MAE = np.around(np.mean( np.abs(target - prediciton) ),2)
    plt.title('MSE: ' + str(MSE) + ' MAE: ' + str(MAE))
    
    plt.ylabel('predicted value')
    plt.xlabel('ground truth')
    plt.savefig(path + 'heat/heat_' + name)

def make_scatter(prediciton, target, name = '', path = '', save=False):
    
    plt.clf()
    vals = []
    vals.append(target)
    vals.append(prediciton)
    colors = makeColours(vals)

    # use smaller dots for the points
    plt.scatter(target, prediciton, c=colors, s=1)
    plt.ylabel('predicted value')
    plt.xlabel('ground truth')

    MSE = np.around(np.mean( (target - prediciton)**2 ),2)
    MAE = np.around(np.mean( np.abs(target - prediciton) ),2)
    plt.title('MSE: ' + str(MSE) + ' MAE: ' + str(MAE))

    # add two lines to the plot to show +- 0.1 delta
    plt.plot([-20,20], [-19.7, 20.3], 'k--', lw=1)
    plt.plot([-20,20], [-20.3, 19.7], 'k--', lw=1)
    
    # set axix to max and min of target or prediciton
    min_target = min(target)
    max_target = max(target)
    min_pred = min(prediciton)
    max_pred = max(prediciton)
    plt.xlim(min(min_target, min_pred), max(max_target, max_pred))
    plt.ylim(min(min_target, min_pred), max(max_target, max_pred))

    if save:
        plt.savefig(path + 'scatter/scatter_' + name)
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
    # set nan to 0
    MSE[np.isnan(MSE)] = 0
    RMSE[np.isnan(RMSE)] = 0
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
    
    if save:
        plt.savefig(path + 'hist/MSE_' + name)
 
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
    
def plot_boxplot(n_list, mse_list_0, mse_list_1, mse_list_2, name = '', path = '', save=False):
    # make one figure containting three subplots with boxplots for each n
    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(hspace=0.5)
    
    # set axis to log scale
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    # axis limit for all subplots to 2

    ax[0].set_ylim(0,0.5)
    ax[1].set_ylim(0,0.5)
    ax[2].set_ylim(0,0.5)

    ax[0].set_title('Val 0')
    ax[0].boxplot(mse_list_0)
    ax[1].set_title('Val 1')
    ax[1].boxplot(mse_list_1)
    ax[2].set_title('Val 2')
    ax[2].boxplot(mse_list_2)
    # lable the boxplots with the n values
    ax[0].set_xticklabels(n_list)
    ax[1].set_xticklabels(n_list)
    ax[2].set_xticklabels(n_list)
    # set the title of the figure to name
    fig.suptitle(name)
    if save:
        plt.savefig(path + 'boxplot/boxplot_' + name)
