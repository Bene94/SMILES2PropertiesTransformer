
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.colors import Normalize, LogNorm
from matplotlib import cm, markers
import matplotlib.ticker as mtick
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
    # create path if it does not exist
    if not os.path.exists(path + 'hist/'):
        os.makedirs(path + 'hist/')
    plt.savefig(path + 'hist/hist_' + name)
    #save as pdf
    plt.savefig(path + 'hist/hist_' + name + '.pdf')

def make_historgam_delta(prediciton, target, name = '', path = '', save=False):
    plt.rcParams['text.usetex'] = True
    MSE = np.around(np.mean( (target - prediciton)**2 ),2)
    MAE = np.around(np.mean( np.abs(target - prediciton) ),2)

    perc_data = np.around(np.sum( (target - prediciton)**2 < 0.3**2 ) / len(target) * 100,2)

    delta = target - prediciton
    weights = np.ones_like(delta)/float(len(delta))
    plt.clf()
    plt.hist(delta, bins=21, alpha=0.5, label='delta', range=(-2,2), weights=weights, edgecolor='black')
    plt.ylabel('count')
    plt.xlabel(r'$\Delta$ ln $\gamma_\infty$')
    plt.xlim(-2,2)
    plt.title(r'{}'.format( '\n MSE: ' + str(MSE) + '\n MAE: ' + str(MAE) + '\n $\Delta$ ln $\gamma_\infty < 0.3$: ' + str(perc_data) + '\%'))
    #plt.title('MSE: ' + str(MSE) + ' MAE: ' + str(MAE) + ' perc_data: ' + str(perc_data))
    # create path if it does not exist
    if not os.path.exists(path + 'hist/'):
        os.makedirs(path + 'hist/')

    plt.savefig(path + 'hist/hist_delta_' + name)
    #save as pdf
    plt.savefig(path + 'hist/hist_delta_' + name + '.pdf')
    plt.rcParams['text.usetex'] = False

# funciton that makes a histogram of the diff but for multiple data sets in a singel plot
def make_historgam_delta_mult(prediction_list, target_list, name_list, path = '', save=False, color_list = None):
    plt.rcParams['text.usetex'] = True
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
    # use color list if provided
    if color_list is not None:
        colours = color_list
    else:
        colours = makeColours( np.concatenate(delta_list) )
    plt.hist(delta_list, bins=11, alpha=0.5, range=(-1,1), weights=weights_list, edgecolor='black', color=colours)

    # set the color of the first 3 histograms to be similar
    # make text for legend by addeing the perc_data to the name
    lable_list = []
    for i in range(len(name_list)):
        lable_list.append(name_list[i])
    plt.legend(lable_list)
    # print percent data in a textbox in the top left corner
    plt.text(0.05, 0.9, '\% of $\Delta$ ln $\gamma_\infty$ $<$ 0.3: ', transform=plt.gca().transAxes)
    for i in range(len(perc_data_list)):
        plt.text(0.05, 0.85 - i*0.05, '\quad ' + name_list[i]+ ': '+  str(np.around(perc_data_list[i],1)), transform=plt.gca().transAxes, fontsize=8)

    plt.text(0.95, 0.4, '* addopted form Damayl et al. compared to\n the Dortmund Datenbank. UNIFAC has a\n $\Delta$ ln $\gamma_\infty$ $<$ 0.3 \% of 71.0', transform=plt.gca().transAxes, fontsize=6, horizontalalignment='right')

    plt.ylabel(r'Frac. of data')
    plt.xlabel(r"$\Delta$ ln $\gamma_\infty$")
    plt.xlim(-1,1)
    plt.x_tick_labels = [-1, -0.5, -0.3, 0, 0.3, 0.5, 1]
    # create path if it does not exist
    if not os.path.exists(path + 'hist/'):
        os.makedirs(path + 'hist/')
    # save high res image
    plt.savefig(path + 'hist/hist_delta_mult', dpi=900)
    # save as pdf
    plt.savefig(path + 'hist/hist_delta_mult.pdf')
    plt.rcParams['text.usetex'] = False

def make_heatmap(prediciton, target, name = '', title='', path = '', save=False):
    # make histogram of the output, use a normalised histogram constant bin width
    plt.rcParams['text.usetex'] = True
    plt.clf()
    plt.rcParams.update({'font.size': 18})
    plt.hist2d(target, prediciton, bins=301, norm=LogNorm())
    heatmap, xedges, yedges = np.histogram2d(target.squeeze(), prediciton.squeeze(), bins=2000)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # extend the bottom margin to make the labels fit
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.xlim(0,5)
    plt.ylim(0,5)
    plt.colorbar()

    MSE = np.around(np.mean( (target - prediciton)**2 ),2)
    MAE = np.around(np.mean( np.abs(target - prediciton) ),2)
    #add MSE and MEA to the top left corner in two lines and white text
    plt.text(0.05, 0.9, 'MSE: ' + str(MSE), transform=plt.gca().transAxes, color='white')
    plt.text(0.05, 0.8, 'MAE: ' + str(MAE), transform=plt.gca().transAxes, color='white')

    plt.title(title)
    plt.ylabel(r'$\ln\gamma^\infty_{\mathrm{prd.}}$')
    plt.xlabel(r'$\ln\gamma^\infty_{\mathrm{truth}}$')
    plt.tight_layout()
    # create path if it does not exist
    if not os.path.exists(path + 'heat/'):
        os.makedirs(path + 'heat/')
    plt.savefig(path + 'heat/heat_' + name, dpi=600)
    plt.savefig(path + 'heat/heat_' + name + '.pdf')
    plt.show()

    plt.rcParams['text.usetex'] = False


def make_scatter(prediciton, target, name = '', title = '',path = '', save=False): 
    plt.clf()
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    plt.rc
    vals = []
    vals.append(target)
    vals.append(prediciton)
    colors = makeColours(vals)

    # use smaller dots for the points
    plt.scatter(target, prediciton, c=colors, s=0.5)
    plt.ylabel(r'$\ln\gamma^\infty_\mathrm{prd.}$')
    plt.xlabel(r'$\ln\gamma^\infty_\mathrm{exp.}$')

    MSE = np.around(np.mean( (target - prediciton)**2 ),2)
    MAE = np.around(np.mean( np.abs(target - prediciton) ),2)
    # write MSE and MEA in the top left corner in two lines
    plt.text(0.05, 0.9, 'MSE: ' + str(MSE), transform=plt.gca().transAxes)
    plt.text(0.05, 0.8, 'MAE: ' + str(MAE), transform=plt.gca().transAxes)
    # add two lines to the plot to show +- 0.1 delta
    plt.plot([-20,20], [-19.7, 20.3], 'k--', lw=1)
    plt.plot([-20,20], [-20.3, 19.7], 'k--', lw=1)

    #plt.title(title)

    plt.ylim(-5, 16)
    plt.xlim(-5, 16)

    plt.xticks([-5, 0, 5, 10, 15])
    plt.yticks([-5, 0, 5, 10, 15])
    # increase fond size
    plt.tight_layout()
    # create path if it does not exist
    if not os.path.exists(path + 'scatter/'):
        os.makedirs(path + 'scatter/')
    if save:
        plt.savefig(path + 'scatter/scatter_' + name)
        plt.savefig(path + 'scatter/scatter_' + name + '.pdf')
    else:
        plt.show()
    plt.rcParams['text.usetex'] = False


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
        plt.savefig(path + 'hist/MSE_' + name + '.pdf')
 
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
    # increase font size
    
    if save:
        plt.savefig(path + 'boxplot/boxplot_' + name)
        plt.savefig(path + 'boxplot/boxplot_' + name + '.pdf')

def plot_err_sorted_combined(val_predction_0, val_target_0, val_predction_1, val_target_1, val_predction_2, val_target_2, name = '', path = '', save=False):
    # make one figure containting three subplots with boxplots for each n
    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(hspace=0.5)

    # calc error
    err_0 = np.abs(val_target_0 - val_predction_0)
    err_1 = np.abs(val_target_1 - val_predction_1)
    err_2 = np.abs(val_target_2 - val_predction_2)
    
    # axis limit for all subplots to 2

    ax[0].set_ylim(0,2)
    ax[1].set_ylim(0,2)
    ax[2].set_ylim(0,2)

    mse_list_0_sorted = np.sort(err_0)
    mse_list_1_sorted = np.sort(err_1)
    mse_list_2_sorted = np.sort(err_2)

    ax[0].set_title('Val 0')
    ax[0].plot(mse_list_0_sorted)
    ax[1].set_title('Val 1')
    ax[1].plot(mse_list_1_sorted)
    ax[2].set_title('Val 2')
    ax[2].plot(mse_list_2_sorted)


    ax[0].set_xlim(0,len(mse_list_0_sorted))
    ax[1].set_xlim(0,len(mse_list_1_sorted))
    ax[2].set_xlim(0,len(mse_list_2_sorted))

    # lable the boxplots with the n values

    # set the title of the figure to name
    fig.suptitle(name)
    if save:
        plt.savefig(path + 'sorted/mse_sorted_' + name)
        plt.savefig(path + 'sorted/mse_sorted_' + name + '.pdf')

def plot_err_curve_mult(prediction_list, target_list, name_list, color_list, line_style, damay_points,name = '', path = '', save=False):
    plt.rcParams['text.usetex'] = True
    # make one figure containting three subplots with boxplots for each n
    plt.rcParams.update({'font.size': 14})
    plt.clf()
    fig, ax = plt.subplots(1, 1, sharex=True)
    err = []
    for i in range(len(prediction_list)):
        err.append(np.abs(target_list[i] - prediction_list[i]))
    
    err_sorted = [np.sort(err[i]) for i in range(len(err))]

    plt.plot(damay_points[1], np.array(damay_points[0])/len(err[0])*100, color='darkorange', linestyle='--', marker='*', label='\emph{Damay et al. 2021}')
    
    percent = np.linspace(0,100,len(err_sorted[0]))

    for i in range(len(err_sorted)):
        plt.plot(err_sorted[i], percent, color=color_list[i], label=name_list[i], linestyle=line_style[i])

    plt.ylim(0,100)
    plt.xlim(0,1.9)
    plt.legend()
    # set the y axis to percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.ylabel('percentage of data points')
    plt.xlabel(r"$|\Delta\ln \gamma^\infty|$")
    plt.xticks([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9])
    plt.tight_layout()
    # plot vertical line at 0.3
    plt.axvline(x=0.3, color='lightgrey', linestyle='--')
    if save:
        plt.savefig(path + 'sorted/err_curve_mult_' + name, dpi=1200)
        plt.savefig(path + 'sorted/err_curve_mult_' + name + '.pdf', dpi=1200)

def plot_err_curve_mult_sund(prediction_list, target_list, name_list, color_list, line_style,name = '', path = '', save=False):
    plt.rcParams['text.usetex'] = True
    # change font size
    plt.rcParams.update({'font.size': 18})
    # make one figure containting three subplots with boxplots for each n
    plt.clf()
    fig, ax = plt.subplots(1, 1, sharex=True)
    err = []
    for i in range(len(prediction_list)):
        err.append(np.abs(target_list[i] - prediction_list[i]))
    
    err_sorted = [np.sort(err[i]) for i in range(len(err))]
    
    percent = np.linspace(0,100,len(err_sorted[0]))

    for i in range(len(err_sorted)):
        plt.plot(err_sorted[i], percent, color=color_list[i], label=name_list[i], linestyle=line_style[i])

    plt.ylim(0,100)
    plt.xlim(0,1.9)
    plt.legend()
    # set the y axis to percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.ylabel('percentage of data points')
    plt.xlabel(r"$|\Delta\ln \gamma^\infty|$")
    plt.xticks([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9])
    plt.tight_layout()
    # plot vertical line at 0.3
    plt.axvline(x=0.3, color='lightgrey', linestyle='--')
    if save:
        plt.savefig(path + 'sorted/err_curve_mult_' + name, dpi=1200)
        plt.savefig(path + 'sorted/err_curve_mult_' + name + '.pdf', dpi=1200)
    # calculate the percentage of data with error below 0.3 for all datasets
    err_below_03 = []
    for i in range(len(err_sorted)):
        err_below_03.append(len(err_sorted[i][err_sorted[i]<0.3])/len(err_sorted[i]))
    print(err_below_03)
