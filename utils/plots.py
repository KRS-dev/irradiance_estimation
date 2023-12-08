import xarray
import random
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cf
import torch
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter, LongitudeLocator, LatitudeLocator

VMIN = 0
VMAX = 1360


def plot_patches(y, y_hat, n_patches = 4):

    shape = y.shape

    y_patches , y_hat_patches = y[:n_patches, : ,:], y_hat[:n_patches, :, :]

    y_patches = y_patches.reshape(-1, shape[1])
    y_hat_patches = y_hat_patches.reshape(-1, shape[1])

    fig, axes = plt.subplots(2,1, sharex=True, sharey=True)

    axes[0].imshow(y_patches, vmin=VMIN, vmax=VMAX)
    axes[0].set_title('Target')
    axes[1].imshow(y_hat_patches, vmin=VMIN, vmax=VMAX)
    axes[1].set_title('Prediction')

    return fig


def scatter_hist(x, y, ax, ax_histx, ax_histy, cax, output_var = 'SIS'):
    # no labels
    c_hist = 'grey'
    ax_histx.tick_params(axis="both", labelbottom=False, labelleft=False)
    ax_histy.tick_params(axis="both", labelleft=False, labelbottom=False)

    # the scatter plot:
    _,_,_, hist = ax.hist2d(x, y, bins=100, cmap='hot_r', norm=colors.LogNorm())
    ax.set_xlim([VMIN,VMAX])
    ax.set_ylim([VMIN,VMAX])
    ax.set_aspect('equal')
    ax.plot([VMIN, VMAX], [VMIN,VMAX], '--')

    ax.get_figure().colorbar(hist, ax=ax, cax=cax, )
    cax.set_ylabel('N')
    cax.yaxis.set_label_position('left')
    cax.tick_params(axis='both', which='both', direction='in', labelleft=True, labelright=False, left=True, right=False)

    # now determine nice limits by hand:
    binwidth = 0.01
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    # plot_importanced
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color=c_hist, density=True)
    ax_histy.hist(y, bins=bins, color=c_hist, orientation='horizontal', density=True)
    ax_histx.set_title(f'{output_var}')
    ax_histy.set_title(f'{output_var} prediction')


def prediction_error_plot(y, y_hat, output_var='SIS'):

    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.numpy()
    # Start with a square Figure.
    fig = plt.figure(figsize=(8, 8))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(3, 4,  width_ratios=(.3, .2, 4, 1), height_ratios=(1, 4, .5),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 2])
    ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
    cax = fig.add_subplot(gs[1,0])
    # Draw the scatter plot and marginals.
    scatter_hist(y, y_hat, ax, ax_histx, ax_histy, cax, output_var=output_var)

    return fig


def best_worst_plot(y:xarray.DataArray, y_hat:torch.Tensor, output_var='SIS', metric=None):

    y_hat = xarray.DataArray(data=y_hat.numpy(),
                             coords= y.coords)
    if metric is None:
        # calculate relative RMSE
        error = np.sqrt((y - y_hat)**2)
    else:
        error = metric(y, y_hat)    
    
    error_per_patch = error.mean(dim=['lat', 'lon'])
    idxmax = error_per_patch.idxmax()
    idxmin = error_per_patch.idxmin()
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(2,2, figsize=(12,6), subplot_kw={'projection': proj},)
    fig.suptitle(output_var)
    for axi in ax.flatten():
        axi.add_feature(cf.BORDERS)
        axi.xaxis.set_major_locator(LongitudeLocator())
        axi.yaxis.set_major_locator(LatitudeLocator())
        axi.xaxis.set_major_formatter(LongitudeFormatter())
        axi.yaxis.set_major_formatter(LatitudeFormatter())

    y.sel(time=idxmin).plot.imshow(
        ax=ax[0,0],
        vmin=VMIN,
        vmax=VMAX,
        transform=proj,
    )
    y_hat.sel(time=idxmin).plot.imshow(
        ax=ax[0, 1],
        vmin=VMIN,
        vmax=VMAX,
        transform=proj,
    )
    y.sel(time=idxmax).plot.imshow(
        ax=ax[1,0],
        vmin=VMIN,
        vmax=VMAX,
        transform=proj,
    )
    y_hat.sel(time=idxmax).plot.imshow(
        ax=ax[1,1],
        vmin=VMIN,
        vmax=VMAX,
        transform=proj,
    )
    fig.tight_layout()

    return fig

