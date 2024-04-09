import scipy
import xarray
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cf
import torch
from cartopy.mpl.gridliner import (
    LongitudeFormatter,
    LatitudeFormatter,
    LongitudeLocator,
    LatitudeLocator,
)
from adjustText import adjust_text
from scipy.stats import binned_statistic_2d
from torchmetrics import R2Score
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

VMIN = 0
VMAX = 1360


def plot_patches(y, y_hat, n_patches=4):
    shape = y.shape

    y_patches = y[:n_patches, :, :]
    y_hat_patches = y_hat[:n_patches, :, :]
    y_patches = np.transpose(y_patches, (2, 0, 1)).reshape(shape[1], -1)
    y_hat_patches = np.transpose(y_hat_patches, (2, 0, 1)).reshape(shape[1], -1)

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

    vmin = 0
    vmax = y_patches.reshape(-1).max()
    im2 = axes[0].imshow(y_patches, vmin=vmin, vmax=vmax)
    axes[0].set_title("Target")
    im = axes[1].imshow(y_hat_patches, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    fig.colorbar(im)
    fig.colorbar(im2)

    return fig


def scatter_hist(x, y, ax, ax_histx, ax_histy, cax, output_var="SIS"):
    # no labels
    c_hist = "grey"
    bins = 100
    ax_histx.tick_params(axis="both", labelbottom=False, labelleft=False)
    ax_histy.tick_params(axis="both", labelleft=False, labelbottom=False)

    # the scatter plot:
    _, _, _, hist = ax.hist2d(x, y, bins=bins, cmap="hot_r", norm=colors.LogNorm())
    # ax.set_xlim([VMIN,VMAX])
    # ax.set_ylim([VMIN,VMAX])
    # ax.set_aspect('equal')
    ax.plot([VMIN, VMAX], [VMIN, VMAX], "--")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    def bias_metric(x, y):
        return np.mean(y - x)

    ax.plot([VMIN, VMAX], intercept + slope*np.array([VMIN, VMAX]), 'b-.')


    metrics = {'R2 [-]':r2_score, 'Bias [W/m2]': bias_metric, 'MAE [W/m2]': mean_absolute_error, 'RMSE [W/m2]' : root_mean_squared_error}
    metrics = {key:val(x, y) for key, val in metrics.items()}
    txt = [f'{key} = {np.round(float(val), 3)}' for key,val in metrics.items()]
    txt = '\n'.join(txt)
    ax.annotate(txt, 
                xycoords='axes fraction',
                xy=[.05, .75])

    ax.get_figure().colorbar(
        hist,
        ax=ax,
        cax=cax,
    )
    cax.set_ylabel("N, Total={0}".format(len(x)))
    cax.yaxis.set_label_position("left")
    cax.tick_params(
        axis="both",
        which="both",
        direction="in",
        labelleft=True,
        labelright=False,
        left=True,
        right=False,
    )

    # now determine nice limits by hand:
    xymax = max(np.max(x), np.max(y))
    xymin = min(np.min(x), np.min(y))
    interval = xymax - xymin
    binwidth = interval / bins
    # lim = (int(interval/binwidth) + 1) * binwidth
    # plot_importanced
    bins = np.arange(xymin, xymax, binwidth)
    ax_histx.hist(x, bins=bins, color=c_hist, density=True)
    ax_histy.hist(y, bins=bins, color=c_hist, orientation="horizontal", density=True)
    ax_histx.set_title(f"{output_var}")
    ax_histy.set_ylabel(f"{output_var} prediction", loc='center')
    ax_histy.yaxis.set_label_position("right")


def prediction_error_plot(y, y_hat, output_var="SIS", title=None):

    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.cpu().numpy()
   
    
    # Start with a square Figure.
    fig = plt.figure(figsize=(8, 8))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=(0.3, 0.5, 4, 1),
        height_ratios=(1, 4, 0.5),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 2])
    ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
    cax = fig.add_subplot(gs[1, 0])
    cax.set_title(title)
    # Draw the scatter plot and marginals.
    scatter_hist(y, y_hat, ax, ax_histx, ax_histy, cax, output_var=output_var)
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 1100)

    return fig


def best_worst_plot(
    y: xarray.DataArray,
    y_hat: torch.Tensor,
    loss=None,
    output_var="SIS",
    metric=None,
    best=True,
):
    ds = y.to_dataset(dim="variable")
    ds = ds.rename({output_var: "y"})
    ds = ds.assign(y_hat=(("lat", "lon"), y_hat.numpy()))

    metric_name = str(metric).replace("()", "") if metric is not None else "RMSE"

    proj = ccrs.PlateCarree()
    cmap = "viridis"
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        subplot_kw={"projection": proj},
    )
    fig.suptitle(
        f'{"Best" if best else "Worst"} {output_var} prediction in {metric_name}'
    )
    for axi in ax.flatten():
        axi.coastlines()
        axi.xaxis.set_major_locator(LongitudeLocator())
        axi.yaxis.set_major_locator(LatitudeLocator())
        axi.xaxis.set_major_formatter(LongitudeFormatter())
        axi.yaxis.set_major_formatter(LatitudeFormatter())

    vmin = min(ds.y.min(), ds.y_hat.min())
    vmax = max(ds.y.max(), ds.y_hat.max())

    ds.y.plot.imshow(
        ax=ax[0],
        vmin=vmin,
        vmax=vmax,
        transform=proj,
        cmap=cmap,
    )
    ax[0].text(
        -0.4,
        0.55,
        f"{metric_name}={loss:.3f}",
        va="bottom",
        ha="center",
        rotation="vertical",
        rotation_mode="anchor",
        transform=ax[0].transAxes,
    )

    ds.y_hat.plot.imshow(
        ax=ax[1],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=proj,
    )

    for axi in ax.flatten():
        axi.coastlines()
        gl = axi.gridlines(
            crs=proj,
            linewidth=0.5,
            color="black",
            alpha=0.5,
            linestyle="--",
            draw_labels=True,
        )
        gl.top_labels = False
        gl.left_labels = True
        gl.right_labels = False
        gl.xlines = True
    fig.tight_layout()

    return fig


def SZA_error_plot(SZA, SIS_error):

    bins =  np.arange(0, 9/16*np.pi, np.pi/16)
    sza_bins_labels = np.rad2deg(bins)
    bin_indices = np.digitize(SZA.cpu(),bins)
    SZAs_errors = [SIS_error[bin_indices == i] for i in range(len(bins))]


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
        
    # ax1.set_xticks(bins[::5])
    SZAboxplot = ax.boxplot(
        SZAs_errors, 
        vert=True,
        sym='',
        notch=True,
        patch_artist=True, 
        labels=sza_bins_labels,
        zorder=3)
    
    ax.yaxis.grid(zorder=0)
    
    ax.set_title('Error distribution due to SZA')
    ax.set_ylabel('SIS error [w/m^2]')
    ax.set_xlabel('Solar Zenith Angle (degrees)')
    return fig, SZAboxplot
 

def dayofyear_error_plot(dayofyear, SIS_error):
    dayofyear_bins = np.arange(0, 365, 7)
    dayofyear_bins_labels = np.arange(0,53,1)
    bin_indices = np.digitize(dayofyear, dayofyear_bins)
    dayofyears_errors = [SIS_error[bin_indices == i] for i in range(len(dayofyear_bins))]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    dayofyearboxplot = ax.boxplot(
        dayofyears_errors,
        vert=True,
        sym='',
        notch=True,
        patch_artist=True,
        labels=dayofyear_bins_labels,
        zorder=3,
        )
    ax.yaxis.grid(zorder=0)
    ax.set_title('Error distribution due to seasonality')
    ax.set_ylabel('SIS error [w/m^2]')
    ax.set_xlabel('Week of the year')

    return fig, dayofyearboxplot

def latlon_error_plot(lat, lon, SIS_error, latlon_step=0.5):
    lat_bins =  np.arange(np.floor(torch.min(lat)), torch.max(lat) + latlon_step, latlon_step)
    lon_bins = np.arange(np.floor(torch.min(lon)), torch.max(lon) + latlon_step, latlon_step)
    mean_error, y_edge, x_edge, _ = binned_statistic_2d(lat, lon, SIS_error, bins = [lat_bins, lon_bins])

    proj = ccrs.PlateCarree()
    cmap = cm.bwr
    divnorm=colors.TwoSlopeNorm(vcenter=0.)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw={'projection': proj})
    pmesh = ax.pcolormesh(x_edge, y_edge, mean_error, shading='auto', transform=proj, cmap=cmap, norm=divnorm)
    fig.colorbar(pmesh)
    ax.set_title('Error distribution Location')
    ax.set_ylabel('Longitude')
    ax.set_xlabel('Latitude')
    ax.xaxis.set_major_locator(LongitudeLocator())
    ax.yaxis.set_major_locator(LatitudeLocator())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.add_feature(cf.COASTLINE, zorder=3)
    ax.add_feature(cf.BORDERS, zorder=3)

    return fig, pmesh

def plot_station_locations(lats, lons, names):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.stock_img()
    ax.set_extent([min(lons) -1, max(lons)+1, min(lats)-1, max(lats)+1], crs=ccrs.PlateCarree())
    for i in range(len(lats)):
        ax.plot(lons[i], lats[i], 'o', markersize=10, transform=ccrs.PlateCarree(), label=names[i])
    return fig, ax


def plot_station_scatter(lats, lons, metrics, station_nms, 
                         metric_nm, vmin=None, vmax=None, 
                         plot_text=True, cmap='plasma', norm=None,
                         title=None):

    # Create a Cartopy projection
    proj = ccrs.PlateCarree()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={'projection': proj})

    # Add map features
    ax.add_feature(cf.LAND)
    ax.add_feature(cf.OCEAN)
    ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.BORDERS, linestyle=':')
    ax.add_feature(cf.LAKES, alpha=0.5)
    # ax.add_feature(cf.RIVERS)

    # Plot the RMS errors with latlon
    sc = ax.scatter(lons, lats, s=200, c=metrics, norm=norm,
                    cmap=cmap, vmin=vmin, vmax=vmax, transform=proj, zorder=3)

    # plot texts
    if plot_text:
        text = [f'{round(val)}' for nm, val in zip(station_nms, metrics)]

        texts = []
        for i in range(len(text)):
            texts.append(ax.text(lons[i], lats[i], text[i]))
        
        adjust_text(
            texts,
            x=lons,
            y=lats,
            force_text=(.4, .4),
            expand=(2.5, 1.5),
            arrowprops=dict(arrowstyle='->', color='k', lw=0.5),)

    # Add a colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label=metric_nm)

    # Set the title and labels
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{metric_nm} over Europe')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    return fig
