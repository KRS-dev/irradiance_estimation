import xarray
import random
import matplotlib.pyplot as plt
from matplotlib import colors
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

    ax.get_figure().colorbar(
        hist,
        ax=ax,
        cax=cax,
    )
    cax.set_ylabel("N")
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
    ax_histy.set_title(f"{output_var} prediction")


def prediction_error_plot(y, y_hat, output_var="SIS"):

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
        width_ratios=(0.3, 0.2, 4, 1),
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
    # Draw the scatter plot and marginals.
    scatter_hist(y, y_hat, ax, ax_histx, ax_histy, cax, output_var=output_var)

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
        axi.add_feature(cf.BORDERS)
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
        axi.add_feature(cf.BORDERS)
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
