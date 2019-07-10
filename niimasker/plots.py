"""Generate figures for visual report"""

import os
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_roi, plot_matrix
from nilearn.image import mean_img
from nilearn.connectome import ConnectivityMeasure


def _plot_roi_timeseries(data, cmap):
    """Plot timeseries traces for each extracted ROI.

    Parameters
    ----------
    data : pandas.core.DataFrame
        Timeseries data extracted from niimasker.py
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap to use.
    Returns
    -------
    matplotlib.pyplot.figure
        Timeseries plot
    """
    n_rois = data.shape[1]
    fig, axes = plt.subplots(n_rois, 1, sharex=True,
                             figsize=(15, int(n_rois / 5)))

    cmap_vals = np.linspace(0, 1, num=n_rois)

    if ((any([x.startswith('roi') for x in data.columns])) |
           (any([x.startswith('voxel') for x in data.columns]))):
           pass
    else:
        data.columns = ['{}. '.format(y) + x
                        for y, x in enumerate(data.columns)]

    for i in np.arange(n_rois):

        ax = axes[i]
        y = data.iloc[:, i]
        x = y.index.values

        # draw plot
        ax.plot(x, y, c=cmap(cmap_vals[i]))
        ax.set_ylabel(data.columns[i], rotation='horizontal',
                      position=(-.1, -.1), ha='right')

        # remove axes and ticks
        plt.setp(ax.spines.values(), visible=False)
        ax.tick_params(left=False, labelleft=False)
        ax.xaxis.set_visible(False)

    fig.tight_layout()
    return fig


def _plot_carpet(data):

    plot_data = data.transpose().values
    vlim = np.max(np.abs(plot_data))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 8),
                           gridspec_kw={'height_ratios': [.2, 1]})
    # mean BOLD plot
    x = np.arange(plot_data.shape[1])
    y = np.mean(plot_data, axis=0)
    y_err = np.std(plot_data, axis=0)
    axes[0].plot(x, y, c='k')
    axes[0].fill_between(x, y - y_err, y + y_err, facecolor='gray', alpha=.4)
    axes[0].set_ylabel('Mean BOLD\n (±1 SD)')

    # carpet plot
    im = axes[1].imshow(plot_data, cmap='coolwarm', aspect='auto', vmin=-vlim,
                   vmax=vlim)
    cbar = axes[1].figure.colorbar(im, ax=axes[1], orientation='horizontal',
                                   fraction=.05)
    axes[1].set_ylabel('Voxelwise BOLD')
    axes[1].set_xlabel('Volumes')
    fig.tight_layout()
    return fig


def plot_timeseries(data, voxelwise, fname, cmap):

    if voxelwise:
        fig = _plot_carpet(data)
    else:
        fig = _plot_roi_timeseries(data, cmap)

    fname += '_timeseries_plot.png'
    fig.savefig(fname, bbox_inches='tight')
    plt.close()
    return os.path.abspath(fname)


def plot_overlay(mask_img, func_img, fname, cmap):
    """Overlay mask/atlas on mean functional image.

    Parameters
    ----------
    atlas_img : str
        File name of atlas/mask image
    func_img : str
        File name of 4D functional image that was used in extraction.
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap to use.

    Returns
    -------
    matplotlib.pyplot.figure
        Atlas/mask plot
    """
    # compute mean of functional image
    bg_img = mean_img(func_img)

    n_cuts = 7
    fig, axes = plt.subplots(3, 1, figsize=(15, 6))

    g = plot_roi(mask_img, bg_img=bg_img, display_mode='z', axes=axes[0],
                 alpha=.66, cut_coords=np.linspace(-50, 60, num=n_cuts),
                 cmap=cmap, black_bg=True, annotate=False)
    g.annotate(size=8)
    g = plot_roi(mask_img,  bg_img=bg_img, display_mode='x', axes=axes[1],
                 alpha=.66, cut_coords=np.linspace(-60, 60, num=n_cuts),
                 cmap=cmap, black_bg=True, annotate=False)
    g.annotate(size=8)
    g = plot_roi(mask_img, bg_img=bg_img, display_mode='y', axes=axes[2],
                 alpha=.66, cut_coords=np.linspace(-90, 60, num=n_cuts),
                 cmap=cmap, black_bg=True, annotate=False)
    g.annotate(size=8)

    fname += '_mask_overlay.png'
    fig.savefig(fname, bbox_inches='tight')
    plt.close()
    return os.path.abspath(fname)


def plot_connectome(data, fname, tick_cmap):

    cm = ConnectivityMeasure(kind='correlation')
    mat = cm.fit_transform([data.values])[0]

    if data.shape[1] < 200:
        labels = ['{} '.format(x) + u"\u25A0" for x in np.arange(data.shape[1])]
    else:
        # exclude numerical labels with large atlases
        labels = [u"\u25A0"] * data.shape[1]

    fig, ax = plt.subplots(figsize=(15, 15))
    im = plot_matrix(mat, labels=labels, tri='lower', figure=fig, vmin=-1, vmax=1,
                     cmap='coolwarm', colorbar=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.figure.colorbar(im, ax=ax, fraction=0.03)

    cmap_vals = np.linspace(0, 1, num=len(labels))
    for i, lab in enumerate(labels):
        ax.get_xticklabels()[i].set_color(tick_cmap(cmap_vals[i]))
        ax.get_yticklabels()[i].set_color(tick_cmap(cmap_vals[i]))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,
                       fontdict={'verticalalignment': 'top', 'horizontalalignment': 'center'})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontdict={'verticalalignment': 'center', 'horizontalalignment': 'right'})

    fname += '_connectome.png'
    fig.savefig(fname, bbox_inches='tight')
    plt.close()
    return os.path.abspath(fname)


def plot_regressor_corr(data, regressors, fname, cmap):

    # regressor by roi matrix
    result = np.zeros((regressors.shape[1], data.shape[1]))
    for i in np.arange(regressors.shape[1]):
        for j in np.arange(data.shape[1]):
            regressor = regressors.values[:, i]
            timeseries = data.values[:, j]
            r, p = pearsonr(timeseries, regressor)
            result[i, j] = r

    cmap_vals = np.linspace(0, 1, num=data.shape[1])
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(result, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=.05)
    ax.set_xticks(np.arange(data.shape[1]))

    if data.shape[1] < 200:
        labels = ['{} '.format(x) + u"\u25A0" for x in np.arange(data.shape[1])]
    else:
        # exclude numerical labels with large atlases
        labels = [u"\u25A0"] * data.shape[1]

    ax.set_xticklabels(labels, rotation=90,
                       fontdict={'verticalalignment': 'top', 'horizontalalignment': 'center'})
    for i, lab in enumerate(data.columns):
        ax.get_xticklabels()[i].set_color(cmap(cmap_vals[i]))
    ax.set_yticks(np.arange(regressors.shape[1]))
    ax.set_yticklabels(regressors.columns, rotation=0,
                       fontdict={'verticalalignment': 'center', 'horizontalalignment': 'right'})
    fname += '_regressors.png'
    fig.savefig(fname, bbox_inches='tight')
    plt.close()
    return os.path.abspath(fname)
