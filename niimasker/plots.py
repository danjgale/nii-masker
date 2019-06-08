"""Generate figures for visual report"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_roi
from nilearn.image import mean_img


def plot_timeseries(data, cmap):
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


def plot_mask(mask_img, func_img, cmap):
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
                 alpha=.75, cut_coords=np.linspace(-50, 60, num=n_cuts),
                 cmap=cmap, black_bg=False, annotate=False)
    g.annotate(size=8)
    g = plot_roi(mask_img,  bg_img=bg_img, display_mode='x', axes=axes[1],
                 alpha=.75, cut_coords=np.linspace(-60, 60, num=n_cuts),
                 cmap=cmap, black_bg=False, annotate=False)
    g.annotate(size=8)
    g = plot_roi(mask_img, bg_img=bg_img, display_mode='y', axes=axes[2],
                 alpha=.75, cut_coords=np.linspace(-90, 60, num=n_cuts),
                 cmap=cmap, black_bg=False, annotate=False)
    g.annotate(size=8)

    return fig


def make_figures(functional_images, timeseries_dir, mask_img):

    cmap = matplotlib.cm.get_cmap('nipy_spectral')

    for func in functional_images:

        func_img_name = os.path.basename(func).split('.')[0]

        timeseries_file = os.path.join(timeseries_dir,
                                       '{}_timeseries.tsv'.format(func_img_name))
        timeseries_data = pd.read_csv(timeseries_file, sep=r'\t', engine='python')
        fig = plot_timeseries(timeseries_data, cmap)
        fig.savefig(os.path.join(timeseries_dir, 'niimasker_data',
                                 '{}_timeseries_plot.png'.format(func_img_name)))

        fig = plot_mask(mask_img, func, cmap)
        fig.savefig(os.path.join(timeseries_dir, 'niimasker_data',
                                 '{}_atlas_plot.png'.format(func_img_name)))
