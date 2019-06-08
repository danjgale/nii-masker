"""Generate figures for visual report"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.plotting import plot_roi
from nilearn.image import mean_img


def plot_timeseries(data, cmap):

    n_rois = data.shape[1]
    fig, axes = plt.subplots(n_rois, 1, sharex=True, figsize=(10, 20))

    cmap_vals = np.linspace(0, 1, num=n_rois)

    for i in np.arange(n_rois):

        ax = axes[i]
        y = data.iloc[:, i]
        x = y.index.values

        # draw plot
        ax.plot(x, y, c=cmap(cmap_vals[i]))
        ax.set_ylabel(data.columns[i], rotation=0, fontsize=10, labelpad=20,
                      horizontalalignment='right')

        # remove axes and ticks
        plt.setp(ax.spines.values(), visible=False)
        ax.tick_params(left=False, labelleft=False)
        ax.xaxis.set_visible(False)

    return fig


def plot_atlas(atlas_img, func_img, cmap):

    # compute mean of functional image
    func_name = os.path.basename(func_img).split('.')[0]
    bg_img = mean_img(func_img)

    n_cuts = 7
    fig, axes = plt.subplots(3, 1, figsize=(20, 6))

    g = plot_roi(atlas_img, bg_img=bg_img, display_mode='z', axes=axes[0],
                 alpha=.5, cut_coords=np.linspace(-50, 60, num=n_cuts),
                 cmap=cmap, black_bg=False, annotate=False)
    g.annotate(size=8)
    g = plot_roi(atlas_img,  bg_img=bg_img, display_mode='x', axes=axes[1],
                 alpha=.5, cut_coords=np.linspace(-60, 60, num=n_cuts),
                 cmap=cmap, black_bg=False, annotate=False)
    g.annotate(size=8)
    g = plot_roi(atlas_img, bg_img=bg_img, display_mode='y', axes=axes[2],
                 alpha=.5, cut_coords=np.linspace(-90, 60, num=n_cuts),
                 cmap=cmap, black_bg=False, annotate=False)
    g.annotate(size=8)

    return fig

