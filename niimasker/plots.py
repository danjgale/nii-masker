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
from niimasker.report import make_report


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


def plot_carpet(data):

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
    return fig


def plot_connectome(data, tick_cmap, labels=None):

    cm = ConnectivityMeasure(kind='correlation')
    mat = cm.fit_transform([data])[0]

    if data.shape[1] < 200:
        labels = ['{} '.format(x) + u"\u25A0" for x in np.arange(data.shape[1])]
    else:
        # exclude numerical labels with large atlases
        labels = [u"\u25A0"] * data.shape[1]

    fig, ax = plt.subplots(figsize=(15, 15))
    plot_matrix(mat, labels=labels, tri='lower', figure=fig, vmin=-1, vmax=1,
                cmap='coolwarm')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cmap_vals = np.linspace(0, 1, num=len(labels))
    for i, lab in enumerate(labels):
        ax.get_xticklabels()[i].set_color(tick_cmap(cmap_vals[i]))
        ax.get_yticklabels()[i].set_color(tick_cmap(cmap_vals[i]))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,
                       fontdict={'verticalalignment': 'top', 'horizontalalignment': 'center'})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontdict={'verticalalignment': 'center', 'horizontalalignment': 'right'})
    return fig


def plot_regressor_correlations(data, regressors, cmap):

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
    return fig


def make_figures(parameters, as_carpet=False, connectivity_metrics=True):

    functional_images = parameters['input_files']
    timeseries_dir = parameters['output_dir']
    mask_img = parameters['mask_img']

    figure_dir = os.path.join(timeseries_dir, 'niimasker_data/figures/')
    os.makedirs(figure_dir, exist_ok=True)
    report_dir = os.path.join(timeseries_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)

    for i, func in enumerate(functional_images):

        func_img_name = os.path.basename(func).split('.')[0]
        timeseries_file = os.path.join(timeseries_dir,
                                       '{}_timeseries.tsv'.format(func_img_name))
        timeseries_data = pd.read_csv(timeseries_file, sep=r'\t', engine='python')

        if ((any([x.startswith('roi') for x in timeseries_data.columns])) |
           (any([x.startswith('voxel') for x in timeseries_data.columns]))):
           pass
        else:
            timeseries_data.columns = ['{}. '.format(y) + x
                                 for y, x in enumerate(timeseries_data.columns)]

        n_rois = timeseries_data.shape[1]
        if (n_rois == 1) | as_carpet:
            roi_cmap = matplotlib.cm.get_cmap('gist_yarg')
        else:
            roi_cmap = matplotlib.cm.get_cmap('nipy_spectral')

        # plot and save timeseries
        if as_carpet:
            fig = plot_carpet(timeseries_data)
            bbox_inches = 'tight'
        else:
            fig = plot_timeseries(timeseries_data, roi_cmap)
            bbox_inches = None
        timeseries_fig = os.path.join(figure_dir,
                                      '{}_timeseries_plot.png'.format(func_img_name))
        fig.savefig(timeseries_fig, bbox_inches=bbox_inches)
        # plot and save mask overlay
        fig = plot_mask(mask_img, func, roi_cmap)
        overlay_fig = os.path.join(figure_dir,
                                 '{}_atlas_plot.png'.format(func_img_name))
        fig.savefig(overlay_fig, bbox_inches='tight')

        # place-holder for connectivity plots
        if connectivity_metrics:
            fig = plot_connectome(timeseries_data.values, roi_cmap,
                                  timeseries_data.columns)
            connectome_fig = os.path.join(figure_dir, '{}_connectome_plot.png'.format(func_img_name))
            fig.savefig(connectome_fig, bbox_inches='tight')

            if parameters['regressor_files'] is not None:

                all_regressors = pd.read_csv(parameters['regressor_files'][i],
                                             sep=r'\t', engine='python')
                regressors = all_regressors[parameters['regressor_names']]
                if parameters['discard_scans'] is not None:
                    n_scans = parameters['discard_scans']
                    regressors = regressors.iloc[n_scans:, :]
                else:
                    regressors = regressors


                fig = plot_regressor_correlations(timeseries_data,
                                                  regressors, roi_cmap)
                regressor_fig = os.path.join(figure_dir,
                                         '{}_regressor_correlation_plot.png'.format(func_img_name))
                fig.savefig(regressor_fig, bbox_inches='tight')
            else:
                regressor_fig = None
        else:
            connectome_fig = None
            regressor_fig = None

        # generate report
        make_report(func, timeseries_dir, overlay_fig, timeseries_fig,
                    connectome_fig, regressor_fig)
