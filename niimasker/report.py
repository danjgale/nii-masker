import os
import json
import matplotlib
from jinja2 import Template, Environment, FileSystemLoader

from niimasker.plots import (plot_overlay, plot_timeseries, plot_connectome,
                             plot_regressor_corr)

pjoin = os.path.join

def generate_report(func_image, output_dir):

    # initialize directories
    report_dir = pjoin(output_dir, 'reports')
    fig_dir = pjoin(report_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    func_img_name = os.path.basename(func_image.fname).split('.')[0]
    fig_fname_base = pjoin(fig_dir, func_img_name)

    n_rois = func_image.data.shape[1]
    if func_image.voxelwise:
        roi_cmap = matplotlib.cm.get_cmap('gist_yarg')
    else:
        roi_cmap = matplotlib.cm.get_cmap('nipy_spectral')


    overlay_fig = plot_overlay(func_image.mask_img, func_image.img,
                               fig_fname_base, cmap=roi_cmap)
    ts_fig = plot_timeseries(func_image.data, func_image.voxelwise,
                             fig_fname_base, cmap=roi_cmap)
    if not func_image.voxelwise:
        connectome_fig = plot_connectome(func_image.data, fig_fname_base,
                                             tick_cmap=roi_cmap)
        if func_image.regressors is not None:
            reg_corr_fig = plot_regressor_corr(func_image.data,
                                               func_image.regressors,
                                               fig_fname_base, cmap=roi_cmap)
    else:
        connectome_fig = None
        reg_corr_fig = None

    param_file = os.path.join(output_dir, 'niimasker_data/parameters.json')
    with open(param_file, 'r') as f:
        parameters = json.load(f)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_loader = FileSystemLoader(os.path.join(dir_path, 'templates'))
    env = Environment(loader=file_loader)
    template = env.get_template('base.html')
    output = template.render(title=func_img_name,
                             parameters=parameters,
                             func_img=func_image.fname,
                             regressor_file=func_image.regressor_file,
                             overlay_fig=overlay_fig,
                             timeseries_fig=ts_fig,
                             connectome_fig=connectome_fig,
                             regressor_fig=reg_corr_fig
                             )

    save_file = os.path.join(report_dir, '{}_report.html'.format(func_img_name))
    with open(save_file, "w") as f:
        f.write(output)











# def generate_report(func_img, timeseries_dir, overlay_fig, timeseries_fig,
#                     connectome_fig=None, regressor_fig=None):

#     dir_path = os.path.dirname(os.path.realpath(__file__))

#     func_img_name = os.path.basename(func_img).split('.')[0]
#     if connectome_fig is not None:
#         connectome_fig = os.path.abspath(connectome_fig)
#     if regressor_fig is not None:
#         regressor_fig = os.path.abspath(regressor_fig)

#     param_file = os.path.join(timeseries_dir, 'niimasker_data/parameters.json')
#     with open(param_file, 'r') as f:
#         parameters = json.load(f)

#     file_loader = FileSystemLoader(os.path.join(dir_path, 'templates'))
#     env = Environment(loader=file_loader)
#     template = env.get_template('base.html')
#     output = template.render(title=func_img_name,
#                              parameters=parameters,
#                              func_img=os.path.abspath(func_img),
#                              overlay_fig=os.path.abspath(overlay_fig),
#                              timeseries_fig=os.path.abspath(timeseries_fig),
#                              connectome_fig=connectome_fig,
#                              regressor_fig=regressor_fig
#                              )

#     save_file = os.path.join(timeseries_dir,
#                              'reports/{}_report.html'.format(func_img_name))
#     with open(save_file, "w") as f:
#         f.write(output)

