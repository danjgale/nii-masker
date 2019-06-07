"""Module that maps input strings to nilearn's atlas functions"""

import pandas as pd
from nilearn.datasets import (fetch_atlas_destrieux_2009, fetch_atlas_yeo_2011,
                              fetch_atlas_aal, fetch_atlas_basc_multiscale_2015,
                              fetch_atlas_talairach, fetch_atlas_schaefer_2018)


def _get_atlas_function(query, data_dir=None):
    """Fetch atlas based on atlas input string"""

    # extract parameters
    params = query.split(':')
    if len(params) == 3:
        _, atlas_name, sub_param = params
    elif len(params) == 2:
        _, atlas_name = params
        sub_param = None
    else:
        raise ValueError('Incorrect atlas query string provided')

    # get atlas
    if atlas_name == 'destrieux':
        atlas = fetch_atlas_destrieux_2009(lateralized=True, data_dir=data_dir)
        img = atlas['maps']
        labels = atlas['labels']
    elif atlas_name == 'yeo':
        atlas = fetch_atlas_yeo_2011(data_dir=data_dir)
        img = atlas[sub_param]
        if '17' in sub_param:
            labels = pd.read_csv(atlas['colors_17'], sep='\s+')['NONE'].tolist()
    elif atlas_name == 'aal':
        version = 'SPM12' if sub_param is None else sub_param
        atlas = fetch_atlas_aal(version=version, data_dir=data_dir)
        img = atlas['maps']
        labels = atlas['labels']
    elif atlas_name == 'talairach':
        atlas = fetch_atlas_talairach(level_name=sub_param, data_dir=data_dir)
        img = atlas['maps']
        labels = atlas['labels']
    elif atlas_name == 'schaefer':
        n_rois, networks, resolution = sub_param.split('-')
        atlas = fetch_atlas_schaefer_2018(n_rois=int(n_rois),
                                          yeo_networks=int(networks),
                                          resolution_mm=int(resolution),
                                          data_dir=data_dir)
        img = atlas['maps']
        labels = atlas['labels']
    else:
        raise ValueError('No atlas detected. Check query string')

    return img, labels