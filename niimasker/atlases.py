"""Module that maps input strings to nilearn's atlas functions"""

import pandas as pd
from nilearn.datasets import (fetch_atlas_destrieux_2009, fetch_atlas_yeo_2011,
                              fetch_atlas_aal, fetch_atlas_basc_multiscale_2015,
                              fetch_atlas_talairach, fetch_atlas_schaefer_2018)


def get_labelled_atlas(query, data_dir=None, return_labels=True):
    """Parses input query to determine which atlas to fetch and what version
    of the atlas to use (if applicable).

    Parameters
    ----------
    query : str
        Input string in the following format:
        nilearn:{atlas_name}:{atlas_parameters}. The following can be for
        `atlas_name`: 'destrieux', 'yeo', 'aal', 'talairach', and 'schaefer'.
        `atlas_parameters` is not available for the `destrieux` atlas.
    data_dir : str, optional
        Directory in which to save atlas data. By default None, which creates
        a ~/nilearn_data/ directory as per nilearn.
    return_labels : bool, optional
        Whether to return atlas labels. Default is True. Not available for the
        'basc' atlas.

    Returns
    -------
    str, list or None
        The atlas image and the accompanying labels (if provided)

    Raises
    ------
    ValueError
        Raised when the query does is not formatted correctly or if the no
        match found.
    """

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
            labels = pd.read_csv(atlas['colors_17'], sep=r'\s+')['NONE'].tolist()
    elif atlas_name == 'aal':
        version = 'SPM12' if sub_param is None else sub_param
        atlas = fetch_atlas_aal(version=version, data_dir=data_dir)
        img = atlas['maps']
        labels = atlas['labels']
    elif atlas_name == 'basc':

        version, scale = sub_param.split('-')
        atlas = fetch_atlas_basc_multiscale_2015(version=version,
                                                 data_dir=data_dir)
        img = atlas['scale{}'.format(scale.zfill(3))]
        labels = None
    elif atlas_name == 'talairach':
        atlas = fetch_atlas_talairach(level_name=sub_param, data_dir=data_dir)
        img = atlas['maps']
        labels = atlas['labels']
    elif atlas_name == 'schaefer':
        n_rois, networks, resolution = sub_param.split('-')
        # corrected version of schaefer labels until fixed in nilearn
        correct_url = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/'
                       'v0.14.3-Update_Yeo2011_Schaefer2018_labelname/'
                       'stable_projects/brain_parcellation/'
                       'Schaefer2018_LocalGlobal/Parcellations/MNI/'
                      )
        atlas = fetch_atlas_schaefer_2018(n_rois=int(n_rois),
                                          yeo_networks=int(networks),
                                          resolution_mm=int(resolution),
                                          data_dir=data_dir,
                                          base_url=correct_url)
        img = atlas['maps']
        labels = atlas['labels']
    else:
        raise ValueError('No atlas detected. Check query string')

    if not return_labels:
        labels = None
    else:
        labels = labels.astype(str).tolist()

    return img, labels
