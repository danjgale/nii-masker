import os
import json
from jinja2 import Template, Environment, FileSystemLoader

def make_report(func_img, timeseries_dir, overlay_fig, timeseries_fig,
                connectome_fig=None, qcfc_fig=None):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    func_img_name = os.path.basename(func_img).split('.')[0]

    param_file = os.path.join(timeseries_dir, 'niimasker_data/parameters.json')
    with open(param_file, 'r') as f:
        parameters = json.load(f)

    file_loader = FileSystemLoader(os.path.join(dir_path, 'templates'))
    env = Environment(loader=file_loader)
    template = env.get_template('base.html')
    output = template.render(title=func_img_name,
                             parameters=parameters,
                             func_img=os.path.abspath(func_img),
                             overlay_fig=os.path.abspath(overlay_fig),
                             timeseries_fig=os.path.abspath(timeseries_fig),
                             connectome_fig=os.path.abspath(connectome_fig)
                             )

    save_file = os.path.join(timeseries_dir,
                             'reports/{}_report.html'.format(func_img_name))
    with open(save_file, "w") as f:
        f.write(output)

