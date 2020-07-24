import bids
import dash
import dash_core_components as dcc
import dash_html_components as html

import flask
import glob
import os


def build_app(layout):

    subjects = sorted([p.split('-')[1][:-1] for p in glob.glob('sub-*/')])

    static_image_route = '/images/'
    image_directory = os.getcwd()
    preproc_steps = [
        ('Susceptibility distortion correction', 'sdc'),
        ('Alignment of functional and anatomical MRI data', 'bbregister'),
        ('Brain mask and (temporal/anatomical) CompCor ROIs', 'rois'),
        ('BOLD Summary', 'carpetplot'),
        ('Correlations among nuisance regressors', 'confoundcorr'),
    ]

    app = dash.Dash()

    app.layout = html.Div([
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{'label': f"sub-{subject}", 'value': subject} for subject in subjects],
            value=subjects[0]
        ),
        dcc.Dropdown(
            id='run-dropdown'
        ),
        dcc.Tabs(
            id='step-tabs',
            children = [dcc.Tab(label=step_name, value=step) for step_name, step in preproc_steps],
            value = preproc_steps[0][1]
        ),
        html.ObjectEl(id='image')
    ])



    @app.callback(
        dash.dependencies.Output('run-dropdown', 'options'),
        [dash.dependencies.Input('subject-dropdown', 'value')])
    def update_subject(subject):
        paths = sorted([os.path.basename(p) for p in glob.glob(f"sub-{subject}/figures/*desc-sdc_bold.svg")])
        runs = ['_'.join([ent for ent in p.split('_') if ent.split('-')[0] in ['session', 'task', 'run']]) for p in paths]
        return [{'label': run, 'value': path} for run, path in zip(runs, paths)]

    @app.callback(
        dash.dependencies.Output('image', 'data'),
        [dash.dependencies.Input('subject-dropdown', 'value'),
         dash.dependencies.Input('run-dropdown', 'value'),
         dash.dependencies.Input('step-tabs', 'value')
        ])
    def update_image_src(subject, fname, step):
        if fname:
            return os.path.join(static_image_route, subject, fname.replace('-sdc_','-%s_'%step))


    @app.server.route('/images/<subject>/<image_path>')
    def serve_image(subject, image_path):
        image_directory = os.path.join(os.getcwd(), 'sub-%s'%subject, 'figures')
        if not os.path.exists(os.path.join(image_directory, image_path)):
            raise RuntimeError('image_path not found')
        return flask.send_from_directory(image_directory, image_path)


    return app

if __name__ == '__main__':
    app = build_app(None)
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
