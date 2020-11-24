import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html

import flask
import glob
import os


def build_app(derivatives_path):

    static_image_route = "/images/"
    image_directory = os.getcwd()
    preproc_steps = [
        ("Susceptibility distortion correction", "sdc"),
        ("Alignment of functional and anatomical MRI data", "bbregister"),
        ("Brain mask and (temporal/anatomical) CompCor ROIs", "rois"),
        ("BOLD Summary", "carpetplot"),
        ("Correlations among nuisance regressors", "confoundcorr"),
    ]

    def list_runs(subject):
        paths = sorted(
            [
                os.path.basename(p)
                for p in glob.glob(
                    f"{derivatives_path}/sub-{subject}/figures/*desc-sdc_bold.svg"
                )
            ]
        )
        print(paths)
        runs = [
            "_".join(
                [
                    ent
                    for ent in p.split("_")
                    if ent.split("-")[0] in ["ses", "task", "run"]
                ]
            )
            for p in paths
        ]
        return runs, paths

    subjects = sorted(
        [
            os.path.basename(p[:-1]).split("-")[1]
            for p in glob.glob(f"{derivatives_path}/sub-*/")
        ]
    )
    default_runs = [
        {"label": run, "value": path} for run, path in zip(*list_runs(subjects[0]))
    ]

    app = dash.Dash()

    app.layout = html.Div(
        [
            dcc.Dropdown(
                id="subject-dropdown",
                options=[
                    {"label": f"sub-{subject}", "value": subject}
                    for subject in subjects
                ],
                value=subjects[0],
            ),
            dcc.Dropdown(
                id="run-dropdown", options=default_runs, value=default_runs[0]["value"]
            ),
            dcc.Tabs(
                id="step-tabs",
                children=[
                    dcc.Tab(label=step_name, value=step)
                    for step_name, step in preproc_steps
                ],
                value=preproc_steps[0][1],
            ),
            html.ObjectEl(id="image", width="100%"),
        ]
    )

    @app.callback(
        dash.dependencies.Output("run-dropdown", "options"),
        [dash.dependencies.Input("subject-dropdown", "value")],
    )
    def update_runs_list(subject):
        return [{"label": run, "value": path} for run, path in zip(*list_runs(subject))]

    @app.callback(
        dash.dependencies.Output("run-dropdown", "value"),
        [dash.dependencies.Input("subject-dropdown", "value")],
    )
    def update_runs_value(subject):
        return list_runs(subject)[1][0]

    @app.callback(
        dash.dependencies.Output("image", "data"),
        [
            dash.dependencies.Input("subject-dropdown", "value"),
            dash.dependencies.Input("run-dropdown", "value"),
            dash.dependencies.Input("step-tabs", "value"),
        ],
    )
    def update_image_src(subject, fname, step):
        if fname:
            return os.path.join(
                static_image_route, subject, fname.replace("-sdc_", "-%s_" % step)
            )

    @app.server.route("/images/<subject>/<image_path>")
    def serve_image(subject, image_path):
        print(subject, image_path)
        image_directory = os.path.abspath(
            os.path.join(derivatives_path, "sub-%s" % subject, "figures")
        )
        if not os.path.exists(os.path.join(image_directory, image_path)):
            raise RuntimeError("image_path not found")
        return flask.send_from_directory(image_directory, image_path)

    return app


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="run dash app to view fmriprep qc images",
    )
    parser.add_argument("derivatives_path", help="fmriprep derivative folder")
    parser.add_argument("--port", action="store", default=8050, help="server port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(args.derivatives_path)
    app.run_server(debug=True, dev_tools_silence_routes_logging=False, port=args.port)
