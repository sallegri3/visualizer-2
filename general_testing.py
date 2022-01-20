from pickletools import StackObject
from tkinter.ttk import Style
from click import style
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import time
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_table as dt
import json
from colour import Color
import dash_bootstrap_components as dbc

from generate_graph import Generate_Graph

# unpickled_df = pd.read_pickle('test_formatted_data')

# graph = Generate_Graph([unpickled_df], {'C0002395': 'AD', 'C0020676': 'Hypothyroidism', 'C0025519': 'Metabolism'})

graph = Generate_Graph()

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])

class UI_Tracker:
    def __init__(self):
        self.set_initial_state()

    def set_initial_state(self):
        self.card_stack_tracking = []
        self.dropdown_cards = []

        self.settings_button_color = None
        self.content_button_color = None

        self.settings_button_toggle = False
        self.settings_button_clicks = 0
        self.settings_button_style = {'width': '10vw'}

        self.graph_sliders_button_toggle = False
        self.graph_sliders_button_clicks = 0
        self.graph_sliders_button_style = {'width': '10vw'}

        self.node_filtering_button_toggle = False
        self.node_filtering_button_clicks = 0
        self.node_filtering_button_style = {'width': '10vw'}

        self.graph_spread_button_toggle = False
        self.graph_spread_button_clicks = 0
        self.graph_spread_button_style = {'width': '10vw'}

        self.color_editing_button_toggle = False
        self.color_editing_button_clicks = 0
        self.color_editing_button_style = {'width': '10vw'}

        self.node_data_button_toggle = False
        self.node_data_button_clicks = 0
        self.node_data_button_style = {'width': '10vw'}

        self.table_data_button_toggle = False
        self.table_data_button_clicks = 0
        self.table_data_button_style = {'width': '10vw'}


ui_tracker = UI_Tracker()

graph_sliders = dbc.Card(
    dbc.CardBody(
        [
            html.H6("Node HeteSim:", className="card-text"),

            html.Div(
                dcc.RangeSlider(
                    id='node_hetesim_range_slider',
                    min=0,
                    max=np.round(graph.mean_hetesim_range[1], 3) + graph.mean_hetesim_step_size,
                    step=graph.mean_hetesim_step_size,
                    value=[
                        (np.round(graph.mean_hetesim_range[0], 3) - graph.mean_hetesim_step_size), 
                        (np.round(graph.mean_hetesim_range[1], 3) + graph.mean_hetesim_step_size)
                        ],
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6("Edge HeteSim:", className="card-text"),

            html.Div(
                dcc.RangeSlider(
                    id='edge_hetesim_range_slider',
                    min=0,
                    max=np.round(graph.max_edge_value, 3) + graph.edge_hetesim_step_size,
                    step=graph.edge_hetesim_step_size,
                    value=[
                        (np.round(graph.min_edge_value, 3) - graph.edge_hetesim_step_size),
                        (np.round(graph.max_edge_value, 3) + graph.edge_hetesim_step_size)
                        ],
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ), 
                style={'marginBottom': '6%'} 
            ),

            html.H6("Max Node Count:", className="card-text"),

            html.Div(
                dcc.Slider(
                    id='max_node_slider',
                    min=1,
                    max=graph.max_node_count,
                    step=1,
                    value=graph.max_node_count,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ), 
                style={'marginBottom': '4%'} 
            ),
        ]
    ),
    className="mt-3",
)

node_filtering = dbc.Card(
    dbc.CardBody(
        [
            html.Div(children=[
                html.H6(children=[
                    'Specific TN CUIs:'
                    ], 
                    style={},
                    className="card-text"
                ),
                html.Div(children=[
                    dcc.Dropdown(
                        id='specific_target_input', 
                        options=[
                            {'label': 'New York City', 'value': 'NYC'},
                            {'label': 'Montreal', 'value': 'MTL'},
                            {'label': 'San Francisco', 'value': 'SF'}
                        ],
                    value=['MTL', 'NYC'])
                    ], 
                    style={}
                )
                ],
                style={'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6(children=[
                    'Specific TN CUIs:'
                    ], 
                    style={},
                    className="card-text"
                ),
                html.Div(children=[
                    dcc.Dropdown(
                        id='specific_source_input', 
                        options=[
                            {'label': 'New York City', 'value': 'NYC'},
                            {'label': 'Montreal', 'value': 'MTL'},
                            {'label': 'San Francisco', 'value': 'SF'}
                        ],
                    value=['MTL', 'NYC'])
                    ], 
                    style={}
                )
                ],
                style={'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6(children=[
                    'Specific TN CUIs:'
                    ], 
                    style={},
                    className="card-text"
                ),
                html.Div(children=[
                    dcc.Dropdown(
                        id='specific_type_input', 
                        options=[
                            {'label': 'New York City', 'value': 'NYC'},
                            {'label': 'Montreal', 'value': 'MTL'},
                            {'label': 'San Francisco', 'value': 'SF'}
                        ],
                    value=['MTL', 'NYC'])
                    ], 
                    style={}
                )
                ],
                style={'marginBottom': '6%'}
            ), 

            html.Div(
                dbc.Button(
                    'Reset Graph', 
                    className="btn shadow-none"
                ),
                style={'marginBottom': '4%'}, 
                className="d-grid gap-2 col-6 mx-auto"
            )
        ]
    ),
    className="mt-3",
)

graph_spread = dbc.Card(
    dbc.CardBody(
        [
            html.H6("Target Node Spread:", className="card-text"),

            html.Div(
                dcc.Slider(
                    id='target_spread_slider',
                    min=0.1,
                    max=3,
                    step=0.01,
                    value=1,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6("Source Node Spread:", className="card-text"),

            html.Div(
                dcc.Slider(
                    id='source_spread_slider',
                    min=0.01,
                    max=0.3,
                    step=0.01,
                    value=0.1,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),
        ]
    ),
    className="mt-3",
)

color_editing = dbc.Card(
    dbc.CardBody(
        [
            html.Div(children=[
                html.Div(children=[
                    html.H6(children=[
                        'Type Gradient:'
                        ], 
                        style={'marginRight': '3%'},
                        className="card-text"
                    ),
                    dbc.Input(
                        type="color",
                        id="gradient_start",
                        value="#000000",
                        style={"width": 50, "height": 25, 'display': 'inline-block', 'marginRight': '1%', 'border': 'none', 'padding': '0'}
                    ),
                    html.Div(',', style={'display': 'inline-block'}),
                    dbc.Input(
                        type="color",
                        id="gradient_end",
                        value="#000000",
                        style={"width": 50, "height": 25, 'display': 'inline-block', 'marginLeft': '1%', 'border': 'none', 'padding': '0'},
                    )               
                    ], 
                    style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
                ), 

                html.Div(children=[
                    html.H6(children=[
                        'Selected Source Node Type Color:'
                        ], 
                        style={'marginRight': '3%'},
                        className="card-text"
                    ),
                    dbc.Input(
                        type="color",
                        id="sn_type_color",
                        value="#000000",
                        style={"width": 50, "height": 25, 'display': 'inline-block', 'border': 'none', 'padding': '0'}
                    )           
                    ], 
                    style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
                ), 

                html.Div(children=[
                    html.H6(children=[
                        'Target Node Color:'
                        ], 
                        style={'marginRight': '3%'},
                        className="card-text"
                    ),
                    dbc.Input(
                        type="color",
                        id="target_color",
                        value="#000000",
                        style={"width": 50, "height": 25, 'display': 'inline-block', 'border': 'none', 'padding': '0'}
                    )           
                    ], 
                    style={'display': 'flex', 'justiftyContent': 'center'}
                ), 

                html.Div(children=[
                    dbc.Button(
                        'Randomize Colors', 
                        className="btn shadow-none"
                    )
                    ],
                    style={'marginTop': '5%', 'marginBottom': '4%'}, 
                    className="d-grid gap-2 col-6 mx-auto"
                )

                ],
                style={}
            )
        ]
    ),
    className="mt-3",
)

node_data = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                html.Div(
                    children='Select Node(s)',
                    id='node_data'
                ), 
                style={'min-height': '50', 'max-height': '500'}     
            )
        ]
    ),
    className="mt-3",
)

table_data = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                html.Div(
                    children=None,
                    id='table_data'
                ), 
                style={} # Include overflow in table
            )
        ]
    ),
    className="mt-3",
)

default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'width': 'data(size)',
            'height': 'data(size)',
            'content': 'data(label)',
            'font-size': '1px',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': 'data(color)'
            }
        },
    {
        'selector': 'edge',
        'style': {
            'width': 'data(size)', 
            'line-opacity': 'data(size)', 
            'label': 'data(label)', 
            'font-size': 1, 
            'font-opacity': 1
            }
        }, 
    {
        'selector': ':selected',
        'style': {
            'border-color': 'black', 
            'border-opacity': '1', 
            'border-width': '0.3px'
            }
        }
    ]

cytoscape_graph = cyto.Cytoscape(
    id='output_graph',
    layout={'name': 'preset'},
    style={'width': '100vw', 'height': '100vh'},
    stylesheet=default_stylesheet,
    elements=graph.elements,
    boxSelectionEnabled=True
    )

def server_layout():
    ui_tracker.set_initial_state()

    app_layout = html.Div([
        html.Div(cytoscape_graph, style={'position': 'fixed', 'zIndex': '1', 'width': '99vw', 'height': '99vh'}),
        html.Div(
            html.Div(children=[
                dbc.Button(
                    'Expand Settings',
                    id='settings_button',
                    className="btn shadow-none",
                    style={'width': '10vw'}
                ), 
                dbc.Collapse(children=[
                    dbc.Button(
                        'Graph Sliders',
                        id='graph_sliders_button',
                        className="btn shadow-none",
                        style=ui_tracker.graph_sliders_button_style
                    ), 
                    dbc.Button(
                        'Node Filtering',
                        id='node_filtering_button',
                        className="btn shadow-none",
                        style=ui_tracker.node_filtering_button_style
                    ), 
                    dbc.Button(
                        'Graph Spread',
                        id='graph_spread_button',
                        className="btn shadow-none",
                        style={'width': '10vw'}
                    ), 
                    dbc.Button(
                        'Color Editing',
                        id='color_editing_button',
                        className="btn shadow-none",
                        style={'width': '10vw'}
                    ), 
                    dbc.Button(
                        'Node Data',
                        id='node_data_button',
                        className="btn shadow-none",
                        style={'width': '10vw'}
                    ), 
                    dbc.Button(
                        'Table Data',
                        id='table_data_button',
                        className="btn shadow-none",
                        style={'width': '10vw'}
                    )
                    ],
                    id="settings_collapse",
                    is_open=False
                ), 
                ], 
                style={'display': 'flex', 'justiftyContent': 'center'}
            ),
            id='settings_div',
            style={'marginLeft': '1vw', 'marginTop': '1vh', 'width': 'fit-content', 'position': 'relative', 'zIndex': '22'}
        ),
        html.Div(children=[
            dbc.Collapse(
                graph_sliders,
                id="graph_sliders_collapse",
                is_open=False
            ), 
            dbc.Collapse(
                node_filtering,
                id="node_filtering_collapse",
                is_open=False
            ), 
            dbc.Collapse(
                graph_spread,
                id="graph_spread_collapse",
                is_open=False
            ), 
            dbc.Collapse(
                color_editing,
                id="color_editing_collapse",
                is_open=False
            ), 
            dbc.Collapse(
                node_data,
                id="node_data_collapse",
                is_open=False
            ), 
            dbc.Collapse(
                table_data,
                id="table_data_collapse",
                is_open=False
            )
            ],
            id='dropdown_card_div',
            style={'marginLeft': '1vw', 'marginTop': '1vh', 'display': 'flex', 'flex-direction': 'column', 'width': '20vw', 'position': 'relative', 'zIndex': '22'}
        )
    ])

    return app_layout

app.layout = server_layout

@app.callback(
    Output("settings_collapse", "is_open"),
    Output("settings_button", "style"),
    Input("settings_button", "n_clicks")
)
def toggle_settings(setting_button_clicks):
    if setting_button_clicks != None:
        if setting_button_clicks > ui_tracker.settings_button_clicks:
            ui_tracker.settings_button_clicks = ui_tracker.settings_button_clicks + 1
            ui_tracker.settings_button_toggle = not ui_tracker.settings_button_toggle

            if ui_tracker.settings_button_toggle:
                ui_tracker.settings_button_style['background'] = 'green'

            else:
                ui_tracker.settings_button_style['background'] = '#3A3F44'
    
    return [ui_tracker.settings_button_toggle, ui_tracker.settings_button_style]

@app.callback(
    Output("graph_sliders_collapse", "is_open"),
    Output("graph_sliders_button", "style"),
    Input("graph_sliders_button", "n_clicks")
)
def toggle_left(graph_sliders_button_clicks):
    if graph_sliders_button_clicks != None:
        if graph_sliders_button_clicks > ui_tracker.graph_sliders_button_clicks:
            ui_tracker.graph_sliders_button_clicks = ui_tracker.graph_sliders_button_clicks + 1
            ui_tracker.graph_sliders_button_toggle = not ui_tracker.graph_sliders_button_toggle

            if ui_tracker.graph_sliders_button_toggle:
                ui_tracker.graph_sliders_button_style['background'] = 'orange'

            else:
                ui_tracker.graph_sliders_button_style['background'] = '#3A3F44'
    
    return [ui_tracker.graph_sliders_button_toggle, ui_tracker.graph_sliders_button_style]

@app.callback(
    Output("node_filtering_collapse", "is_open"),
    Output("node_filtering_button", "style"),
    Input("node_filtering_button", "n_clicks")
)
def toggle_left(node_filtering_button_clicks):
    if node_filtering_button_clicks != None:
        if node_filtering_button_clicks > ui_tracker.node_filtering_button_clicks:
            ui_tracker.node_filtering_button_clicks = ui_tracker.node_filtering_button_clicks + 1
            ui_tracker.node_filtering_button_toggle = not ui_tracker.node_filtering_button_toggle

            if ui_tracker.node_filtering_button_toggle:
                ui_tracker.node_filtering_button_style['background'] = 'orange'

            else:
                ui_tracker.node_filtering_button_style['background'] = '#3A3F44'

    return [ui_tracker.node_filtering_button_toggle, ui_tracker.node_filtering_button_style]

@app.callback(
    Output("graph_spread_collapse", "is_open"),
    Output("graph_spread_button", "style"),
    Input("graph_spread_button", "n_clicks")
)
def toggle_left(graph_spread_button_clicks):
    if graph_spread_button_clicks != None:
        if graph_spread_button_clicks > ui_tracker.graph_spread_button_clicks:
            ui_tracker.graph_spread_button_clicks = ui_tracker.graph_spread_button_clicks + 1
            ui_tracker.graph_spread_button_toggle = not ui_tracker.graph_spread_button_toggle

            if ui_tracker.graph_spread_button_toggle:
                ui_tracker.graph_spread_button_style['background'] = 'orange'

            else:
                ui_tracker.graph_spread_button_style['background'] = '#3A3F44'

    return [ui_tracker.graph_spread_button_toggle, ui_tracker.graph_spread_button_style]

@app.callback(
    Output("color_editing_collapse", "is_open"),
    Output("color_editing_button", "style"),
    Input("color_editing_button", "n_clicks")
)
def toggle_left(color_editing_button_clicks):
    if color_editing_button_clicks != None:
        if color_editing_button_clicks > ui_tracker.color_editing_button_clicks:
            ui_tracker.color_editing_button_clicks = ui_tracker.color_editing_button_clicks + 1
            ui_tracker.color_editing_button_toggle = not ui_tracker.color_editing_button_toggle

            if ui_tracker.color_editing_button_toggle:
                ui_tracker.color_editing_button_style['background'] = 'orange'

            else:
                ui_tracker.color_editing_button_style['background'] = '#3A3F44'

    return [ui_tracker.color_editing_button_toggle, ui_tracker.color_editing_button_style]

@app.callback(
    Output("node_data_collapse", "is_open"),
    Output("node_data_button", "style"),
    Input("node_data_button", "n_clicks")
)
def toggle_left(node_data_button_clicks):
    if node_data_button_clicks != None:
        if node_data_button_clicks > ui_tracker.node_data_button_clicks:
            ui_tracker.node_data_button_clicks = ui_tracker.node_data_button_clicks + 1
            ui_tracker.node_data_button_toggle = not ui_tracker.node_data_button_toggle

            if ui_tracker.node_data_button_toggle:
                ui_tracker.node_data_button_style['background'] = 'orange'

            else:
                ui_tracker.node_data_button_style['background'] = '#3A3F44'

    return [ui_tracker.node_data_button_toggle, ui_tracker.node_data_button_style]

@app.callback(
    Output("table_data_collapse", "is_open"),
    Output("table_data_button", "style"),
    Input("table_data_button", "n_clicks")
)
def toggle_left(table_data_button_clicks):
    if table_data_button_clicks != None:
        if table_data_button_clicks > ui_tracker.table_data_button_clicks:
            ui_tracker.table_data_button_clicks = ui_tracker.table_data_button_clicks + 1
            ui_tracker.table_data_button_toggle = not ui_tracker.table_data_button_toggle

            if ui_tracker.table_data_button_toggle:
                ui_tracker.table_data_button_style['background'] = 'orange'

            else:
                ui_tracker.table_data_button_style['background'] = '#3A3F44'

    return [ui_tracker.table_data_button_toggle, ui_tracker.table_data_button_style]


if __name__ == "__main__":
    app.run_server(debug=True)