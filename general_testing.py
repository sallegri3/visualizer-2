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

unpickled_df = pd.read_pickle('test_formatted_data')

graph = Generate_Graph([unpickled_df], {'C0002395': 'AD', 'C0020676': 'Hypothyroidism', 'C0025519': 'Metabolism'})

# graph = Generate_Graph()

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])

class UI_Tracker:
    def __init__(self):
        self.set_initial_state()

    def set_initial_state(self):
        self.card_stack_tracking = []
        self.dropdown_cards = []

        self.settings_button_color = '#3A3F44'
        self.settings_button_color_selected = '#00465d'

        self.content_button_color = '#3A3F44'
        self.content_button_color_selected = '#0fa3b1'

        self.settings_button_toggle = False
        self.settings_button_clicks = 0
        self.settings_button_style = {'width': '10vw'}
        self.settings_button_text = 'Expand Settings'

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
            html.H6("Node HeteSim:", className="card-text", style={'marginBottom': '3%'}),
            html.Div(
                dcc.RangeSlider(
                    id='node_hetesim_range_slider',
                    min=graph.node_hetesim_range_start,
                    max=graph.node_hetesim_range_end,
                    step=graph.node_hetesim_step_size,
                    value=[
                        graph.node_hetesim_range[0], 
                        graph.node_hetesim_range[1]
                        ],
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6("Edge HeteSim:", className="card-text", style={'marginBottom': '3%'}),

            html.Div(
                dcc.RangeSlider(
                    id='edge_hetesim_range_slider',
                    min=graph.edge_hetesim_range_start,
                    max=graph.edge_hetesim_range_end,
                    step=graph.edge_hetesim_step_size,
                    value=[
                        graph.edge_hetesim_range[0],
                        graph.edge_hetesim_range[1]
                        ],
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ), 
                style={'marginBottom': '6%'} 
            ),

            html.H6("Max Node Count:", className="card-text", style={'marginBottom': '3%'}),

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
                style={'marginBottom': '6%'} 
            )
        ]
    ),
    className="mt-3",
)

node_filtering = dbc.Card(
    dbc.CardBody(
        [
            html.Div(children=[
                html.H6('Select Target Node CUIs:', className="card-text"),
                dcc.Dropdown(
                    id='specific_target_dropdown', 
                    options=graph.specific_target_input_options,
                    value=None,
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Select Source Node CUIs:', className="card-text"),
                dcc.Dropdown(
                    id='specific_source_dropdown', 
                    options=graph.specific_source_input_options,
                    value=None,
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Select Source Node Types:', className="card-text"),
                dcc.Dropdown(
                    id='specific_type_dropdown', 
                    options=graph.specific_type_input_options,
                    value=None,
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            )          
        ]
    ),
    className="mt-3",
)

graph_spread = dbc.Card(
    dbc.CardBody(
        [
            html.H6("Target Node Spread:", className="card-text", style={'marginBottom': '3%'}),
            html.Div(
                dcc.Slider(
                    id='target_spread_slider',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6("Source Node Spread:", className="card-text", style={'marginBottom': '3%'}),
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
                html.H6('Type Gradient:', style={'marginRight': '5%'}, className="card-text"
                ),
                dbc.Input(
                    type="color",
                    id="gradient_start",
                    value="#000000",
                    style={"width": 50, "height": 25, 'display': 'inline-block', 'marginRight': '1%', 'border': 'none', 'padding': '0'}
                ),
                html.Div(',', style={'display': 'inline-block', 'marginRight': '1%', 'marginLeft': '1%'}),
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
                html.H6('Selected Source Node Type Color:', style={'marginRight': '5%'}, className="card-text"),
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
                html.H6('Target Node Color:', style={'marginRight': '5%'}, className="card-text"),
                dbc.Input(
                    type="color",
                    id="target_color",
                    value="#000000",
                    style={"width": 50, "height": 25, 'display': 'inline-block', 'border': 'none', 'padding': '0'}
                )           
                ], 
                style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
            ), 

            html.Div(children=[
                dbc.Button(
                    'Randomize Colors', 
                    id='randomize_colors_button', 
                    className="btn shadow-none"
                )
                ],
                style={'marginBottom': '4%'}, 
                className="d-grid gap-2 col-6 mx-auto"
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
                style={'max-height': '500px', 'overflowY': 'scroll'}
            )
        ]
    ),
    className="mt-3",
)

table_data = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                dt.DataTable(
                    id='data_table', 
                    columns=graph.data_table_columns, 
                    data=graph.table_data, 
                    style_as_list_view=True,
                    style_cell={
                        'backgroundColor': '#32383E', 
                        'textAlign': 'left'
                    },
                    style_header={
                        'border': '#32383E'
                    },
                    style_table={'overflowX': 'auto', 'overflowY': 'auto', 'max-height': '500px'},
                    style_data_conditional=[                
                        {
                            "if": {"state": "selected"},
                            "backgroundColor": "#8ecae6",
                            "border": '#FFFFFF',
                            'color': '#000000'
                        }
                    ],
                    css=[
                        { 'selector': '.current-page', 'rule': 'visibility: hidden;'}, 
                        { 'selector': '.current-page-shadow', 'rule': 'color: #AAAAAA; font-size: 16px;'}
                        # { 'selector': '.current-page-shadow', 'rule': 'font-size: 16px;'}
                    ],
                    page_size=50
                ), 
                style={'max-height': '500px'}
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
                dbc.Button(children=[
                    'Expand Settings'
                    ],
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
    Output('settings_button', 'style'),
    Output('settings_button', 'children'),
    Input("settings_button", "n_clicks")
)
def toggle_settings(settings_button_clicks):
    if settings_button_clicks != None:
        if settings_button_clicks > ui_tracker.settings_button_clicks:
            ui_tracker.settings_button_clicks = ui_tracker.settings_button_clicks + 1
            ui_tracker.settings_button_toggle = not ui_tracker.settings_button_toggle

            if ui_tracker.settings_button_toggle:
                ui_tracker.settings_button_style['background'] = ui_tracker.settings_button_color_selected
                ui_tracker.settings_button_text = 'Collapse Settings'

            else:
                ui_tracker.settings_button_style['background'] = ui_tracker.settings_button_color
                ui_tracker.settings_button_text = 'Expand Settings'
    
    return [ui_tracker.settings_button_toggle, ui_tracker.settings_button_style, ui_tracker.settings_button_text]

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
                ui_tracker.graph_sliders_button_style['background'] = ui_tracker.content_button_color_selected

            else:
                ui_tracker.graph_sliders_button_style['background'] = ui_tracker.content_button_color
    
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
                ui_tracker.node_filtering_button_style['background'] = ui_tracker.content_button_color_selected

            else:
                ui_tracker.node_filtering_button_style['background'] = ui_tracker.content_button_color

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
                ui_tracker.graph_spread_button_style['background'] = ui_tracker.content_button_color_selected

            else:
                ui_tracker.graph_spread_button_style['background'] = ui_tracker.content_button_color

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
                ui_tracker.color_editing_button_style['background'] = ui_tracker.content_button_color_selected

            else:
                ui_tracker.color_editing_button_style['background'] = ui_tracker.content_button_color

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
                ui_tracker.node_data_button_style['background'] = ui_tracker.content_button_color_selected

            else:
                ui_tracker.node_data_button_style['background'] = ui_tracker.content_button_color

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
                ui_tracker.table_data_button_style['background'] = ui_tracker.content_button_color_selected

            else:
                ui_tracker.table_data_button_style['background'] = ui_tracker.content_button_color

    return [ui_tracker.table_data_button_toggle, ui_tracker.table_data_button_style]

@app.callback(
    Output(component_id='output_graph', component_property='elements'),
    Output(component_id='node_hetesim_range_slider', component_property='value'),
    Output(component_id='edge_hetesim_range_slider', component_property='value'),
    Output(component_id='max_node_slider', component_property='value'),
    Output(component_id='specific_target_dropdown', component_property='value'),
    Output(component_id='specific_source_dropdown', component_property='value'),
    Output(component_id='specific_type_dropdown', component_property='value'),
    Output(component_id='target_spread_slider', component_property='value'),
    Output(component_id='source_spread_slider', component_property='value'),
    Output(component_id='gradient_start', component_property='value'), 
    Output(component_id='gradient_end', component_property='value'), 
    Output(component_id='target_color', component_property='value'),
    Output(component_id='data_table', component_property='columns'),
    Output(component_id='data_table', component_property='data'),
    [Input(component_id='node_hetesim_range_slider', component_property='value')],
    [Input(component_id='edge_hetesim_range_slider', component_property='value')],
    Input(component_id='max_node_slider', component_property='value'),
    Input(component_id='specific_target_dropdown', component_property='value'), 
    Input(component_id='specific_source_dropdown', component_property='value'), 
    Input(component_id='specific_type_dropdown', component_property='value'), 
    Input(component_id='target_spread_slider', component_property='value'), 
    Input(component_id='source_spread_slider', component_property='value'), 
    Input(component_id='gradient_start', component_property='value'), 
    Input(component_id='gradient_end', component_property='value'), 
    Input(component_id='sn_type_color', component_property='value'), 
    Input(component_id='target_color', component_property='value'), 
    Input(component_id='randomize_colors_button', component_property='n_clicks')
)
def toggle_left(
    input_node_hetesim_range_slider, 
    input_edge_hetesim_range_slider, 
    input_max_node_slider, 
    input_specific_target_dropdown, 
    input_specific_source_dropdown, 
    input_specific_type_dropdown, 
    input_target_spread,
    input_sn_spread, 
    input_gradient_start, 
    input_gradient_end, 
    input_sn_type_color,
    input_target_color, 
    input_randomize_colors_button_clicks):

    if input_node_hetesim_range_slider != graph.node_hetesim_range:
        graph.node_hetesim_range = input_node_hetesim_range_slider
        graph.node_hetesim_range_adjusted = True

    if input_edge_hetesim_range_slider != graph.edge_hetesim_range:
        graph.edge_hetesim_range = input_edge_hetesim_range_slider
        graph.edge_hetesim_range_adjusted = True

    if input_max_node_slider != graph.max_node_count:
        graph.max_node_count = input_max_node_slider
        graph.max_node_count_adjusted = True

    if input_specific_target_dropdown != graph.specific_target_dropdown:
        graph.specific_target_dropdown = input_specific_target_dropdown

    if input_specific_source_dropdown != graph.specific_source_dropdown:
        graph.specific_source_dropdown = input_specific_source_dropdown

    if input_specific_type_dropdown != graph.specific_type_dropdown:
        graph.specific_type_dropdown = input_specific_type_dropdown

    graph.target_spread = input_target_spread
    graph.sn_spread = input_sn_spread

    if input_gradient_start != graph.gradient_start:
        graph.gradient_start = input_gradient_start    

    if input_gradient_end != graph.gradient_end:
        graph.gradient_end = input_gradient_end

    if input_sn_type_color != graph.selected_type_color:
        graph.selected_type_color = input_sn_type_color

    if input_target_color != graph.target_color:
        graph.target_color = input_target_color

    if input_randomize_colors_button_clicks != graph.randomized_color_button_clicks:
        graph.randomized_color_button_clicks = input_randomize_colors_button_clicks
        graph.random_color = True

    graph._generate_color_mapping()
    graph.random_color = False

    return [
        graph.update_graph_elements(),
        graph.node_hetesim_range, 
        graph.edge_hetesim_range, 
        graph.max_node_count,
        graph.specific_target_dropdown, 
        graph.specific_source_dropdown, 
        graph.specific_type_dropdown, 
        graph.target_spread, 
        graph.sn_spread,
        input_gradient_start, 
        input_gradient_end, 
        input_target_color, 
        graph.data_table_columns, 
        graph.table_data]


@app.callback(
    Output('node_data', 'children'),
    Input('output_graph', 'selectedNodeData'))
def displayTapNodeData(input_selected_nodes):

    if (input_selected_nodes == []) or (input_selected_nodes == None):
        return 'Select Node(s)'

    display_data = []
    selected_set = set()

    for i, node in enumerate(input_selected_nodes):
        if node['sn_or_tn'] == 'source_node':
            selected_set.add(graph.starting_nx_graph.nodes[node['id']]['type'])
            display_data.append(html.Div( 
                children=[graph.starting_nx_graph.nodes[node['id']]['name']], 
                style={'font-weight': 'bold'}))

            edges = {}
            for j, connecting_node in enumerate(graph.starting_nx_graph[node['id']]):
                edges[str(graph.target_cui_target_name_dict[connecting_node]) + ' (CUI:' + str(connecting_node) + ')'] = np.round(graph.starting_nx_graph[node['id']][connecting_node]['hetesim'], 3)

            edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

            data_dump = {
                'node_cui': graph.starting_nx_graph.nodes[node['id']]['cui'], 
                'node_name': graph.starting_nx_graph.nodes[node['id']]['name'], 
                'node_type': graph.starting_nx_graph.nodes[node['id']]['type'], 
                'mean_hetesim': np.round(graph.starting_nx_graph.nodes[node['id']]['mean_hetesim'], 3), 
                'sn_or_tn': 'source_node', 
                'edge_hetesim': edges_sorted
                }

            display_data.append(html.Pre(json.dumps(data_dump, indent=2)))

        if node['sn_or_tn'] == 'target_node':
            display_data.append(html.Div( 
                children=[graph.starting_nx_graph.nodes[node['id']]['name']], 
                style={'font-weight': 'bold'}))

            edges = {}
            for j, connecting_node in enumerate(graph.starting_nx_graph[node['id']]):
                edges[str(graph.starting_nx_graph.nodes[connecting_node]['name']) + ' (CUI:' + str(graph.starting_nx_graph.nodes[connecting_node]['cui']) + ')'] = float(np.round(graph.starting_nx_graph[node['id']][connecting_node]['hetesim'], 3))
            
            edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

            data_dump = {
                'node_cui': graph.starting_nx_graph.nodes[node['id']]['id'], 
                'node_name': graph.starting_nx_graph.nodes[node['id']]['name'], 
                'sn_or_tn': 'target_node', 
                'edge_hetesim': edges_sorted
                }

            display_data.append(html.Pre(json.dumps(data_dump, indent=2)))

    return display_data


if __name__ == "__main__":
    app.run_server(debug=True)


# Include button hover .css in assets
# Style scrollbar using .css

# Fix button activation
# Expanding menu