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

from generate_graph import Generate_Graph

unpickled_df = pd.read_pickle('test_formatted_data')

graph = Generate_Graph([unpickled_df], {'C0002395': 'AD', 'C0020676': 'Hypothyroidism', 'C0025519': 'Metabolism'})

# graph = Generate_Graph()

app = dash.Dash(__name__)

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

app.layout = html.Div([
    html.Div(children=[
        dcc.Tabs(
            id='tabs_global',
            children=[
                dcc.Tab(
                    label='Graph Data',
                    children=[
                        html.Div(children=[

                            html.Div(children=[
                                html.Div(children=[
                                    'Node HeteSim:'
                                    ], 
                                    style={'width': '30%', 'height': '30px', 'font-size': '0.9vw'}
                                ),
                                html.Div(children=[
                                    dcc.RangeSlider(
                                        id='hetesim_mean_range_slider',
                                        min=0,
                                        max=np.round(graph.mean_hetesim_range[1], 3) + graph.mean_hetesim_step_size,
                                        step=graph.mean_hetesim_step_size,
                                        value=[
                                            (np.round(graph.mean_hetesim_range[0], 3) - graph.mean_hetesim_step_size), 
                                            (np.round(graph.mean_hetesim_range[1], 3) + graph.mean_hetesim_step_size)
                                            ],
                                        tooltip={'placement': 'bottom', 'always_visible': True},
                                        vertical=False
                                        )
                                    ], 
                                    style={'width': '100%', 'height': '30px'}
                                )
                                ], 
                                style={'margin':'5%', 'width': '90%', 'height': '30%'}
                            ), 

                            html.Div(children=[
                                html.Div(children=[
                                    'Edge HeteSim:'
                                    ], style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                                ),
                                html.Div(children=[
                                    dcc.RangeSlider(
                                        id='hetesim_edge_range_slider',
                                        min=0,
                                        max=np.round(graph.max_edge_value, 3) + graph.edge_hetesim_step_size,
                                        step=graph.edge_hetesim_step_size,
                                        value=[
                                            (np.round(graph.min_edge_value, 3) - graph.edge_hetesim_step_size),
                                            (np.round(graph.max_edge_value, 3) + graph.edge_hetesim_step_size)
                                            ],
                                        tooltip={'placement': 'bottom', 'always_visible': True},
                                        vertical=False
                                        )
                                    ], 
                                    style={'width': '70%', 'height': '10px', 'display': 'inline-block'}
                                )
                                ], 
                                style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                            ),

                            html.Div(children=[
                                html.Div(children=[
                                    'Max Nodes:'
                                    ], style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                                ),
                                html.Div(children=[
                                    dcc.Slider(
                                        id='node_max_slider',
                                        min=1,
                                        max=graph.max_node_count,
                                        step=1,
                                        value=graph.max_node_count,
                                        tooltip={'placement': 'bottom', 'always_visible': True},
                                        vertical=False
                                        )
                                    ], 
                                    style={'width': '70%', 'height': '10px', 'display': 'inline-block'}
                                )
                                ], 
                                style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                            ),

                            html.Div(style={'border': '1px black solid', 'margin':'5%', 'width': '90%'}),

                            html.Div(children=[
                                html.Div(children=[
                                    'Specific TN CUIs:'
                                    ], 
                                    style={'width': '35%', 'display': 'inline-block', 'font-size': '0.9vw'}
                                ),
                                html.Div(children=[
                                    dcc.Input(id='specific_target_input', type='text', placeholder='Separate with commas', value=None)
                                    ], 
                                    style={'width': '45%', 'display': 'inline-block'}
                                ),
                                html.Div(children=[
                                    html.Button('Submit', id='specific_target_submit_button', n_clicks=0)
                                    ], 
                                    style={'width': '20%', 'display': 'inline-block'}
                                )
                                ], 
                                style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                            ),

                            html.Div(children=[
                                html.Div(children=[
                                    'Specific SN CUIs:'
                                    ], style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                                ),
                                html.Div(children=[
                                    dcc.Input(id='specific_source_input', type='text', placeholder='Separate with commas')
                                    ], style={'width': '45%', 'height': '10px', 'display': 'inline-block'}
                                ),
                                html.Div(children=[
                                    html.Button('Submit', id='specific_source_submit_button', n_clicks=0)
                                    ], 
                                    style={'width': '20%', 'height': '10px', 'display': 'inline-block'}
                                )
                                ], 
                                style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                            ), 

                            html.Div(children=[
                                html.Div(children=[
                                    'Specific SN Types:'
                                    ], style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                                ),
                                html.Div(children=[
                                    dcc.Input(id='specific_type_input', type='text', placeholder='Separate with commas')
                                    ], style={'width': '45%', 'height': '10px', 'display': 'inline-block'}
                                ),
                                html.Div(children=[
                                    html.Button('Submit', id='specific_type_submit_button', n_clicks=0)
                                    ], 
                                    style={'width': '20%', 'height': '10px', 'display': 'inline-block'}
                                )
                                ], 
                                style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                            ), 

                            html.Div(style={'border': '1px black solid', 'margin':'5%', 'width': '90%'}),

                            html.Div(children=[
                                html.Div(children=[
                                    'Reset Graph:'
                                    ], 
                                    style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                                ),
                                html.Div(children=[
                                    html.Button(
                                        'Submit',
                                        id='graph_reset_submit_button', n_clicks=0)
                                    ], 
                                    style={'width': '20%', 'height': '10px', 'display': 'inline-block'}
                                )
                                ], 
                                style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                            )

                        ], 
                        style={'width': '24vw', 'height': '63.5vh', 'overflowY': 'scroll'}
                    )
                    ],
                    style= {
                        'borderRight': '2px solid black', 
                        'borderBottom': '2px solid black',
                        'font-size': '10px', 
                        'padding': '0.75vw', 
                        'text-align': 'center', 
                    },
                    selected_style={
                        'borderTop': 'None',
                        'borderLeft': 'None', 
                        'font-size': '10px', 
                        'padding': '0.75vw', 
                        'text-align': 'center', 
                    }
                ),

                dcc.Tab(
                    label='Graph Visuals',
                    children=[

                        html.Div(children=[
                            html.Div(children=[
                                'Target Spread:'
                                ], 
                                style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                            ),
                            html.Div(children=[
                                dcc.Slider(
                                    id='target_spread_slider',
                                    min=0.1,
                                    max=3,
                                    step=0.01,
                                    value=1,
                                    tooltip={'placement': 'bottom', 'always_visible': True},
                                    vertical=False
                                    )
                                ], 
                                style={'width': '70%', 'height': '10px', 'display': 'inline-block'})
                            ], 
                            style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                        ),

                        html.Div(children=[
                            html.Div(children=[
                                'SN Spread:'
                                ], 
                                style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                            ),
                            html.Div(children=[
                                dcc.Slider(
                                    id='source_spread_slider',
                                    min=0.01,
                                    max=0.3,
                                    step=0.01,
                                    value=0.1,
                                    tooltip={'placement': 'bottom', 'always_visible': True},
                                    vertical=False
                                    )
                                ], 
                                style={'width': '70%', 'height': '10px', 'display': 'inline-block'})
                            ], 
                            style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                        ), 

                        html.Div(style={'border': '1px black solid', 'margin':'5%', 'width': '90%'}),

                        html.Div(children=[
                            html.Div(children=[
                                'Type Gradient Start:'
                                ], 
                                style={'width': '35%', 'display': 'inline-block', 'font-size': '0.9vw'}
                            ),
                            html.Div(children=[
                                dcc.Input(id='type_gradient_start_input', type='text', placeholder='Specify in RGB, Hex Code, or HSL')
                                ], 
                                style={'width': '45%', 'display': 'inline-block'}
                            ),
                            html.Div(children=[
                                html.Button('Submit', id='type_gradient_start_submit_button', n_clicks=0)
                                ], 
                                style={'width': '20%', 'display': 'inline-block'}
                            )
                            ], 
                            style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                        ),

                        html.Div(children=[
                            html.Div(children=[
                                'Type Gradient End:'
                                ], 
                                style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                            ),
                            html.Div(children=[
                                dcc.Input(id='type_gradient_end_input', type='text', placeholder='Specify in RGB, Hex Code, or HSL')
                                ], 
                                style={'width': '45%', 'height': '10px', 'display': 'inline-block'}
                            ),
                            html.Div(children=[
                                html.Button('Submit', id='type_gradient_end_submit_button', n_clicks=0)
                                ], 
                                style={'width': '20%', 'height': '10px', 'display': 'inline-block'}
                            )
                            ], 
                            style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                        ), 

                        html.Div(children=[
                            html.Div(children=[
                                'Type Color Map:'
                                ], 
                                style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                            ),
                            html.Div(children=[
                                dcc.Input(id='type_color_mapping_input', type='text', placeholder='type_1: color_1, type_2: color_2, etc.')
                                ], 
                                style={'width': '45%', 'height': '10px', 'display': 'inline-block'}
                            ),
                            html.Div(children=[
                                html.Button('Submit', id='type_color_mapping_submit_button', n_clicks=0)
                                ], 
                                style={'width': '20%', 'height': '10px', 'display': 'inline-block'}
                            )
                            ], 
                            style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}), 

                        html.Div(children=[
                            html.Div(children=[
                                'Target Color:'
                                ], 
                                style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                            ),
                            html.Div(children=[
                                dcc.Input(id='target_color_mapping_input', type='text', placeholder='Specify in RGB, Hex Code, or HSL')
                                ], 
                                style={'width': '45%', 'height': '10px', 'display': 'inline-block'}
                            ),
                            html.Div(children=[
                                html.Button('Submit', id='target_color_mapping_submit_button', n_clicks=0)
                                ], 
                                style={'width': '20%', 'height': '10px', 'display': 'inline-block'}
                            )
                            ], 
                            style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}
                        ), 

                        html.Div(children=[
                            html.Div(children=[
                                'Randomize Type Colors:'
                                ], 
                                style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}
                            ),
                            html.Div(children=[
                                html.Button('Submit', id='color_randomize_submit_button', n_clicks=0)
                                ], 
                                style={'width': '20%', 'height': '10px', 'display': 'inline-block'}
                            )
                            ], 
                            style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'})

                    ],
                    style={
                        'borderLeft': '2px solid black', 
                        'borderBottom': '2px solid black',
                        'font-size': '2vh', 
                        'text-align': 'center'
                    },
                    selected_style={
                        'borderTop': 'None',
                        'borderRight': 'None', 
                        'font-size': '2vh', 
                        'text-align': 'center'
                    }
                )
            ], 
            style={'height': '15vh'}
        )
        ], 
        style={'width': '24.5%', 'height': '70vmin', 'border':'2px black solid', 'display': 'inline-block', 'vertical-align': 'bottom'}
    ),

    html.Div(children=[
        cyto.Cytoscape(
            id='output_graph',
            layout={'name': 'preset'},
            style={'width': '100%', 'height': '100%'},
            stylesheet=default_stylesheet,
            elements=graph.elements,
            boxSelectionEnabled=True
        )
        ], 
        style={'width': '50%', 'height': '70vmin', 'border':'2px black solid', 'display': 'inline-block', 'vertical-align': 'bottom'}
    ),
    
    html.Div(children=[
        dcc.Tabs(
            children=[
                dcc.Tab(
                    label='Node Data',
                    children=[
                        html.Div(children=[
                            html.Div(children=[
                                html.Div(
                                    id='node_select_data_1',
                                    children='Select Node(s)', 
                                    style={'width': '99%', 'height': '99%'}
                                )
                                ]
                            )
                            ], 
                            style={'overflowY':'scroll', 'height': '26.8vw'}
                        )
                    ],
                    style= {
                        'borderRight': '2px solid black', 
                        'borderBottom': '2px solid black',
                        'text-align': 'center'
                    },
                    selected_style={
                        'text-align': 'center',
                        'borderTop': 'None',
                        'borderLeft': 'None'
                    }
                ),
                dcc.Tab(
                    label='Table Data',
                    children=[
                        html.Div(
                            id='table_data', 
                            children=[], 
                            style={'overflowX': 'scroll', 'overflowY': 'scroll', 'height': '26.8vw'}
                        )
                    ],
                    style= {
                        'borderLeft': '2px solid black', 
                        'borderBottom': '2px solid black',
                        'text-align': 'center', 
                    },
                    selected_style={
                        'text-align': 'center',
                        'borderTop': 'None',
                        'borderRight': 'None', 
                    }
                )
            ]
        )
        ], 
        style={'width': '24.5%', 'height': '70vmin', 'border':'2px black solid', 'display': 'inline-block', 'vertical-align': 'bottom'}
    )
    ], 
    style={}
)

@app.callback(
    Output(component_id='output_graph', component_property='elements'),
    Output(component_id='table_data', component_property='children'),
    Output(component_id='hetesim_mean_range_slider', component_property='value'),
    Output(component_id='hetesim_edge_range_slider', component_property='value'),
    Output(component_id='node_max_slider', component_property='value'),
    Output(component_id='specific_target_input', component_property='value'),
    Output(component_id='specific_source_input', component_property='value'),
    Output(component_id='specific_type_input', component_property='value'),
    Output(component_id='target_spread_slider', component_property='value'),
    Output(component_id='source_spread_slider', component_property='value'),
    Output(component_id='type_gradient_start_input', component_property='value'),
    Output(component_id='type_gradient_end_input', component_property='value'),
    Output(component_id='type_color_mapping_input', component_property='value'),
    Output(component_id='target_color_mapping_input', component_property='value'),
    [Input(component_id='hetesim_mean_range_slider', component_property='value')],
    [Input(component_id='hetesim_edge_range_slider', component_property='value')],
    Input(component_id='node_max_slider', component_property='value'),
    Input(component_id='target_spread_slider', component_property='value'),
    Input(component_id='source_spread_slider', component_property='value'),
    State(component_id='specific_target_input', component_property='value'),
    Input(component_id='specific_target_submit_button', component_property='n_clicks'), 
    State(component_id='specific_source_input', component_property='value'),
    Input(component_id='specific_source_submit_button', component_property='n_clicks'), 
    State(component_id='specific_type_input', component_property='value'),
    Input(component_id='specific_type_submit_button', component_property='n_clicks'), 
    State(component_id='type_gradient_start_input', component_property='value'),
    Input(component_id='type_gradient_start_submit_button', component_property='n_clicks'), 
    State(component_id='type_gradient_end_input', component_property='value'),
    Input(component_id='type_gradient_end_submit_button', component_property='n_clicks'), 
    State(component_id='type_color_mapping_input', component_property='value'),
    Input(component_id='type_color_mapping_submit_button', component_property='n_clicks'),
    State(component_id='target_color_mapping_input', component_property='value'),
    Input(component_id='target_color_mapping_submit_button', component_property='n_clicks'), 
    Input(component_id='graph_reset_submit_button', component_property='n_clicks'), 
    Input(component_id='color_randomize_submit_button', component_property='n_clicks')
)
def update_graph_hetesim_slider(
    input_mean_hetesim_range_value, 
    input_edge_hetesim_range_value, 
    max_node_slider_value, 
    target_spread_val, 
    sn_spread_val,
    specific_targets_val, 
    specific_targets_submit_count, 
    specific_sources_val, 
    specific_sources_submit_count, 
    specific_types_val, 
    specific_types_submit_count, 
    type_gradient_start_val, 
    type_gradient_start_submit_count, 
    type_gradient_end_val, 
    type_gradient_end_submit_count, 
    type_color_mapping_val, 
    type_color_mapping_submit_count, 
    target_color_mapping_val, 
    target_color_mapping_submit_count, 
    graph_reset_submit_count, 
    color_randomization_submit_count):

    if graph_reset_submit_count > graph.graph_reset_clicks:
        graph.graph_reset_clicks = graph.graph_reset_clicks + 1

        graph.target_spread = 1
        graph.sn_spread = 0.1
        graph.specified_targets=None
        graph.specified_sources=None
        graph.specified_types=None
        graph.max_number_nodes=None
        graph.min_mean_hetesim=0
        graph.max_mean_hetesim=1
        graph.min_edge_hetesim = 0
        graph.max_edge_hetesim=1

        return [
            graph.update_graph_elements(), 
            graph.generate_table(), 
            graph.mean_hetesim_range, 
            [graph.min_edge_value, graph.max_edge_value], 
            graph.max_node_count, 
            '', 
            '', 
            '', 
            1, 
            0.1, 
            '', 
            '', 
            '', 
            '']

    targets_string_input = None
    sources_string_input = None
    type_string_input = None

    if ((specific_targets_val != None) and (specific_targets_val != '')):
        targets_string_input = []
        for s in specific_targets_val.split(','):
            targets_string_input.append(s.strip())

    if ((specific_sources_val != None) and (specific_sources_val != '')):
        sources_string_input = []
        for s in specific_sources_val.split(','):
            sources_string_input.append(s.strip())

    if ((specific_types_val != None) and (specific_types_val != '')):
        type_string_input = []
        for s in specific_types_val.split(','):
            type_string_input.append(s.strip())

    if ((type_gradient_start_val != None) and (type_gradient_start_val != '')):
        graph.type_pallet_start = type_gradient_start_val

    if ((type_gradient_end_val != None) and (type_gradient_end_val != '')):
        graph.type_pallet_end = type_gradient_end_val

    if ((type_color_mapping_val != None) and (type_color_mapping_val != '')):
        specified_type_color_dict = {}
        for entry in type_color_mapping_val.split(','):
            split_entry = entry.split(':')
            specified_type_color_dict[split_entry[0].strip()] = split_entry[1].strip()

        graph.type_color_mapping = specified_type_color_dict

    if ((target_color_mapping_val != None) and (target_color_mapping_val != '')):
        graph.target_color_mapping = target_color_mapping_val

    if color_randomization_submit_count > graph.color_randomization_clicks:
        graph.color_randomization_clicks = graph.color_randomization_clicks + 1
        graph.type_pallet_start=None
        graph.type_pallet_end=None
        graph.target_color_mapping=None
        graph.type_color_mapping=None
        graph.random_color = True

    graph._generate_color_mapping()
    graph.random_color = False

    graph.target_spread = target_spread_val
    graph.sn_spread = sn_spread_val
    graph.specified_targets = targets_string_input
    graph.specified_sources=sources_string_input
    graph.specified_types=type_string_input
    graph.max_number_nodes=max_node_slider_value
    graph.min_mean_hetesim=input_mean_hetesim_range_value[0]
    graph.max_mean_hetesim=input_mean_hetesim_range_value[1]
    graph.min_edge_hetesim=input_edge_hetesim_range_value[0]
    graph.max_edge_hetesim=input_edge_hetesim_range_value[1]

    return [
        graph.update_graph_elements(), 
        graph.generate_table(), 
        [graph.min_mean_hetesim, graph.max_mean_hetesim], 
        [graph.min_edge_hetesim, graph.max_edge_hetesim], 
        graph.max_number_nodes, 
        specific_targets_val, 
        specific_sources_val, 
        specific_types_val, 
        target_spread_val, 
        sn_spread_val, 
        type_gradient_start_val, 
        type_gradient_end_val, 
        type_color_mapping_val, 
        target_color_mapping_val]

@app.callback(
    Output('node_select_data_1', 'children'),
    Input('output_graph', 'selectedNodeData'))
def displayTapNodeData(data):

    if (data == []) or (data == None):
        return 'Select Node(s)'

    display_data = []

    for i, node in enumerate(data):
        if node['sn_or_tn'] == 'source_node':
            display_data.append(html.Div( 
                children=[graph.starting_nx_graph.nodes[node['id']]['name']], 
                style={'font-weight': 'bold'}))

            edges = {}
            for j, connecting_node in enumerate(graph.starting_nx_graph[node['id']]):
                edges[str(graph.target_cui_target_name_dict[connecting_node]) + '(CUI: ' + str(connecting_node) + ')'] = np.round(graph.starting_nx_graph[node['id']][connecting_node]['hetesim'], 3)

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
                edges[str(graph.starting_nx_graph.nodes[connecting_node]['name']) + ' (CUI: ' + str(graph.starting_nx_graph.nodes[connecting_node]['cui']) + ')'] = float(np.round(graph.starting_nx_graph[node['id']][connecting_node]['hetesim'], 3))
            
            edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

            data_dump = {
                'node_cui': graph.starting_nx_graph.nodes[node['id']]['id'], 
                'node_name': graph.starting_nx_graph.nodes[node['id']]['name'], 
                'sn_or_tn': 'target_node', 
                'edge_hetesim': edges_sorted
                }

            display_data.append(html.Pre(json.dumps(data_dump, indent=2)))

    return display_data


if __name__ == '__main__':
    app.run_server(debug=True)

# Future Ideas:
# Add additional tab for instructions, etc.
# Select edge Data
# User set target node positions and simulate sources
# User specified node coloring
# Host on website ORRRRRR Just run the app within a notebook... provide the link
# Truly randomize colors given one, two etc. types