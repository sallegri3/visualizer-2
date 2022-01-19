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

start = time.time()

base_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

relationship_list = [{'T1', 'T2'}, {'T2', 'T3'}, {'T2', 'T3', 'T4'}, {'T4', 'T5'}, {'T5', 'T6'}, {'T6', 'T7'}, {'T7', 'T8'}, {'T7', 'T4'}]
type_list = ['type_1', 'type_2', 'type_3', 'type_4']
df_list_input = []

for i in range(len(relationship_list)):
    sn_list = []
    sn_name_list = []
    target_list = []
    hetesim_list = []
    type_list_random = []

    for name in base_names:
        sn_list.append(name)
        sn_name_list.append(name + '_name')

        temp_dict = {}
        for j in relationship_list[i]:
            temp_random_val = np.abs(np.random.normal(0, 0.5))

            if temp_random_val > 1:
                temp_random_val = 1
            if temp_random_val < 0:
                temp_random_val = 0

            temp_dict[j] = temp_random_val
            
        target_list.append(temp_dict)
        type_list_random.append(type_list[np.random.randint(0, len(type_list))])

    data = {
        'sn': sn_list, # CUI
        'sn_name': sn_name_list, # SN name
        'targets': target_list, # Target CUI
        'type': type_list_random # SN type
        }

    df_list_input.append(pd.DataFrame(data))

target_cui_target_name_dict_input = {
    'T1': 'T1_name', 
    'T2': 'T2_name', 
    'T3': 'T3_name', 
    'T4': 'T4_name', 
    'T5': 'T5_name', 
    'T6': 'T6_name', 
    'T7': 'T7_name', 
    'T8': 'T8_name'
    }

###############################################################################################

def format_visualizer_data(df_list):

    df_list_formatted = []

    target_relationship_list_formatted = []
    sn_mean_hetesim_dict_formatted = {}
    sn_type_dict_formatted = {}

    for i, df in enumerate(df_list):
        sub_df_formatted = pd.DataFrame(columns=['sn', 'type', 'sn_name', 'target', 'hetesim', 'mean_hetesim'])

        for _, row in df.iterrows():
            sn_adjusted = str(row['sn']) + '_' + str(i)
            sn_type_adjusted = row['type']
            sn_name_adjusted = row['sn_name']
            sn_mean_hetesim_adjusted = []

            target_dict = row['targets']

            for target in target_dict:
                sn_mean_hetesim_adjusted.append(target_dict[target])
            
            for target in target_dict:
                sub_df_formatted = sub_df_formatted.append({
                    'sn': sn_adjusted,
                    'type': sn_type_adjusted,
                    'sn_cui': row['sn'],
                    'sn_name': sn_name_adjusted,
                    'target': target, 
                    'hetesim': target_dict[target],
                    'mean_hetesim': np.mean(sn_mean_hetesim_adjusted)
                    }, ignore_index=True)

                sub_df_formatted = sub_df_formatted.sort_values(by=['mean_hetesim'], ascending=False)

            sn_mean_hetesim_dict_formatted[sn_adjusted] = np.mean(sn_mean_hetesim_adjusted)
            sn_type_dict_formatted[sn_adjusted] = sn_type_adjusted

        target_relationship_list_formatted.append(tuple(df.iloc[0]['targets'].keys()))

        df_list_formatted.append(sub_df_formatted)

    return [df_list_formatted, target_relationship_list_formatted, sn_mean_hetesim_dict_formatted, sn_type_dict_formatted]


def adjust_visualizer_data(
    formatted_df_list_adjusting,
    specified_targets=None,
    specified_sources=None, 
    specified_types=None,
    max_number_nodes=None,
    min_mean_hetesim=0,
    max_mean_hetesim=1, 
    min_edge_hetesim = 0,
    max_edge_hetesim=1):

    formatted_df_list_generating_local = formatted_df_list_adjusting.copy()

    if specified_targets != None:
        formatted_df_list_generating_local = _select_specific_targets(formatted_df_list_generating_local, specified_targets)

    if specified_sources != None:
        formatted_df_list_generating_local = _select_specific_sources(formatted_df_list_generating_local, specified_sources)

    if specified_types != None:
        formatted_df_list_generating_local = _select_specific_types(formatted_df_list_generating_local, specified_types)
    
    if max_number_nodes != None:
        formatted_df_list_generating_local = _select_max_nodes(formatted_df_list_generating_local, max_number_nodes)

    if (min_mean_hetesim != 0) or (max_mean_hetesim != 1):
        formatted_df_list_generating_local = _select_max_min_hetesim_range(formatted_df_list_generating_local, 'mean_hetesim', min_mean_hetesim, max_mean_hetesim)

    if (min_edge_hetesim != 0) or (max_edge_hetesim != 1):
        formatted_df_list_generating_local = _select_max_min_hetesim_range(formatted_df_list_generating_local, 'hetesim', min_edge_hetesim, max_edge_hetesim)
    
    return formatted_df_list_generating_local


def generate_nx_graph(
    formatted_df_list_generating, 
    target_cui_target_name_dict, 
    target_relationship_list_generating, 
    sn_mean_hetesim_dict_generating, 
    sn_type_dict_generating,
    target_spread=1,
    sn_spread=0.1):

    combined_df_generating = pd.concat(formatted_df_list_generating)
    
    unique_types = combined_df_generating['type'].unique()

    type_color_dict = _generate_color_mapping(
        unique_types, 
        random=True)

    target_edges = []
    for relationship in target_relationship_list_generating:
        for subset in itertools.combinations(relationship, 2):
            target_edges.append(subset)

    initial_graph = nx.Graph(target_edges)
    initial_spring = nx.spring_layout(initial_graph, dim=2, k=target_spread, iterations=100)

    pos_dict = {}
    fixed_list = []

    for entry in initial_graph.nodes:
        pos_dict[entry] = [initial_spring[entry][0], initial_spring[entry][1]]
        fixed_list.append(entry)

    final_graph = nx.from_pandas_edgelist(combined_df_generating, 'sn', 'target', ['sn_name', 'sn_cui', 'hetesim', 'mean_hetesim'])
    final_spring_graph = nx.spring_layout(final_graph, dim=2, pos=pos_dict, fixed=fixed_list, k=sn_spread, iterations=100)

    print(type_color_dict)

    for _, row in combined_df_generating.iterrows():
        final_graph.nodes[row['sn']]['id'] = row['sn']
        final_graph.nodes[row['sn']]['cui'] = row['sn_cui']
        final_graph.nodes[row['sn']]['name'] = row['sn_name']
        final_graph.nodes[row['sn']]['type'] = row['type']
        final_graph.nodes[row['sn']]['mean_hetesim'] = row['mean_hetesim']
        final_graph.nodes[row['sn']]['size'] = 10 * row['mean_hetesim']
        final_graph.nodes[row['sn']]['color'] = str(type_color_dict[sn_type_dict_generating[row['sn']]])
        #final_graph.nodes[row['sn']]['color'] = 'hsl(' + str(type_color_dict[sn_type_dict_generating[row['sn']]]) + ', 100%, 54%)'
        final_graph.nodes[row['sn']]['sn_or_tn'] = 'source_node'
        final_graph.nodes[row['sn']]['position'] = {'x': 100 * final_spring_graph[row['sn']][0], 'y': 100 * final_spring_graph[row['sn']][1]}

    for target_cui in combined_df_generating['target'].unique():
        final_graph.nodes[target_cui]['id'] = target_cui
        final_graph.nodes[target_cui]['name'] = target_cui_target_name_dict[target_cui]
        final_graph.nodes[target_cui]['color'] = type_color_dict['target']
        final_graph.nodes[target_cui]['size'] = 10
        final_graph.nodes[target_cui]['sn_or_tn'] = 'target_node'
        final_graph.nodes[target_cui]['position'] = {'x': 100 * final_spring_graph[target_cui][0], 'y': 100 * final_spring_graph[target_cui][1]}

    return final_graph


def generate_graph_elements(nx_graph):

    elements = []
    for node in nx_graph.nodes:
        if nx_graph.nodes[node]['sn_or_tn'] == 'target_node':
            elements.append({
                'data': {
                    'id': nx_graph.nodes[node]['id'], 
                    'label': nx_graph.nodes[node]['name'], 
                    'size': nx_graph.nodes[node]['size'], 
                    'color': nx_graph.nodes[node]['color'],
                    'sn_or_tn': 'target_node'}, 
                'position': nx_graph.nodes[node]['position']})

        if nx_graph.nodes[node]['sn_or_tn'] == 'source_node':
            elements.append({
                'data': {
                    'id': nx_graph.nodes[node]['id'], 
                    'label': nx_graph.nodes[node]['name'],
                    'mean_hetesim': nx_graph.nodes[node]['mean_hetesim'], 
                    'size': nx_graph.nodes[node]['size'], 
                    'color': nx_graph.nodes[node]['color'],
                    'sn_or_tn': 'source_node'}, 
                'position': nx_graph.nodes[node]['position']})

    for node_1 in nx_graph:
        for node_2 in nx_graph[node_1]:
            elements.append({
                'data': {'source': node_1, 
                'target': node_2, 
                'size': np.round(nx_graph[node_1][node_2]['hetesim'], 2), 
                'label': np.round(nx_graph[node_1][node_2]['hetesim'], 2)}})

    return elements


def _generate_color_mapping(
    unique_types_generating, 
    random=True,
    type_pallet_start=None, 
    type_pallet_end=None, 
    target_color_mapping=None, 
    type_color_mapping=None):

    # '#940D02' burgundy color

    type_color_dict_generating = {}

    if random:
        color_intervals = (330 / len(unique_types_generating))

        for i, type in enumerate(unique_types_generating):

            normal_val = np.abs(np.random.normal(0.5, 0.33, 1)[0])
            if normal_val > 1 : normal_val = 1

            normal_color = color_intervals * normal_val
            type_color_dict_generating[type] = 'hsl(' + str((color_intervals * i) + normal_color) + ', 100%, 60%)'

        type_color_dict_generating['target'] = 'hsl(0, 100%, 60%)'

    if (type_pallet_start != None) and (type_pallet_end != None):
        starting_color = Color(type_pallet_start)
        color_gradient_list = list(starting_color.range_to(Color(type_pallet_end), len(unique_types_generating)))

        for i, type in enumerate(unique_types_generating):
            type_color_dict_generating[type] = color_gradient_list[i]

    if type_color_mapping != None:
        for type in type_color_mapping:
            type_color_dict_generating[type] = type_color_mapping[type]

    if target_color_mapping != None:
        type_color_dict_generating['target'] = target_color_mapping

    return type_color_dict_generating

def _select_max_min_hetesim_range(df_list, df_column, min_hetesim, max_hetesim):

    adjusted_df_list = []

    for df in df_list:
        df = df[df[df_column] >= min_hetesim]
        df = df[df[df_column] <= max_hetesim]
        adjusted_df_list.append(df)

    return adjusted_df_list


def _select_max_nodes(df_list, max_node_count):

    adjusted_df_list = []

    for df in df_list:
        unique_max_count_sns = df['sn'].unique()[:max_node_count]
        df = df[df['sn'].isin(unique_max_count_sns)]
        adjusted_df_list.append(df)

    return adjusted_df_list


def _select_specific_targets(df_list, target_list):

    adjusted_df_list = []

    for df in df_list:
        adjusted_df_list.append(df[df['target'].isin(target_list)])

    return adjusted_df_list


def _select_specific_types(df_list, type_list):

    adjusted_df_list = []

    for df in df_list:
        adjusted_df_list.append(df[df['type'].isin(type_list)])

    return adjusted_df_list


def _select_specific_sources(df_list, source_list):

    adjusted_df_list = []

    for df in df_list:
        adjusted_df_list.append(df[df['sn_cui'].isin(source_list)])

    return adjusted_df_list


def mean_hetesim_range(formatted_df_list_mean_hetesim_range):

    combined_df_mean_hetesim_range = pd.concat(formatted_df_list_mean_hetesim_range)

    return [combined_df_mean_hetesim_range['mean_hetesim'].min(), combined_df_mean_hetesim_range['mean_hetesim'].max()]


def generate_table(adjusted_df_list_generating):

    combined_df_generating = pd.concat(adjusted_df_list_generating)
    combined_df_generating = combined_df_generating.round({'hetesim': 3, 'mean_hetesim': 3})
    combined_df_generating = combined_df_generating[['target', 'sn_name', 'hetesim', 'mean_hetesim', 'type', 'sn_cui']]
    combined_df_generating = combined_df_generating.replace({"target": target_cui_target_name_dict_input})

    sorted_df_list = []

    for target in combined_df_generating['target'].unique():
        specific_target_df = combined_df_generating[combined_df_generating['target'] == target].copy()
        sorted_df_list.append(specific_target_df.sort_values(['hetesim'], ascending=False))

    combined_df_local = pd.concat(sorted_df_list)

    return dt.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in combined_df_local.columns],
        data=combined_df_local.to_dict('records'), 
        page_size=50)

formatted_data_list = format_visualizer_data(df_list_input)

formatted_df_list = formatted_data_list[0]
target_relationship_list = formatted_data_list[1]
sn_mean_hetesim_dict = formatted_data_list[2]
sn_type_dict = formatted_data_list[3]

starting_nx_graph = generate_nx_graph(formatted_df_list, target_cui_target_name_dict_input, target_relationship_list, sn_mean_hetesim_dict, sn_type_dict)

starting_elements = generate_graph_elements(starting_nx_graph)

starting_hetesim_range = mean_hetesim_range(formatted_df_list)

# starting_color_type_dict = _generate_color_mapping()

n_submit_clicks = 0

max_edge_value = -np.inf
for df in formatted_df_list:
    if df['hetesim'].max() > max_edge_value:
        max_edge_value = df['hetesim'].max()
max_edge_value = max_edge_value

min_edge_value = np.inf
for df in formatted_df_list:
    if df['hetesim'].min() < min_edge_value:
        min_edge_value = df['hetesim'].min()
min_edge_value = min_edge_value

max_node_count = 0
for df in formatted_df_list:
    if df.shape[0] > max_node_count:
        max_node_count = df.shape[0]
max_node_count = max_node_count / 2

app = dash.Dash(__name__)

default_stylesheet = [
        {
            "selector": "node",
            "style": {
                'width': "data(size)",
                'height': "data(size)",
                "content": "data(label)",
                "font-size": "1px",
                "text-valign": "center",
                "text-halign": "center",
                'background-color': 'data(color)'
            }
        },
            {
            "selector": "edge",
            "style": {
                'width': "data(size)", 
                'line-opacity': "data(size)", 
                'label': "data(label)", 
                'font-size': 1, 
                'font-opacity': 1
            }
        }, 
        {
            "selector": ":selected",
            "style": {
                'border-color': 'black', 
                "border-opacity": "1", 
                "border-width": "0.3px"
            }
        }
    ]

app.layout = html.Div([

    html.Div(children=[

        dcc.Tabs(
            id="tabs_global",
            value='T1',
            children=[

                dcc.Tab(
                    label='Graph Data',
                    value='T1',
                    children=[

                        html.Div(children=[
                            html.Div(children=[
                                'Node HeteSim:'
                            ], style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.RangeSlider(
                                id='hetesim_mean_range_slider',
                                min=0,
                                max=1,
                                step=0.01,
                                value=starting_hetesim_range,
                                tooltip={"placement": "bottom", "always_visible": True},
                                vertical=False
                            )], style={'width': '70%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}), 

                        html.Div(children=[
                            html.Div(children=[
                                'Edge HeteSim:'
                            ], style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.RangeSlider(
                                id='hetesim_edge_range_slider',
                                min=0,
                                max=1,
                                step=0.01,
                                value=[min_edge_value, max_edge_value],
                                tooltip={"placement": "bottom", "always_visible": True},
                                vertical=False
                            )], style={'width': '70%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}),

                        html.Div(children=[
                            html.Div(children=[
                                'Max Nodes:'
                            ], style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Slider(
                                id='node_max_slider',
                                min=0,
                                max=max_node_count,
                                step=1,
                                value=max_node_count,
                                tooltip={"placement": "bottom", "always_visible": True},
                                vertical=False
                            )], style={'width': '70%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}),

                        html.Div(style={"border": "1px black solid", 'margin':'5%', 'width': '90%'}),

                        html.Div(children=[
                            html.Div(children=[
                                'Specific TN CUIs:'
                            ], style={'width': '35%', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Input(id="specific_target_input", type="text", placeholder="Separate with commas")
                            ], style={'width': '45%', 'display': 'inline-block'}),
                            html.Div(children=[
                                html.Button('Submit', id='specific_target_submit_button', n_clicks=0)
                            ], style={'width': '20%', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}),

                        html.Div(children=[
                            html.Div(children=[
                                'Specific SN CUIs:'
                            ], style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Input(id="specific_source_input", type="text", placeholder="Separate with commas")
                            ], style={'width': '45%', 'height': '10px', 'display': 'inline-block'}),
                            html.Div(children=[
                                html.Button('Submit', id='specific_source_submit_button', n_clicks=0)
                            ], style={'width': '20%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}), 

                        html.Div(children=[
                            html.Div(children=[
                                'Specific SN Types:'
                            ], style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Input(id="specific_type_input", type="text", placeholder="Separate with commas")
                            ], style={'width': '45%', 'height': '10px', 'display': 'inline-block'}),
                            html.Div(children=[
                                html.Button('Submit', id='specific_type_submit_button', n_clicks=0)
                            ], style={'width': '20%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'})

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
                    label='Graph Visuals',
                    value='T2',
                    className='custom-tab',
                    selected_className='custom-tab--selected', 
                    children=[

                        html.Div(children=[
                            html.Div(children=[
                                'Target Spread:'
                            ], style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Slider(
                                id='node_spread_slider',
                                min=0.1,
                                max=3,
                                step=0.01,
                                value=1,
                                tooltip={"placement": "bottom", "always_visible": True},
                                vertical=False
                            )], style={'width': '70%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}),
                        html.Div(children=[
                            html.Div(children=[
                                'SN Spread:'
                            ], style={'width': '30%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Slider(
                                id='node_spread_slider_2',
                                min=0.01,
                                max=0.3,
                                step=0.01,
                                value=0.1,
                                tooltip={"placement": "bottom", "always_visible": True},
                                vertical=False
                            )], style={'width': '70%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}), 

                        html.Div(style={"border": "1px black solid", 'margin':'5%', 'width': '90%'}),

                        html.Div(children=[
                            html.Div(children=[
                                'Type Gradient Start:'
                            ], style={'width': '35%', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Input(id="type_gradient_start_input", type="text", placeholder="Separate with commas")
                            ], style={'width': '45%', 'display': 'inline-block'}),
                            html.Div(children=[
                                html.Button('Submit', id='type_gradient_start_submit_button', n_clicks=0)
                            ], style={'width': '20%', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}),

                        html.Div(children=[
                            html.Div(children=[
                                'Type Gradient End:'
                            ], style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Input(id="type_gradient_end_input", type="text", placeholder="Separate with commas")
                            ], style={'width': '45%', 'height': '10px', 'display': 'inline-block'}),
                            html.Div(children=[
                                html.Button('Submit', id='type_gradient_end_submit_button', n_clicks=0)
                            ], style={'width': '20%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}), 

                        html.Div(children=[
                            html.Div(children=[
                                'Type Color Mapping:'
                            ], style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Input(id="type_color_mapping_input", type="text", placeholder="Separate with commas")
                            ], style={'width': '45%', 'height': '10px', 'display': 'inline-block'}),
                            html.Div(children=[
                                html.Button('Submit', id='type_color_mapping_submit_button', n_clicks=0)
                            ], style={'width': '20%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'}), 

                        html.Div(children=[
                            html.Div(children=[
                                'Target Color Mapping:'
                            ], style={'width': '35%', 'height': '10px', 'display': 'inline-block', 'font-size': '0.9vw'}),
                            html.Div(children=[
                                dcc.Input(id="target_color_mapping_input", type="text", placeholder="Separate with commas")
                            ], style={'width': '45%', 'height': '10px', 'display': 'inline-block'}),
                            html.Div(children=[
                                html.Button('Submit', id='target_color_mapping_submit_button', n_clicks=0)
                            ], style={'width': '20%', 'height': '10px', 'display': 'inline-block'})
                        ], style={'margin':'5%', 'width': '90%', 'height': '10%', 'display': 'inline-block'})
                    ],
                    style= {
                        'borderLeft': '2px solid black', 
                        'borderBottom': '2px solid black',
                        'text-align': 'center'
                    },
                    selected_style={
                        'text-align': 'center',
                        'borderTop': 'None',
                        'borderRight': 'None'
                    }
                )

            ])

    ], style={'margin':'0.5%', "border":"2px black solid", 'width': '24%', 'height': '30vw', 'display': 'inline-block'}),

    html.Div(children=[

        cyto.Cytoscape(
            id='output_graph',
            layout={'name': 'preset'},
            style={'width': '100%', 'height': '30vw'},
            stylesheet=default_stylesheet,
            elements=starting_elements,
            boxSelectionEnabled=True
        )

    ], style={'margin':'0.5%', "border":"2px black solid", 'width': '48%', 'height': '30vw'}),

    html.Div(children=[
        dcc.Tabs(
            children=[
                dcc.Tab(
                    label='Node Data',
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                        id='node_select_data_1',
                                        children='Select Node(s)', 
                                        style={'width': '99%', 'height': '99%'})
                                    ], 
                                    style={})
                            ], style={"overflowY":"scroll", 'height': '26.8vw'})
                    ],
                    style= {
                        'borderRight': '2px solid black', 
                        'borderBottom': '2px solid black',
                        'text-align': 'center'
                        },
                    selected_style={
                        'text-align': 'center',
                        'borderTop': 'None',
                        'borderLeft': 'None'}
                ),
                    dcc.Tab(
                        label='Table Data',
                        children=[
                            html.Div(
                                id='table_data', 
                                children=[], 
                                style={'overflowX': 'scroll', "overflowY": "scroll", 'height': '26.8vw'}
                            )
                        ],
                    style= {
                        'borderLeft': '2px solid black', 
                        'borderBottom': '2px solid black',
                        'text-align': 'center'
                    },
                    selected_style={
                        'text-align': 'center',
                        'borderTop': 'None',
                        'borderRight': 'None'
                    }
                    
                )
            ])

    ], style={'margin':'0.5%', "border":"2px black solid", 'width': '24%', 'height': '30vw', 'display': 'inline-block'})

    #html.Div(children=[
    #    html.Div(
    #        id='node_select_data_1',
    #        children='Select Node(s)', 
    #        style={'width': '99%', 'margin':'0.5%', 'font-size': '0.9vw'}), 
    #], style={'margin':'0.5%', 'overflowY': 'scroll', "border": "2px black solid", 'width': '24%', 'height': '30vw'})
    
], style={'display': 'flex', 'flex-direction': 'row'})

@app.callback(
    Output(component_id='output_graph', component_property='elements'),
    Output(component_id='table_data', component_property='children'),
    [Input(component_id='hetesim_mean_range_slider', component_property='value')],
    [Input(component_id='hetesim_edge_range_slider', component_property='value')],
    Input(component_id='node_max_slider', component_property='value'),
    Input(component_id='node_spread_slider', component_property='value'),
    Input(component_id='node_spread_slider_2', component_property='value'),
    State(component_id='specific_target_input', component_property='value'),
    Input(component_id='specific_target_submit_button', component_property='n_clicks'), 
    State(component_id='specific_source_input', component_property='value'),
    Input(component_id='specific_source_submit_button', component_property='n_clicks'), 
    State(component_id='specific_type_input', component_property='value'),
    Input(component_id='specific_type_submit_button', component_property='n_clicks')
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
    specific_types_submit_count):

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

    adjusted_df_list = adjust_visualizer_data(
        formatted_df_list,
        specified_targets=targets_string_input, 
        specified_sources=sources_string_input, 
        specified_types=type_string_input,
        max_number_nodes=max_node_slider_value,
        min_mean_hetesim=input_mean_hetesim_range_value[0],
        max_mean_hetesim=input_mean_hetesim_range_value[1], 
        min_edge_hetesim=input_edge_hetesim_range_value[0],
        max_edge_hetesim=input_edge_hetesim_range_value[1])

    return [generate_graph_elements(generate_nx_graph(
        adjusted_df_list, 
        target_cui_target_name_dict_input,
        target_relationship_list, 
        sn_mean_hetesim_dict, 
        sn_type_dict,
        target_spread_val, 
        sn_spread_val)), 
        generate_table(adjusted_df_list)]

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
                children=[starting_nx_graph.nodes[node['id']]['name']], 
                style={'font-weight': 'bold'}))

            edges = {}
            for j, connecting_node in enumerate(starting_nx_graph[node['id']]):
                edges[str(target_cui_target_name_dict_input[connecting_node]) + '(CUI: ' + str(connecting_node) + ')'] = np.round(starting_nx_graph[node['id']][connecting_node]['hetesim'], 3)

            edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

            data_dump = {
                'node_cui': starting_nx_graph.nodes[node['id']]['cui'], 
                'node_name': starting_nx_graph.nodes[node['id']]['name'], 
                'node_type': starting_nx_graph.nodes[node['id']]['type'], 
                'mean_hetesim': np.round(starting_nx_graph.nodes[node['id']]['mean_hetesim'], 3), 
                'sn_or_tn': 'source_node', 
                'edge_hetesim': edges_sorted
                }

            display_data.append(html.Pre(json.dumps(data_dump, indent=2)))

        if node['sn_or_tn'] == 'target_node':
            display_data.append(html.Div( 
                children=[starting_nx_graph.nodes[node['id']]['name']], 
                style={'font-weight': 'bold'}))

            edges = {}
            for j, connecting_node in enumerate(starting_nx_graph[node['id']]):
                edges[str(starting_nx_graph.nodes[connecting_node]['cui']) + '(CUI: ' + str(connecting_node) + ')'] = float(np.round(starting_nx_graph[node['id']][connecting_node]['hetesim'], 3))
            
            edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

            data_dump = {
                'node_cui': starting_nx_graph.nodes[node['id']]['id'], 
                'node_name': starting_nx_graph.nodes[node['id']]['name'], 
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