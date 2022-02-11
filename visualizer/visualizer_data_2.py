from decimal import MAX_EMAX
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
import numpy as np
from scipy.signal import argrelextrema
import networkx as nx
from itertools import combinations

import dash
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from dash import html
from dash import dash_table as dt
from dash import dcc

import base64
import json

import matplotlib.pyplot as plt

app = dash.Dash()

# df_1 = pd.read_csv('D:\\GT_Research\\mitchell_lab\\semnet_visualizer_v2.0\\visualizer-2_repository\\visualizer-2\\semnet_test_data\\semnet_run_2\\SemNet_results_target=C0002395_semnet_run_1.csv', index_col=0)
# df_2 = pd.read_csv('D:\\GT_Research\\mitchell_lab\\semnet_visualizer_v2.0\\visualizer-2_repository\\visualizer-2\\semnet_test_data\\semnet_run_2\\SemNet_results_target=C0025519_semnet_run_1.csv', index_col=0)
# df_3 = pd.read_csv('D:\\GT_Research\\mitchell_lab\\semnet_visualizer_v2.0\\visualizer-2_repository\\visualizer-2\\semnet_test_data\\semnet_run_2\\SemNet_results_target=C0037313_semnet_run_1.csv', index_col=0)

def df_list_from_topdown_directory(path=str):
    df_list = []
    semnet_run_count = 0

    for root, _, files in os.walk(path, topdown=True):
        for name in files:
            file = os.path.join(root, name)

            if 'csv' in file:
                df = pd.read_csv(file, index_col=0)
                df['source_node'] = df['source_node'].astype(str) + '_' + str(semnet_run_count)
                df_list.append(df)
        
        semnet_run_count = semnet_run_count + 1

    return df_list

df_list = df_list_from_topdown_directory('D:\\GT_Research\\mitchell_lab\\semnet_visualizer_v2.0\\visualizer-2_repository\\visualizer-2\\semnet_test_data')

combined_df = pd.concat(df_list, ignore_index=True)

nx_graph = nx.from_pandas_edgelist(combined_df, 'source_node', 'target_node')

source_edge_series = combined_df['source_node']
target_edge_series = combined_df['target_node']

all_edge_df = pd.concat([source_edge_series, target_edge_series], ignore_index=True).to_frame(name='nodes')
all_edge_df['edge_count'] = all_edge_df.groupby('nodes')['nodes'].transform('count')
all_edge_df = all_edge_df.drop_duplicates(ignore_index=True)
all_edge_df = all_edge_df.sort_values(by=['edge_count'], ascending=False)

all_edge_array = all_edge_df['edge_count'].to_numpy()

X = all_edge_array.reshape(-1, 1)

kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(X)
edge_domain = np.linspace(0, np.max(X), np.shape(X)[0])
kde_curve = kde.score_samples(edge_domain.reshape(-1, 1))

min_array, max_array = argrelextrema(kde_curve, np.less)[0], argrelextrema(kde_curve, np.greater)[0]

range_list = [0]

for i in list(min_array):
    range_list.append(edge_domain[i])

range_list.append(np.max(X))
range_list = range_list[::-1]

'''
cluster_dict = {}
for i in range(len(range_list) - 1):
    cluster_dict[i] = set(all_edge_df[all_edge_df['edge_count'].between(range_list[i + 1], range_list[i])]['nodes'].unique())
'''

df_list = []
for i in range(len(range_list) - 1):
    sub_df = all_edge_df[all_edge_df['edge_count'].between(range_list[i + 1], range_list[i])]
    sub_df['cluster'] = i

    df_list.append(sub_df)

clustered_df = pd.concat(df_list, ignore_index=True)
clustered_df.to_csv('test')

# Additional second testing go:

cluster_val_array = clustered_df['cluster'].to_numpy()

X = cluster_val_array.reshape(-1, 1)

(unique, counts) = np.unique(X, return_counts=True)
frequencies = np.asarray((unique, counts)).T

freq_vals = frequencies[:, 1].reshape(-1, 1)

kmeans = KMeans(n_clusters=2, random_state=0).fit(freq_vals)

mapping_array = np.append(frequencies, kmeans.labels_.reshape(-1, 1), axis=1)
mapping_array = np.delete(mapping_array, 1, 1)

mapping_dict = {}
for cluster_1, cluster_2 in mapping_array:
    mapping_dict[cluster_1] = cluster_2

clustered_df['cluster_2'] = clustered_df['cluster'].map(mapping_dict)

clustered_df.to_csv('test_2')

initial_nodes = clustered_df[clustered_df['cluster_2'] == 0]['nodes'].to_list()

combination_set = set(list(combinations(initial_nodes, 2)))

nx_graph_copy = nx_graph.copy()

highly_connected_nodes_pairings = {}
combination_number_value_map = {}

for i, combination in enumerate(combination_set):
    nx_graph_copy = nx_graph.copy()

    intermediate_highly_connected_node = [x for x in initial_nodes if (x != combination[0]) and (x != combination[1])]

    nx_graph_copy.remove_nodes_from(intermediate_highly_connected_node)

    paths = list(nx.all_simple_paths(nx_graph_copy, combination[0], combination[1]))

    distance_dict = {}
    if len(paths) != 0:
        for path in paths:
            path_len = len(path) - 1
            if path_len in distance_dict:
                distance_dict[path_len] = distance_dict[path_len] + 1
            else:
                distance_dict[path_len] = 1

        highly_connected_nodes_pairings['combination_' + str(i)] = distance_dict
        combination_number_value_map['combination_' + str(i)] = combination

connected_nodes_dict = {}

max_edges = 0
min_edges = np.inf

max_path_len = 0
min_path_len = np.inf

initial_graph = nx.Graph()

for combination in highly_connected_nodes_pairings:
    for path_len in highly_connected_nodes_pairings[combination]:
        edge_count = highly_connected_nodes_pairings[combination][path_len]

        if edge_count > max_edges:
            max_edges = edge_count

        if (edge_count < min_edges) and (edge_count != 0):
            min_edges = edge_count

        if path_len > max_path_len:
            max_path_len = path_len

        if path_len < min_path_len:
            min_path_len = path_len

if (max_path_len - min_path_len) != 0:
    for comb in combination_number_value_map:
        cumulative_weight = 0

        differing_weight_spread = 1 / len(highly_connected_nodes_pairings[combination])

        for path_len in highly_connected_nodes_pairings[combination]:
            edge_count = highly_connected_nodes_pairings[combination][path_len]

            path_len_weight = 1 - 0.5 * ((path_len - min_path_len) / (max_path_len - min_path_len))
            edge_weight = 1 - 0.5 * ((edge_count - min_edges) / (max_edges - min_edges))

            cumulative_weight = cumulative_weight + (differing_weight_spread * path_len_weight * edge_weight)

        cumulative_weight = 100 - 50 * (cumulative_weight)

        initial_graph.add_edge(combination_number_value_map[comb][0], combination_number_value_map[comb][1], weight=cumulative_weight)

else:
    for comb in combination_number_value_map:
        cumulative_weight = 0

        for path_len in highly_connected_nodes_pairings[combination]:
            edge_count = highly_connected_nodes_pairings[combination][path_len]

            edge_weight = 1 - 0.5 * ((edge_count - min_edges) / (max_edges - min_edges))

            cumulative_weight = cumulative_weight + edge_weight

        cumulative_weight = 10 - 9 * (cumulative_weight)

        initial_graph.add_edge(combination_number_value_map[comb][0], combination_number_value_map[comb][1], weight=cumulative_weight)

# Determine all one-hop connected nodes
for comb in combination_set:
    paths = list(nx.all_simple_paths(nx_graph, comb[0], comb[1], cutoff=2))
    num_edges = len(list(paths))  
    if num_edges > 0:
        weight_adjusted = 5 - 4 * ((num_edges - min_edges) / (max_edges - min_edges))

        for path in paths:
            if len(path) == 3:
                if path[1] in connected_nodes_dict:
                    connected_nodes_dict[path[1]] = set.union(connected_nodes_dict[path[1]], {comb[0], comb[1]})

                else:
                    connected_nodes_dict[path[1]] = {comb[0], comb[1]}

initial_spring = nx.spring_layout(initial_graph, dim=2, weight='weight', k=1, iterations=50)

pos_dict = {}
noise_array = np.arange(-0.01, 0.01, 0.001)

print(connected_nodes_dict)

# Good place for added efficiency
for node in connected_nodes_dict:
    middle_x = 0
    middle_y = 0

    min_x = 0
    min_y = 0

    max_x = 0
    max_y = 0

    for connected_target in connected_nodes_dict[node]:
        middle_x = middle_x + initial_spring[connected_target][0]
        middle_y = middle_y + initial_spring[connected_target][1]

        if initial_spring[connected_target][0] > max_x:
            max_x = initial_spring[connected_target][0]
        
        if initial_spring[connected_target][0] < min_x:
            min_x = initial_spring[connected_target][0]

        if initial_spring[connected_target][1] > max_y:
            max_y = initial_spring[connected_target][1]
        
        if initial_spring[connected_target][1] < min_y:
            min_y = initial_spring[connected_target][1]

    middle_x = middle_x / len(connected_nodes_dict[node])
    middle_y = middle_y / len(connected_nodes_dict[node])

    min_range = min([(max_x - min_x), (max_y - min_y)])

    x_pos = np.random.normal(middle_x, (np.abs(min_range) / 8), 1)[0]
    y_pos = np.random.normal(middle_y, (np.abs(min_range) / 8), 1)[0]

    middle_x = middle_x + (middle_x * np.random.choice(noise_array))
    middle_y = middle_y + (middle_y * np.random.choice(noise_array))

    pos_dict[node] = (x_pos, y_pos)

final_graph = nx.from_pandas_edgelist(combined_df, 'source_node', 'target_node')
fixed_list = []

for entry in initial_graph.nodes:
    pos_dict[entry] = [initial_spring[entry][0], initial_spring[entry][1]]
    fixed_list.append(entry)

final_graph = nx.from_pandas_edgelist(combined_df, 'source_node', 'target_node')
final_spring_graph = nx.spring_layout(final_graph, dim=2, pos=pos_dict, fixed=fixed_list, k=0.03, iterations=5)

elements = []

for node in final_graph:
    elements.append({
        'data': {
            'id': node, 
            'label': node, 
            'size': '0.01px'
        }, 
        'position': {'x': final_spring_graph[node][0], 'y': final_spring_graph[node][1]}
    })
    

for node_1 in final_graph:
    for node_2 in final_graph[node_1]:
        elements.append({
            'data': {
                'source': node_1,
                'target': node_2
            }
        })



default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'width': 'data(size)',
            'height': 'data(size)',
            'content': '',
            'font-size': '2px', 
            'text-size': '0px'
            }
        },
    {
        'selector': 'edge',
        'style': {
            'width': '0.001px', 
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
    elements=elements,
    boxSelectionEnabled=True
    )

app.layout = html.Div([
    cytoscape_graph
])

if __name__ == '__main__':
    app.run_server(debug=True)

'''
'''
cluster_dict = {}
for i in range(len(range_list) - 1):
    cluster_dict[i] = set(clustered_df[clustered_df['cluster'].between(range_list[i + 1], range_list[i])]['nodes'].unique())

# For testing purposes
df_list = []
for i in range(len(range_list) - 1):
    sub_df = clustered_df[clustered_df['cluster'].between(range_list[i + 1], range_list[i])]
    sub_df['cluster_2'] = i

    df_list.append(sub_df)

clustered_df_2 = pd.concat(df_list, ignore_index=True)
clustered_df_2.to_csv('test2')


# Useful kde link: https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit