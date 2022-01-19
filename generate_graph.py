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

class Generate_Graph:
    def __init__(self, df_list=None, target_cui_target_name_dict=None) -> None:

        self.target_spread = 1
        self.sn_spread = 0.1
        self.specified_targets=None
        self.specified_sources=None
        self.specified_types=None
        self.max_number_nodes=None
        self.min_mean_hetesim=0
        self.max_mean_hetesim=1
        self.min_edge_hetesim = 0
        self.max_edge_hetesim=1

        self.graph_reset_clicks = 0
        self.color_randomization_clicks = 0

        self.random_color=True,
        self.type_pallet_start=None
        self.type_pallet_end=None
        self.target_color_mapping=None
        self.type_color_mapping=None

        self._format_data(df_list, target_cui_target_name_dict)
        self._generate_color_mapping()
        self._adjust_data()

        self.starting_nx_graph = self._generate_nx_graph()
        self.starting_elements = self.generate_graph_elements()

        self.starting_mean_hetesim_range = self.mean_hetesim_range
        self.starting_min_edge_value = self.min_edge_value
        self.starting_max_edge_value = self.max_edge_value
        self.starting_max_node_count = self.max_node_count

        self.random_color = False

    def _generate_dummy_graph(self):

        base_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        relationship_list = [{'T1', 'T2'}, {'T2', 'T3'}, {'T2', 'T3', 'T4'}, {'T4', 'T5'}, {'T5', 'T6'}, {'T6', 'T7'}, {'T7', 'T8'}, {'T7', 'T4'}]
        type_list = ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'type_6', 'type_7', 'type_8']
        dummy_df_list = []

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

            dummy_df_list.append(pd.DataFrame(data))

        dummy_target_cui_target_name_dict = {
            'T1': 'T1_name', 
            'T2': 'T2_name', 
            'T3': 'T3_name', 
            'T4': 'T4_name', 
            'T5': 'T5_name', 
            'T6': 'T6_name', 
            'T7': 'T7_name', 
            'T8': 'T8_name'
            }

        return [dummy_df_list, dummy_target_cui_target_name_dict]

    def _format_data(self, df_list_input, target_cui_target_name_dict_input):

        if (df_list_input == None) & (target_cui_target_name_dict_input == None):
            dummy_data = self._generate_dummy_graph()
            df_list_input = dummy_data[0]
            target_cui_target_name_dict_input = dummy_data[1]

        formatted_df_list = []

        target_relationship_list_formatted = []
        sn_mean_hetesim_dict_formatted = {}
        sn_type_dict_formatted = {}

        for i, df in enumerate(df_list_input):
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

            formatted_df_list.append(sub_df_formatted)

        self.formatted_df_list = formatted_df_list
        self.target_relationship_list_formatted = target_relationship_list_formatted
        self.sn_mean_hetesim_dict_formatted = sn_mean_hetesim_dict_formatted
        self.sn_type_dict_formatted = sn_type_dict_formatted
        self.target_cui_target_name_dict = target_cui_target_name_dict_input

        self.combined_formatted_df = pd.concat(formatted_df_list)
        self.unique_types = self.combined_formatted_df['type'].unique()
        
        max_node_count = 0
        for df in formatted_df_list:
            if df.shape[0] > max_node_count:
                max_node_count = df.shape[0]
        self.max_node_count = int(max_node_count / 2)

        max_edge_value = -np.inf
        for df in formatted_df_list:
            if df['hetesim'].max() > max_edge_value:
                max_edge_value = df['hetesim'].max()
        self.max_edge_value = max_edge_value

        min_edge_value = np.inf
        for df in formatted_df_list:
            if df['hetesim'].min() < min_edge_value:
                min_edge_value = df['hetesim'].min()
        self.min_edge_value = min_edge_value

        self.mean_hetesim_range = [self.combined_formatted_df['mean_hetesim'].min(), self.combined_formatted_df['mean_hetesim'].max()]

        self.max_node_size_ref = 10 / self.combined_formatted_df['mean_hetesim'].max()
        self.mean_hetesim_step_size = np.round((np.round(self.mean_hetesim_range[1], 3) - np.round(self.mean_hetesim_range[0], 3)) / 100, 3)
        self.edge_hetesim_step_size = np.round((self.max_edge_value - self.min_edge_value) / 100, 3)

        return formatted_df_list
    
    def _adjust_data(self):

        adjusted_df_list = self.formatted_df_list.copy()

        if self.specified_targets != None:
            adjusted_df_list = self._select_specific_targets(adjusted_df_list, self.specified_targets)

        if self.specified_sources != None:
            adjusted_df_list = self._select_specific_sources(adjusted_df_list, self.specified_sources)

        if self.specified_types != None:
            adjusted_df_list = self._select_specific_types(adjusted_df_list, self.specified_types)
        
        if self.max_number_nodes != None:
            adjusted_df_list = self._select_max_nodes(adjusted_df_list, self.max_number_nodes)

        if (self.min_mean_hetesim != 0) or (self.max_mean_hetesim != 1):
            adjusted_df_list = self._select_max_min_hetesim_range(adjusted_df_list, 'mean_hetesim', self.min_mean_hetesim, self.max_mean_hetesim)

        if (self.min_edge_hetesim != 0) or (self.max_edge_hetesim != 1):
            adjusted_df_list = self._select_max_min_hetesim_range(adjusted_df_list, 'hetesim', self.min_edge_hetesim, self.max_edge_hetesim)
        
        self.adjusted_df_list = adjusted_df_list
        self.combined_adjusted_df = pd.concat(adjusted_df_list)

    def _generate_nx_graph(self):

        target_edges = []
        for relationship in self.target_relationship_list_formatted:
            for subset in itertools.combinations(relationship, 2):
                target_edges.append(subset)

        initial_graph = nx.Graph(target_edges)
        initial_spring = nx.spring_layout(initial_graph, dim=2, k=self.target_spread, iterations=100)

        pos_dict = {}
        fixed_list = []

        for entry in initial_graph.nodes:
            pos_dict[entry] = [initial_spring[entry][0], initial_spring[entry][1]]
            fixed_list.append(entry)

        final_graph = nx.from_pandas_edgelist(self.combined_adjusted_df, 'sn', 'target', ['sn_name', 'sn_cui', 'hetesim', 'mean_hetesim'])
        final_spring_graph = nx.spring_layout(final_graph, dim=2, pos=pos_dict, fixed=fixed_list, k=self.sn_spread, iterations=100)

        for _, row in self.combined_adjusted_df.iterrows():
            final_graph.nodes[row['sn']]['id'] = row['sn']
            final_graph.nodes[row['sn']]['cui'] = row['sn_cui']
            final_graph.nodes[row['sn']]['name'] = row['sn_name']
            final_graph.nodes[row['sn']]['type'] = row['type']
            final_graph.nodes[row['sn']]['mean_hetesim'] = row['mean_hetesim']
            final_graph.nodes[row['sn']]['size'] = self.max_node_size_ref * row['mean_hetesim']
            final_graph.nodes[row['sn']]['color'] = str(self.type_color_dict[self.sn_type_dict_formatted[row['sn']]])
            final_graph.nodes[row['sn']]['sn_or_tn'] = 'source_node'
            final_graph.nodes[row['sn']]['position'] = {'x': 100 * final_spring_graph[row['sn']][0], 'y': 100 * final_spring_graph[row['sn']][1]}

        for target_cui in self.combined_adjusted_df['target'].unique():
            final_graph.nodes[target_cui]['id'] = target_cui
            final_graph.nodes[target_cui]['name'] = self.target_cui_target_name_dict[target_cui]
            final_graph.nodes[target_cui]['color'] = self.type_color_dict['target']
            final_graph.nodes[target_cui]['size'] = 10
            final_graph.nodes[target_cui]['sn_or_tn'] = 'target_node'
            final_graph.nodes[target_cui]['position'] = {'x': 100 * final_spring_graph[target_cui][0], 'y': 100 * final_spring_graph[target_cui][1]}

        self.nx_graph = final_graph

        return final_graph

    def generate_graph_elements(self):

        elements = []
        for node in self.nx_graph.nodes:
            if self.nx_graph.nodes[node]['sn_or_tn'] == 'target_node':
                elements.append({
                    'data': {
                        'id': self.nx_graph.nodes[node]['id'], 
                        'label': self.nx_graph.nodes[node]['name'], 
                        'size': self.nx_graph.nodes[node]['size'], 
                        'color': self.nx_graph.nodes[node]['color'],
                        'sn_or_tn': 'target_node'}, 
                    'position': self.nx_graph.nodes[node]['position']})

            if self.nx_graph.nodes[node]['sn_or_tn'] == 'source_node':
                elements.append({
                    'data': {
                        'id': self.nx_graph.nodes[node]['id'], 
                        'label': self.nx_graph.nodes[node]['name'],
                        'mean_hetesim': self.nx_graph.nodes[node]['mean_hetesim'], 
                        'size': self.nx_graph.nodes[node]['size'], 
                        'color': self.nx_graph.nodes[node]['color'],
                        'sn_or_tn': 'source_node'}, 
                    'position': self.nx_graph.nodes[node]['position']})

        for node_1 in self.nx_graph:
            for node_2 in self.nx_graph[node_1]:
                elements.append({
                    'data': {'source': node_1, 
                    'target': node_2, 
                    'size': np.round(self.nx_graph[node_1][node_2]['hetesim'], 2), 
                    'label': np.round(self.nx_graph[node_1][node_2]['hetesim'], 2)}})

        self.elements = elements

        return elements

    def update_graph_elements(self):

        self._adjust_data()
        self._generate_nx_graph()
        self.generate_graph_elements()

        return self.elements

    def _generate_color_mapping(self):

        if self.random_color:
            self.type_color_dict = {}
            color_intervals = (330 / len(self.unique_types))
            random_color_list = []

            for i in range(len(self.unique_types)):

                normal_val = np.abs(np.random.normal(0.5, 0.33, 1)[0])
                if normal_val > 1 : normal_val = 1

                normal_color = color_intervals * normal_val
                random_color_list.append('hsl(' + str((color_intervals * i) + normal_color) + ', 100%, 60%)')

            random_color_list = np.random.choice(random_color_list, len(self.unique_types), replace=False)

            for i, type in enumerate(self.unique_types):
                self.type_color_dict[type] = random_color_list[i]

            self.type_color_dict['target'] = 'hsl(0, 100%, 60%)'

        if (self.type_pallet_start != None) and (self.type_pallet_end != None) and (self.type_pallet_start != '') and (self.type_pallet_end != ''):
            starting_color = Color(self.type_pallet_start)
            color_gradient_list = list(starting_color.range_to(Color(self.type_pallet_end), len(self.unique_types)))

            for i, type in enumerate(self.unique_types):
                self.type_color_dict[type] = color_gradient_list[i]

        if (self.type_color_mapping != None) and (self.type_color_mapping != ''):
            for type in self.type_color_mapping:
                self.type_color_dict[type] = self.type_color_mapping[type]

        if (self.target_color_mapping != None) and (self.target_color_mapping != ''):
            self.type_color_dict['target'] = self.target_color_mapping

    def _select_max_min_hetesim_range(self, df_list, df_column, min_hetesim, max_hetesim):

        adjusted_df_list = []

        for df in df_list:
            df = df[df[df_column] >= min_hetesim]
            df = df[df[df_column] <= max_hetesim]
            adjusted_df_list.append(df)

        return adjusted_df_list

    def _select_max_nodes(self, df_list, max_node_count):

        adjusted_df_list = []

        for df in df_list:
            unique_max_count_sns = df['sn'].unique()[:max_node_count]
            df = df[df['sn'].isin(unique_max_count_sns)]
            adjusted_df_list.append(df)

        return adjusted_df_list

    def _select_specific_targets(self, df_list, target_list):

        adjusted_df_list = []

        for df in df_list:
            adjusted_df_list.append(df[df['target'].isin(target_list)])

        return adjusted_df_list

    def _select_specific_types(self, df_list, type_list):

        adjusted_df_list = []

        for df in df_list:
            adjusted_df_list.append(df[df['type'].isin(type_list)])

        return adjusted_df_list

    def _select_specific_sources(self, df_list, source_list):

        adjusted_df_list = []

        for df in df_list:
            adjusted_df_list.append(df[df['sn_cui'].isin(source_list)])

        return adjusted_df_list

    def generate_table(self):

        combined_df_generating = pd.concat(self.adjusted_df_list)
        combined_df_generating = combined_df_generating.round({'hetesim': 3, 'mean_hetesim': 3})
        combined_df_generating = combined_df_generating[['target', 'sn_name', 'hetesim', 'mean_hetesim', 'type', 'sn_cui']]
        combined_df_generating = combined_df_generating.replace({"target": self.target_cui_target_name_dict})

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

'''

source_node	source_name	target_node	target_name	hetesim_score
0	C0021641	Insulin	C0020676	Hypothyroidism	0.075061
1	C0028754	Obesity	C0020676	Hypothyroidism	0.067793
2	C0020538	Hypertensive disease	C0020676	Hypothyroidism	0.067543
3	C0011847	Diabetes	C0020676	Hypothyroidism	0.064721
4	C0021665	Insulin-Like Growth Factor I	C0020676	Hypothyroidism	0.060495
...	...	...	...	...	...
381	C0002886	Anemia, Macrocytic	C0020676	Hypothyroidism	0.002914
382	C0007810	Cerebrospinal Fluid Proteins	C0020676	Hypothyroidism	0.002080
383	C1281300	Vascular degeneration	C0020676	Hypothyroidism	0.001762
384	C0277552	Sporadic disorder, NOS	C0020676	Hypothyroidism	0.001733
385	C0154271	Hypercarotinemia	C0020676	Hypothyroidism	0.000939
386 rows × 5 columns
'''

'''
'''