from generate_graph import Generate_Graph
import pandas as pd
import json

graph = Generate_Graph()

df_list = []

df_list.append(pd.read_csv('SemNet_results_target=C0037313.csv'))

df_list.append(pd.read_csv('SemNet_results_target=C0002395.csv'))

sub_json = graph.generate_formatted_json(df_list)

with open('json_data_2.json', 'w') as outfile:
    json.dump(sub_json, outfile)