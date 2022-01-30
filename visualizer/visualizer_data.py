import pandas as pd
import json
import os

def generate_formatted_dict(semnet_df_list=list, export=False, filename='semnet_json_data'):
    formatted_dict = {}

    for df in semnet_df_list:
        for _, row in df.iterrows():
            if row['source_node'] in formatted_dict:
                formatted_dict[row['source_node']]['targets'][row['target_node']] = {'name': row['target_name'], 'hetesim_score': row['hetesim_score']}
            
            else:
                formatted_dict[row['source_node']] = {}
                formatted_dict[row['source_node']]['source_name'] = row['source_name']
                formatted_dict[row['source_node']]['source_type'] = row['source_type']
                formatted_dict[row['source_node']]['targets'] = {}
                formatted_dict[row['source_node']]['targets'][row['target_node']] = {'name': row['target_name'], 'hetesim_score': row['hetesim_score']}

    if export:
        with open(filename + '.json', 'w') as outfile:
            json.dump(formatted_dict, outfile)

    return formatted_dict

def visualizer_json_from_dicts(semnet_dict_list=list, export=False, filename='semnet_combined_json_data'):
    final_dict = {}

    for i, semnet_dict in enumerate(semnet_dict_list):
        final_dict['semnet_run_' + str(i)] = semnet_dict

    if export:
        with open(filename + '.json', 'w') as outfile:
            json.dump(final_dict, outfile)

    return final_dict

def generate_formatted_dict_from_path(path=str, export=False, filename='semnet_json_data'):
    df_list = []

    for name in os.listdir(path):
        if 'csv' in name:
            df_list.append(pd.read_csv(os.path.join(path, name), index_col=0))

    final_dict = generate_formatted_dict(df_list)

    if export:
        with open(filename + '.json', 'w') as outfile:
            json.dump(final_dict, outfile)

    return final_dict

def visualizer_json_from_path(path=str, export=False, filename='semnet_combined_json_data'):
    dict_list = []
    final_output_dict = {}

    for root, _, _ in os.walk(path, topdown=True):
        generated_dict = generate_formatted_dict_from_path(root)
        if len(generated_dict) !=0:
            dict_list.append(generated_dict)

    for i, semnet_dict in enumerate(dict_list):
        final_output_dict['semnet_run_' + str(i)] = semnet_dict

    if export:
        with open(filename + '.json', 'w') as outfile:
            json.dump(final_output_dict, outfile)

    return final_output_dict