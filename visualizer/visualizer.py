import dash
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from dash import html
from dash import dash_table as dt
from dash import dcc

import base64
import json

from generate_graph import Generate_Graph

graph = Generate_Graph()

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])

class UI_Tracker:
    def __init__(self):
        self.set_initial_state()

    def set_initial_state(self):
        self.card_stack_tracking = []
        self.dropdown_cards = []

        self.settings_button_toggle = False
        self.settings_button_class = 'button_disabled'
        self.settings_button_text = 'Expand Settings'

        self.graph_sliders_button_toggle = False
        self.graph_sliders_button_class = 'button_disabled'

        self.node_filtering_button_toggle = False
        self.node_filtering_button_class = 'button_disabled'

        self.graph_spread_button_toggle = False
        self.graph_spread_button_class = 'button_disabled'

        self.color_editing_button_toggle = False
        self.color_editing_button_class = 'button_disabled'

        self.node_data_button_toggle = False
        self.node_data_button_class = 'button_disabled'

        self.table_data_button_toggle = False
        self.table_data_button_class = 'button_disabled'

        self.reset_button_clicks = None
        self.randomized_color_button_clicks = None

        self.display_gradient_start_color = graph.gradient_start_initial
        self.display_gradient_end_color = graph.gradient_end_initial
        self.display_selected_type_color = graph.selected_type_color_initial
        self.display_target_color = graph.target_color_initial

ui_tracker = UI_Tracker()

graph_sliders = dbc.Card(
    dbc.CardBody(
        [
            html.H6('Node HeteSim:', className='card-text', style={'marginBottom': '3%'}),
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

            html.H6('Edge HeteSim:', className='card-text', style={'marginBottom': '3%'}),

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

            html.H6('Max Node Count:', className='card-text', style={'marginBottom': '3%'}),

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
    )
)

node_filtering = dbc.Card(
    dbc.CardBody(
        [
            html.Div(children=[
                html.H6('Select Target Node CUIs:', className='card-text'),
                dcc.Dropdown(
                    id='specific_target_dropdown', 
                    options=graph.specific_target_input_options,
                    value=[],
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Select Source Node CUIs:', className='card-text'),
                dcc.Dropdown(
                    id='specific_source_dropdown', 
                    options=graph.specific_source_input_options,
                    value=[],
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Select Source Node Types:', className='card-text'),
                dcc.Dropdown(
                    id='specific_type_dropdown', 
                    options=graph.specific_type_input_options,
                    value=[],
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            )          
        ]
    )
)

graph_spread = dbc.Card(
    dbc.CardBody(
        [
            html.H6('Target Node Spread:', className='card-text', style={'marginBottom': '3%'}),
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

            html.H6('Source Node Spread:', className='card-text', style={'marginBottom': '3%'}),
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
    )
)

color_editing = dbc.Card(
    dbc.CardBody(
        [
            html.Div(children=[
                html.H6('Type Gradient:', style={'marginRight': '5%'}, className='card-text'
                ),
                dbc.Input(
                    type='color',
                    id='gradient_start',
                    value=graph.gradient_start_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'marginRight': '1%', 'border': 'none', 'padding': '0'}
                ),
                html.Div(',', style={'display': 'inline-block', 'marginRight': '1%', 'marginLeft': '1%'}),
                dbc.Input(
                    type='color',
                    id='gradient_end',
                    value=graph.gradient_end_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'marginLeft': '1%', 'border': 'none', 'padding': '0'},
                )               
                ], 
                style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Selected Source Node Type Color:', style={'marginRight': '5%'}, className='card-text'),
                dbc.Input(
                    type='color',
                    id='selected_type_color',
                    value=graph.selected_type_color_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'border': 'none', 'padding': '0'}
                )           
                ], 
                style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Target Node Color:', style={'marginRight': '5%'}, className='card-text'),
                dbc.Input(
                    type='color',
                    id='target_color',
                    value=graph.target_color_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'border': 'none', 'padding': '0'}
                )           
                ], 
                style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
            ), 

            html.Div(children=[
                dbc.Button(
                    'Randomize Colors', 
                    id='randomize_colors_button', 
                    className='btn shadow-none'
                )
                ],
                style={'marginBottom': '4%'}, 
                className='d-grid gap-2 col-6 mx-auto'
            )
        ]
    ),
)

node_data = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                html.Div(
                    children='Select Node(s)',
                    id='node_data'
                ), 
                style={'max-height': '400px', 'overflowX': 'scroll', 'overflowY': 'scroll'}
            )
        ]
    )
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
                    style_table={'overflowX': 'auto', 'overflowY': 'auto', 'max-height': '400px'},
                    style_data_conditional=[                
                        {
                            'if': {'state': 'selected'},
                            'backgroundColor': '#8ecae6',
                            'border': '#FFFFFF',
                            'color': '#000000'
                        }
                    ],
                    css=[
                        { 'selector': '.current-page', 'rule': 'visibility: hidden;'}, 
                        { 'selector': '.current-page-shadow', 'rule': 'color: #AAAAAA; font-size: 16px;'}, 
                        { 'selector': '.next-page:hover', 'rule': 'color: #8ecae6;'}, 
                        { 'selector': '.last-page:hover', 'rule': 'color: #8ecae6;'}, 
                        { 'selector': '.previous-page:hover', 'rule': 'color: #8ecae6;'}, 
                        { 'selector': '.first-page:hover', 'rule': 'color: #8ecae6;'}
                    ],
                    export_format='csv',
                    page_size=50
                )
            )
        ]
    )
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

def parse_contents(json_input, filenames):

    formatted_json = {}

    for i, data in enumerate(json_input):
        _, content_string = data.split(',')

        decoded = base64.b64decode(content_string)

        try:
            formatted_json = json.loads(decoded)

        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

    return formatted_json

def server_layout():
    ui_tracker.set_initial_state()

    app_layout = html.Div([
        html.Div(cytoscape_graph, style={'position': 'fixed', 'zIndex': '1', 'width': '99vw', 'height': '99vh'}),
        html.Div(children=[
            html.Div(children=[
                dbc.Button(children=[
                    'Expand Settings'
                    ],
                    id='settings_button',
                    className=ui_tracker.settings_button_class,
                    style={'width': '50%'}
                ), 
                dbc.Button(children=[
                    'Reset Graph'
                    ],
                    id='reset_button',
                    className='reset_button', 
                    style={'width': '50%', 'background': 'linear-gradient(#8a0000, #af0000 40%, #8a0000'}
                ) 
                ], 
                style={'width': '20vw', 'background': '#272B30'}
            ),
            dbc.Collapse(children=[
                dbc.Button(
                    'Graph Sliders',
                    id='graph_sliders_button',
                    className=ui_tracker.graph_sliders_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    graph_sliders,
                    id='graph_sliders_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Node Filtering',
                    id='node_filtering_button',
                    className=ui_tracker.node_filtering_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    node_filtering,
                    id='node_filtering_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Graph Spread',
                    id='graph_spread_button',
                    className=ui_tracker.graph_spread_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    graph_spread,
                    id='graph_spread_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Color Editing',
                    id='color_editing_button',
                    className=ui_tracker.color_editing_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    color_editing,
                    id='color_editing_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Node Data',
                    id='node_data_button',
                    className=ui_tracker.node_data_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    node_data,
                    id='node_data_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Table Data',
                    id='table_data_button',
                    className=ui_tracker.table_data_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    table_data,
                    id='table_data_collapse',
                    is_open=False
                ), 
                html.Div(
                    dcc.Upload(
                        dbc.Button(
                            'Upload Data', 
                            style={'width': '20vw', 'background': 'linear-gradient(#008a07, #00af0e 40%, #008a07)'}
                        ),
                        id='data_upload', 
                        multiple=True
                    )
                )
                ],
                id='settings_collapse',
                is_open=False, 
                style={'width': '20vw'}
            )
            ], 
            style={
                'display': 'flex', 'flex-direction': 'column', 'marginLeft': '1vw', 
                'marginTop': '1vh', 'width': 'fit-content', 'padding': '0px', 'background': '#32383E', 'border': 'none', 
                'position': 'relative', 'zIndex': '22'}
        )
    ])
    
    return app_layout

app.layout = server_layout

@app.callback(
    Output('settings_collapse', 'is_open'),
    Output('settings_button', 'className'),
    Output('settings_button', 'children'),
    Input('settings_button', 'n_clicks')
)
def toggle_settings(settings_button_clicks):
    if settings_button_clicks:
        ui_tracker.settings_button_toggle = not ui_tracker.settings_button_toggle

        if ui_tracker.settings_button_toggle:
            ui_tracker.settings_button_class = 'button_enabled'
            ui_tracker.settings_button_text = 'Collapse Settings'

        else:
            ui_tracker.settings_button_class = 'button_disabled'
            ui_tracker.settings_button_text = 'Expand Settings'
    
    return [ui_tracker.settings_button_toggle, ui_tracker.settings_button_class, ui_tracker.settings_button_text]

@app.callback(
    Output('graph_sliders_collapse', 'is_open'),
    Output('graph_sliders_button', 'className'),
    Input('graph_sliders_button', 'n_clicks')
)
def toggle_left(graph_sliders_button_clicks):
    if graph_sliders_button_clicks:
        ui_tracker.graph_sliders_button_toggle = not ui_tracker.graph_sliders_button_toggle

        if ui_tracker.graph_sliders_button_toggle:
            ui_tracker.graph_sliders_button_class = 'button_enabled'

        else:
            ui_tracker.graph_sliders_button_class = 'button_disabled'
    
    return [ui_tracker.graph_sliders_button_toggle, ui_tracker.graph_sliders_button_class]

@app.callback(
    Output('node_filtering_collapse', 'is_open'),
    Output('node_filtering_button', 'className'),
    Input('node_filtering_button', 'n_clicks')
)
def toggle_left(node_filtering_button_clicks):
    if node_filtering_button_clicks:
        ui_tracker.node_filtering_button_toggle = not ui_tracker.node_filtering_button_toggle

        if ui_tracker.node_filtering_button_toggle:
            ui_tracker.node_filtering_button_class = 'button_enabled'

        else:
            ui_tracker.node_filtering_button_class = 'button_disabled'

    return [ui_tracker.node_filtering_button_toggle, ui_tracker.node_filtering_button_class]

@app.callback(
    Output('graph_spread_collapse', 'is_open'),
    Output('graph_spread_button', 'className'),
    Input('graph_spread_button', 'n_clicks')
)
def toggle_left(graph_spread_button_clicks):
    if graph_spread_button_clicks:
        ui_tracker.graph_spread_button_toggle = not ui_tracker.graph_spread_button_toggle

        if ui_tracker.graph_spread_button_toggle:
            ui_tracker.graph_spread_button_class = 'button_enabled'

        else:
            ui_tracker.graph_spread_button_class = 'button_disabled'

    return [ui_tracker.graph_spread_button_toggle, ui_tracker.graph_spread_button_class]

@app.callback(
    Output('color_editing_collapse', 'is_open'),
    Output('color_editing_button', 'className'),
    Input('color_editing_button', 'n_clicks')
)
def toggle_left(color_editing_button_clicks):
    if color_editing_button_clicks:
        ui_tracker.color_editing_button_toggle = not ui_tracker.color_editing_button_toggle

        if ui_tracker.color_editing_button_toggle:
            ui_tracker.color_editing_button_class = 'button_enabled'

        else:
            ui_tracker.color_editing_button_class = 'button_disabled'

    return [ui_tracker.color_editing_button_toggle, ui_tracker.color_editing_button_class]

@app.callback(
    Output('node_data_collapse', 'is_open'),
    Output('node_data_button', 'className'),
    Input('node_data_button', 'n_clicks')
)
def toggle_left(node_data_button_clicks):
    if node_data_button_clicks:
        ui_tracker.node_data_button_toggle = not ui_tracker.node_data_button_toggle

        if ui_tracker.node_data_button_toggle:
            ui_tracker.node_data_button_class = 'button_enabled'

        else:
            ui_tracker.node_data_button_class = 'button_disabled'

    return [ui_tracker.node_data_button_toggle, ui_tracker.node_data_button_class]

@app.callback(
    Output('table_data_collapse', 'is_open'),
    Output('table_data_button', 'className'),
    Input('table_data_button', 'n_clicks')
)
def toggle_left(table_data_button_clicks):
    if table_data_button_clicks:
        ui_tracker.table_data_button_toggle = not ui_tracker.table_data_button_toggle

        if ui_tracker.table_data_button_toggle:
            ui_tracker.table_data_button_class = 'button_enabled'

        else:
            ui_tracker.table_data_button_class = 'button_disabled'

    return [ui_tracker.table_data_button_toggle, ui_tracker.table_data_button_class]

@app.callback(
    Output(component_id='output_graph', component_property='elements'),
    Output(component_id='node_hetesim_range_slider', component_property='min'),
    Output(component_id='node_hetesim_range_slider', component_property='max'),
    Output(component_id='node_hetesim_range_slider', component_property='step'), 
    Output(component_id='node_hetesim_range_slider', component_property='value'),
    Output(component_id='edge_hetesim_range_slider', component_property='min'),
    Output(component_id='edge_hetesim_range_slider', component_property='max'),
    Output(component_id='edge_hetesim_range_slider', component_property='step'),
    Output(component_id='edge_hetesim_range_slider', component_property='value'),
    Output(component_id='max_node_slider', component_property='max'), 
    Output(component_id='max_node_slider', component_property='value'), 
    Output(component_id='specific_target_dropdown', component_property='options'),
    Output(component_id='specific_target_dropdown', component_property='value'),
    Output(component_id='specific_source_dropdown', component_property='options'), 
    Output(component_id='specific_source_dropdown', component_property='value'),
    Output(component_id='specific_type_dropdown', component_property='options'), 
    Output(component_id='specific_type_dropdown', component_property='value'), 
    Output(component_id='target_spread_slider', component_property='value'),
    Output(component_id='source_spread_slider', component_property='value'),
    Output(component_id='gradient_start', component_property='value'), 
    Output(component_id='gradient_end', component_property='value'), 
    Output(component_id='selected_type_color', component_property='value'), 
    Output(component_id='target_color', component_property='value'),
    Output(component_id='data_table', component_property='columns'),
    Output(component_id='data_table', component_property='data'),
    Output(component_id='data_upload', component_property='contents'),
    Output(component_id='data_upload', component_property='filename'),
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
    Input(component_id='selected_type_color', component_property='value'), 
    Input(component_id='target_color', component_property='value'), 
    Input(component_id='randomize_colors_button', component_property='n_clicks'), 
    Input(component_id='reset_button', component_property='n_clicks'), 
    Input(component_id='data_upload', component_property='contents'),
    Input(component_id='data_upload', component_property='filename'),
    prevent_initial_call=True
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
    input_selected_type_color,
    input_target_color, 
    input_randomize_colors_button_clicks, 
    input_reset_button, 
    data_upload_content, 
    data_upload_filenames):

    if input_node_hetesim_range_slider != graph.node_hetesim_range:
        graph.node_hetesim_range = input_node_hetesim_range_slider

    if input_edge_hetesim_range_slider != graph.edge_hetesim_range:
        graph.edge_hetesim_range = input_edge_hetesim_range_slider

    if input_max_node_slider != graph.max_node_count:
        graph.max_node_count = input_max_node_slider

    if input_specific_target_dropdown != graph.specific_target_dropdown:
        graph.specific_target_dropdown = input_specific_target_dropdown

    if input_specific_source_dropdown != graph.specific_source_dropdown:
        graph.specific_source_dropdown = input_specific_source_dropdown

    if input_specific_type_dropdown != graph.specific_type_dropdown:
        graph.specific_type_dropdown = input_specific_type_dropdown

    if input_target_spread != graph.target_spread:
        graph.target_spread = input_target_spread

    if input_sn_spread != graph.sn_spread:
        graph.sn_spread = input_sn_spread

    graph.target_spread = input_target_spread
    graph.sn_spread = input_sn_spread

    if (input_gradient_start != graph.gradient_start) and (input_gradient_start != graph.gradient_start_initial):
        ui_tracker.display_gradient_start_color = input_gradient_start

        if input_gradient_end != graph.gradient_end_initial:
            graph.gradient_start = input_gradient_start
            graph.gradient_end = input_gradient_end
            graph.gradient_color_primacy = True
        
    if (input_gradient_end != graph.gradient_end) and (input_gradient_end != graph.gradient_end_initial):
        ui_tracker.display_gradient_end_color = input_gradient_end

        if input_gradient_start != graph.gradient_start_initial:
            graph.gradient_start = input_gradient_start
            graph.gradient_end = input_gradient_end
            graph.gradient_color_primacy = True

    if (input_selected_type_color != graph.selected_type_color) and (input_selected_type_color != graph.selected_type_color_initial):
        graph.selected_type_color = input_selected_type_color

        ui_tracker.display_selected_type_color = input_selected_type_color
        graph.type_color_primacy = True

    if (input_target_color != graph.target_color) and (input_target_color != graph.target_color_initial):
        graph.target_color = input_target_color

        ui_tracker.display_target_color = input_target_color
        graph.target_color_primacy = True

    if input_randomize_colors_button_clicks != ui_tracker.randomized_color_button_clicks:
        ui_tracker.randomized_color_button_clicks = input_randomize_colors_button_clicks

        ui_tracker.display_gradient_start_color = graph.gradient_start_initial
        ui_tracker.display_gradient_end_color = graph.gradient_end_initial
        ui_tracker.display_selected_type_color = graph.selected_type_color_initial
        ui_tracker.display_target_color = graph.target_color_initial
        
        graph.random_color_primacy = True

    if data_upload_content != None:
        json_user_input = parse_contents(data_upload_content, data_upload_filenames)
        elements = graph.load_additional_data(json_user_input)

        ui_tracker.display_gradient_start_color = graph.gradient_start_initial
        ui_tracker.display_gradient_end_color = graph.gradient_end_initial
        ui_tracker.display_selected_type_color = graph.selected_type_color_initial
        ui_tracker.display_target_color = graph.target_color_initial

    elif input_reset_button != ui_tracker.reset_button_clicks:
        ui_tracker.reset_button_clicks = input_reset_button
        graph.reset_graph()
        elements = graph.starting_elements

        ui_tracker.display_gradient_start_color = graph.gradient_start_initial
        ui_tracker.display_gradient_end_color = graph.gradient_end_initial
        ui_tracker.display_selected_type_color = graph.selected_type_color_initial
        ui_tracker.display_target_color = graph.target_color_initial

    else:
        elements = graph.update_graph_elements()

    return [
        elements,
        graph.node_hetesim_range_start,
        graph.node_hetesim_range_end,
        graph.node_hetesim_step_size, 
        graph.node_hetesim_range, 
        graph.edge_hetesim_range_start,
        graph.edge_hetesim_range_end,
        graph.edge_hetesim_step_size,
        graph.edge_hetesim_range, 
        graph.max_node_count_initial, 
        graph.max_node_count, 
        graph.specific_target_input_options,
        graph.specific_target_dropdown, 
        graph.specific_source_input_options, 
        graph.specific_source_dropdown, 
        graph.specific_type_input_options, 
        graph.specific_type_dropdown, 
        graph.target_spread, 
        graph.sn_spread,
        ui_tracker.display_gradient_start_color, 
        ui_tracker.display_gradient_end_color, 
        ui_tracker.display_selected_type_color,
        ui_tracker.display_target_color, 
        graph.data_table_columns, 
        graph.table_data, 
        None, 
        None
        ]

@app.callback(
    Output('node_data', 'children'),
    Input('output_graph', 'selectedNodeData'))
def displayTapNodeData(input_selected_nodes):

    if (input_selected_nodes == []) or (input_selected_nodes == None):
        return 'Select Node(s)'

    formatted_node_data = graph.generate_node_data(input_selected_nodes)

    display_data = []

    for i, node in enumerate(input_selected_nodes):
        display_data.append(html.Div(
            children=[
                node['label']
                ],
                style={'font-weight': 'bold'}
            )
        )

        display_data.append(html.Pre(formatted_node_data[i]))

    return display_data


if __name__ == '__main__':
    app.run_server(debug=False)