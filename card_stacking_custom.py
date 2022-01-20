'''
app.layout = html.Div([
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
    html.Div(children=None,
        id='dropdown_card_div',
        style={'marginLeft': '1vw', 'marginTop': '1vh', 'display': 'flex', 'flex-direction': 'column', 'width': '20vw', 'position': 'relative', 'zIndex': '22'}
    )
])

@app.callback(
    Output("settings_collapse", "is_open"),
    Output("settings_button", "style"),
    Input("settings_button", "n_clicks"),
    Input("settings_collapse", "is_open"),
)
def toggle_settings(settings_button_clicks, settings_collapse_is_open):
    if settings_button_clicks:
        ui_tracker.settings_button_toggle = not ui_tracker.settings_button_toggle

        if settings_collapse_is_open:
            return [
                not settings_collapse_is_open, 
                {'width': '10vw'}
                ]
        else:
            return [
                not settings_collapse_is_open, 
                {'width': '10vw', 'background': 'green'}
                ]

@app.callback(
    Output("dropdown_card_div", "children"),
    Output("graph_sliders_button", "style"),
    Output("node_filtering_button", "style"),
    Output("graph_spread_button", "style"),
    Output("color_editing_button", "style"),
    Output("node_data_button", "style"),
    Input("graph_sliders_button", "n_clicks"), 
    Input("node_filtering_button", "n_clicks"), 
    Input("graph_spread_button", "n_clicks"), 
    Input("color_editing_button", "n_clicks"), 
    Input("node_data_button", "n_clicks")
)
def toggle_buttons(
    graph_sliders_button_clicks, 
    node_filtering_button_clicks, 
    graph_spread_button_clicks, 
    color_editing_button_clicks, 
    node_data_button_clicks):

    if graph_sliders_button_clicks != None:
        if graph_sliders_button_clicks > ui_tracker.graph_sliders_button_clicks:
            ui_tracker.graph_sliders_button_clicks = ui_tracker.graph_sliders_button_clicks + 1
            ui_tracker.graph_sliders_button_toggle = not ui_tracker.graph_sliders_button_toggle

            if ui_tracker.graph_sliders_button_toggle:
                ui_tracker.card_stack_tracking.append('graph_sliders')
                ui_tracker.dropdown_cards.append(graph_sliders)

                ui_tracker.graph_sliders_button_style['background'] = 'orange'

            else:
                idx = ui_tracker.card_stack_tracking.index('graph_sliders')
                ui_tracker.card_stack_tracking.pop(idx)
                ui_tracker.dropdown_cards.pop(idx)

                ui_tracker.graph_sliders_button_style['background'] = '#3A3F44'

    if node_filtering_button_clicks != None:
        if node_filtering_button_clicks > ui_tracker.node_filtering_button_clicks:
            ui_tracker.node_filtering_button_clicks = ui_tracker.node_filtering_button_clicks + 1
            ui_tracker.node_filtering_button_toggle = not ui_tracker.node_filtering_button_toggle

            if ui_tracker.node_filtering_button_toggle:
                ui_tracker.card_stack_tracking.append('node_filtering')
                ui_tracker.dropdown_cards.append(node_filtering)

                ui_tracker.node_filtering_button_style['background'] = 'orange'

            else:
                idx = ui_tracker.card_stack_tracking.index('node_filtering')
                ui_tracker.card_stack_tracking.pop(idx)
                ui_tracker.dropdown_cards.pop(idx)

                ui_tracker.node_filtering_button_style['background'] = '#3A3F44'

    if graph_spread_button_clicks != None:
        if graph_spread_button_clicks > ui_tracker.graph_spread_button_clicks:
            ui_tracker.graph_spread_button_clicks = ui_tracker.graph_spread_button_clicks + 1
            ui_tracker.graph_spread_button_toggle = not ui_tracker.graph_spread_button_toggle

            if ui_tracker.graph_spread_button_toggle:
                ui_tracker.card_stack_tracking.append('graph_spread')
                ui_tracker.dropdown_cards.append(graph_spread)

                ui_tracker.graph_spread_button_style['background'] = 'orange'

            else:
                idx = ui_tracker.card_stack_tracking.index('graph_spread')
                ui_tracker.card_stack_tracking.pop(idx)
                ui_tracker.dropdown_cards.pop(idx)

                ui_tracker.graph_spread_button_style['background'] = '#3A3F44'

    if color_editing_button_clicks != None:
        if color_editing_button_clicks > ui_tracker.color_editing_button_clicks:
            ui_tracker.color_editing_button_clicks = ui_tracker.color_editing_button_clicks + 1
            ui_tracker.color_editing_button_toggle = not ui_tracker.color_editing_button_toggle

            if ui_tracker.color_editing_button_toggle:
                ui_tracker.card_stack_tracking.append('color_editing')
                ui_tracker.dropdown_cards.append(color_editing)

                ui_tracker.color_editing_button_style['background'] = 'orange'

            else:
                idx = ui_tracker.card_stack_tracking.index('color_editing')
                ui_tracker.card_stack_tracking.pop(idx)
                ui_tracker.dropdown_cards.pop(idx)

                ui_tracker.color_editing_button_style['background'] = '#3A3F44'

    if node_data_button_clicks != None:
        if node_data_button_clicks > ui_tracker.node_data_button_clicks:
            ui_tracker.node_data_button_clicks = ui_tracker.node_data_button_clicks + 1
            ui_tracker.node_data_button_toggle = not ui_tracker.node_data_button_toggle

            if ui_tracker.node_data_button_toggle:
                ui_tracker.card_stack_tracking.append('node_data')
                ui_tracker.dropdown_cards.append(node_data)

                ui_tracker.node_data_button_style['background'] = 'orange'

            else:
                idx = ui_tracker.card_stack_tracking.index('node_data')
                ui_tracker.card_stack_tracking.pop(idx)
                ui_tracker.dropdown_cards.pop(idx)

                ui_tracker.node_data_button_style['background'] = '#3A3F44'

    return [
        ui_tracker.dropdown_cards, 
        ui_tracker.graph_sliders_button_style,
        ui_tracker.node_filtering_button_style, 
        ui_tracker.graph_spread_button_style, 
        ui_tracker.color_editing_button_style, 
        ui_tracker.node_data_button_style
    ]
'''