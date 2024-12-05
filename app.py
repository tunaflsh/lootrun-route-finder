from dash import Dash, html, dcc, dash_table, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from pandas.errors import UndefinedVariableError
from tokenize import TokenError
from PIL import Image

from msgspec import msgpack
from itertools import pairwise
from time import perf_counter
import os, io, psutil

from lootrun_route_finder import *


t = perf_counter()


process = psutil.Process()

def memory_usage():
    size = process.memory_info().rss
    for prefix in [''] + list(np.array(list('KMGTPEZ')) + 'i'):
        if size < 1024:
            print(f'Memory usage: {size:.1f}{prefix}B')
            return
        size /= 1024
    print(f'Memory usage: {size:.1f}YiB')


def enc_hook(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise NotImplementedError(f'Encoding objects of type {type(obj)} is unsupported')

encoder = msgpack.Encoder(enc_hook=enc_hook)
decoder = msgpack.Decoder()


wp = pd.read_csv('assets/waypoints.csv', skipinitialspace=True)
cf = '.cache/bps{bps}ft{fast_travel}sk{slash_kill}cy{cycle}.msgspec'
wg = WaypointGraph(wp, cache_file_template=cf)
fig = go.Figure(layout_template='plotly_dark',
                layout_margin=dict(l=20, r=20, t=20, b=20),
                layout_dragmode='pan',
                layout_legend=dict(x=0.01, y=0.01))
img = Image.open('assets/TopographicMap.png')
fig.add_layout_image(name='map', source=img, layer='below',
                     xref='x', yref='y', x=-2392, y=-6607,
                     sizex=img.size[0], sizey=img.size[1],
                     opacity=0.5)
fig.update_yaxes(scaleanchor='x', autorange='reversed')
fig.add_traces([
    go.Scatter(mode='lines+markers', showlegend=True,
               hovertext='', hoverinfo='text', name=name,
               line=go.scatter.Line(color=color, dash=dash),
               marker=go.scatter.Marker(symbol='arrow', angleref='previous',
                                        opacity=1))
    for name, color, dash in zip(
        ['flight', 'fast travel', '/kill', 'teleport scroll'],
        ['turquoise', 'limegreen', 'crimson', 'darkorange'],
        ['dot'] + ['solid'] * 3)
    ])
fig.add_trace(go.Scatter(
    mode='markers', text=wp['Name'],
    x=wp['X'], y=wp['Z'],
    customdata=wp['Y'], showlegend=False,
    hovertemplate='%{text}<br>%{x} %{customdata} %{y}<extra></extra>',
    marker=go.scatter.Marker(color='orange')))


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.layout = dbc.Row([
    dcc.Store(id='figure'),

    dbc.Col(dcc.Graph(figure=fig, id='graph', className='wynncraft-map',
                      config={'scrollZoom': True, 'displayModeBar': False})),
    dbc.Col(
        html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText('query'),
                        dbc.Input(type='text', id='query-input',
                                  placeholder='e.g. "72 <= Level <= 80 and Cave"'),
                        ]),
                    ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText('start at'),
                        dbc.Input(type='text', id='start-at-input',
                                  placeholder='Name',
                                  list='suggested-names'),
                        html.Datalist(id='suggested-names'),
                        ]),
                    width=5,
                    ),
                ],
                align='center',
                class_name='g-1',
                ),
            dbc.Row([
                dbc.Col(
                    dbc.Checklist(
                        options=[
                            {'label': 'fast travel', 'value': wg.FAST_TRAVEL},
                            {'label': '/kill', 'value': wg.SLASH_KILL},
                            {'label': 'cycle', 'value': 'cycle'},
                            ],
                        value=[wg.FAST_TRAVEL, wg.SLASH_KILL],
                        id='parameter-checklist',
                        inline=True,
                        switch=True,
                        ),
                    width='auto',
                    ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText('scrolls'),
                        dbc.Input(type='number', min=0, max=3, step=1,
                                  value=3,
                                  id='scrolls-input'),
                        ],
                        style={'width': 125},
                        ),
                    width='auto',
                    ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText('bps'),
                        dbc.Input(type='number', min=0,
                                  value=18,
                                  id='bps-input'),
                        ]),
                    width=2,
                    ),
                dbc.Col(
                    html.Div(
                        dbc.Button('Find Route',
                                   id='find-route-button',
                                   color='primary',
                                   class_name='me-1'),
                        className='d-grid gap-2'),
                    ),
                ],
                align='center',
                class_name='g-1',
                ),
            html.Div(
                dbc.Table.from_dataframe(wp, striped=True, bordered=True,
                                         hover=True, color='dark'),
                id='table-container',
                className='table-wrapper'),
            ],
            className='d-grid gap-2')
        )
    ],
    class_name='g-1',
    style={'width': '100vw'})


@callback(
        Output('table-container', 'children'),
        Output('query-input', 'invalid'),
        Input('query-input', 'value'),
        State('table-container', 'children'))
def update_query(query, table):
    try:
        queried_wp = wp.query(query)
    except ValueError:
        queried_wp = wp
    except (SyntaxError, KeyError, NotImplementedError, TokenError,
            UndefinedVariableError):
        return table, True
    if isinstance(queried_wp, pd.Series):
        queried_wp = queried_wp.to_frame().T
    return (dbc.Table.from_dataframe(queried_wp, striped=True, bordered=True,
                                    hover=True, color='dark'),
            False)


@callback(
        Output('suggested-names', 'children'),
        Output('start-at-input', 'invalid'),
        Input('start-at-input', 'value'),
        prevent_initial_call=True)
def suggest_name(name):
    if not name:
        return [], False
    suggestions = [html.Option(value=n) for n in wp['Name'] if n.startswith(name)]
    if 1 == len(suggestions) and name == suggestions[0].value:
        return [], False
    return suggestions, len(suggestions) == 0


@callback(
        Output('find-route-button', 'disabled'),
        Input('parameter-checklist', 'value'),
        Input('scrolls-input', 'value'),
        Input('bps-input', 'value'),
        runnint=[(Output('find-route-button', 'disabled'), True, False)])
def update_parameters(toggles, scrolls, bps):
    global wg
    ft = wg.FAST_TRAVEL in toggles
    sk = wg.SLASH_KILL in toggles
    cy = 'cycle' in toggles
    bps = bps
    scrolls = scrolls
    wg.update(fast_travel=ft, slash_kill=sk, cycle=cy, bps=bps, scrolls=scrolls)
    return False


@callback(
        output=Output('graph', 'figure'),
        inputs=[
            Input('find-route-button', 'n_clicks'),
            (State('query-input', 'value'), State('query-input', 'invalid')),
            (State('start-at-input', 'value'), State('start-at-input', 'invalid'))],
        prevent_initial_call=True,
        running=[
            (Output('find-route-button', 'disabled'), True, False),
            (Output('find-route-button', 'children'), 'Computing Route', 'Find Route')])
def find_route(_, query, start):
    query, invalid = query
    if invalid:
        raise PreventUpdate
    query = wp.query(query).index

    start, invalid = start
    if invalid:
        raise PreventUpdate
    start = wp.query('Name == @start').index[0] if start else None

    order, distance = wg.find_route_between(query, start)
    lines = {tt: {'x': [], 'y': [], 'marker.size': [], 'hovertext': []}
             for tt in [wg.FLIGHT, wg.FAST_TRAVEL, wg.SLASH_KILL, wg.SCROLL, wg.BLOCKED]}

    for i, j in pairwise(order.iloc):
        length = ((i['X'] - j['X'])**2 + (i['Z'] - j['Z'])**2)**0.5

        mid_points = max(1, int(length / 200 - 1))
        mid = {ax: [i[ax] + n/(mid_points+1) * (j[ax] - i[ax])
                    for n in range(1, mid_points+1)]
               for ax in 'XZ'}

        size = 18 * (200 < length)

        if wg.FLIGHT == j['Travel']:
            hovertext = f'{wg.distance_matrix[i.name, j.name]:.0f}m'
        elif wg.FAST_TRAVEL == j['Travel'] or wg.SLASH_KILL == j['Travel']:
            hovertext = f'{wg.distance_matrix[i.name, j.name] / wg.bps:.0f}s'
        elif wg.SCROLL == j['Travel']:
            hovertext = f'{wg._SCROLL / wg.bps:.0f}s'

        lines[j['Travel']]['x'].extend([i['X'], *mid['X'], j['X'], None])
        lines[j['Travel']]['y'].extend([i['Z'], *mid['Z'], j['Z'], None])
        lines[j['Travel']]['marker.size'].extend(
                [0] * (mid_points + 1) + [size, 0])
        lines[j['Travel']]['hovertext'].extend(
                [''] + [hovertext] * mid_points + ['', ''])

    for trace, line in zip(fig.data[:4], lines.values()):
        trace.x = line['x']
        trace.y = line['y']
        trace.marker.size = line['marker.size']
        trace.hovertext = line['hovertext']

    memory_usage()

    return fig


print(f'Initializing app took {perf_counter() - t:.3f}s')
memory_usage()
t = perf_counter()


if __name__ == '__main__':
    app.run(debug=False, jupyter_mode='external')
