from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
from itertools import pairwise
from time import perf_counter
import psutil

from lootrun_route_finder import *


process = psutil.Process()

def memory_usage():
    size = process.memory_info().rss
    for prefix in [''] + list(np.array(list('KMGTPEZ')) + 'i'):
        if size < 1024:
            print(f'{size:.1f}{prefix}B')
            return
        size /= 1024
    print(f'{size:.1f}YiB')


wp = pd.read_csv('assets/waypoints.csv', skipinitialspace=True)
wg = WaypointGraph(wp)

fig = go.Figure(layout=go.Layout(template='plotly_dark',
                                 margin=dict(l=20, r=20, t=20, b=20)))
img = Image.open('assets/TopographicMap.png')
fig.update_yaxes(scaleanchor='x', autorange='reversed')

fig.add_traces([
    go.Scatter(mode='lines+markers', hoverinfo='skip', showlegend=True,
               name=name, line=go.scatter.Line(color=color, dash=dash),
               marker=go.scatter.Marker(symbol='arrow', size=15,
                                        angleref='previous'),)
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
fig.add_layout_image(name='map', source=img, layer='below',
                     xref='x', yref='y', x=-2392, y=-6607,
                     sizex=img.size[0], sizey=img.size[1],
                     opacity=0.5)

app = Dash('Lootrun Route Finder')
app.layout = [
        html.Div(className='row', children=[
            html.Div(className='six columns', children=[
                dcc.Graph(figure=fig, id='graph',
                          style={'height': '97vh'})
                ])
            ])
        ]

bps = 18
ft = True
sk = True
sc = 3
cy = False
cf = '.cache/bps{bps}ft{fast_travel}sk{slash_kill}cy{cycle}.msgspec'
wg = WaypointGraph(wp, bps=bps, fast_travel=ft, slash_kill=sk,
                   scrolls=sc, cycle=cy, cache_file_format=cf)

order, distance = wg.find_route_between((72 <= wp.Level) & (wp.Level <= 80) & (wp.Cave == 1))
lines = {wg.FLIGHT: [], wg.FAST_TRAVEL: [], wg.SLASH_KILL: [],
         wg.SCROLL: [], wg.BLOCKED: []}

for i, j in pairwise(order.iloc):
    lines[j['Travel']].extend([i[['X', 'Z']].to_numpy(),
                               j[['X', 'Z']].to_numpy(),
                               [np.nan, np.nan]])
lines = {k: np.stack(v).T if v else np.empty((2,0)) for k, v in lines.items()}

for trace, line in zip(fig.data[:4], lines.values()):
    trace.x = line[0]
    trace.y = line[1]

if __name__ == '__main__':
    app.run(debug=True, jupyter_mode='external')
