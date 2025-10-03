from dash import (
        Dash, html, dcc, callback, Output, Input, State, ALL)
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
from pandas.errors import UndefinedVariableError
from tokenize import TokenError
from PIL import Image


wp = pd.read_csv('waypoints.csv', skipinitialspace=True)
# Corner coordinates:
# -a_ullr -2392 -6607 1699 -122
# Coordinate projection:
# -a_srs EPSG:3857
fig = 


if __name__ == '__main__':
    pass
