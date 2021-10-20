# Imports
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import ipywidgets as widgets
import time
from plotly.offline import init_notebook_mode, plot
init_notebook_mode()

# Compute surface data
x = np.linspace(-10,10,201)
y = np.linspace(-10,10,201)
X,Y = np.meshgrid(x,y)

z_bath = X+Y
z_tnm = np.zeros((10,201,201),dtype=float)

for t in range(10):
    tsunami = t*np.sin(X+t/10*np.pi)
    z_tnm[t] = tsunami

# Create FigureWidget and add surface trace
#fig = go.Figure()
#surface = fig.add_surface(z=z_bath,name='bath')

# ----- change part ------ 
bath = go.Surface(z=z_bath)
fig = go.Figure(data=[bath,bath])

# Set axis ranges to fixed values to keep them from retting during animation
fig.layout.scene.zaxis.range = [-10, 10]
fig.layout.scene.yaxis.range = [0, 150]
fig.layout.scene.xaxis.range = [0, 150]
#surface.cmin = -10
#surface.cmax = 10

frames = []
for i in range(10):
    frames.append(go.Frame(data=[{'type': 'surface', 'z': z_tnm[i], 'name':'tsunami'}]))

fig.frames = frames

fig.layout.updatemenus = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.55,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

plot(fig)


