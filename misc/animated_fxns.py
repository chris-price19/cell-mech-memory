#!/usr/bin/python

import plotly.graph_objects as go
import pandas as pd
import numpy as np
# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
z = z_data.values
sh_0, sh_1 = z.shape
x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

print('here')
fig.show()

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



def sample_surface_animate():

    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from plotly.offline import init_notebook_mode, plot

    init_notebook_mode()

    # stiff genes
    res = 300
    m0 = 1.
    tau = 1.
    n = 3
    z
    choose_m = 0.45

    x_space = np.linspace(0, 2.5, res)
    a_space = np.linspace(0.2, 2.2, res)
    x_stiff = 1-np.exp(-choose_m/m0)

    U_data = np.zeros((len(x_space), len(a_space)))

    for ai, aa in enumerate(a_space):
        for xi, xx in enumerate(x_space):
            U_data[xi, ai] = U(x_stiff, xx, aa, tau, n)


    print(np.shape(np.abs(np.diff(U_data, axis=0))))
            
    # print(x_space[argrelextrema(U_data[:,-1], np.less)[0]])
    # print(argrelextrema(np.abs(np.diff(U_data, axis=0)), np.less, axis=0))
    # print(np.diff(U_data[:,-1], n=2))
    # plt.plot(x_space[2:], np.diff(U_data[:,-50], n=2))
    m1coords = []
    m2coords = []
    for ai, aa in enumerate(a_space):
        xargs = argrelextrema(np.abs(np.diff(U_data[:,ai])), np.less)[0]
    #     xargs = argrelextrema(np.diff(U_data[:,-1], n=2), np.greater)[0]
        x_mins = x_space[xargs]
        if len(x_mins) > 1:
            m1coords.append([np.amin(x_mins),aa, U_data[np.amin(xargs), ai]])
            m2coords.append([np.amax(x_mins),aa, U_data[np.amax(xargs), ai]])
        else:
            m1coords.append([np.amin(x_mins),aa, U_data[np.amin(xargs), ai]])
            
    m1coords = np.array(m1coords)
    m2coords = np.array(m2coords)
    print(m1coords.shape); print(m2coords.shape)

    fig = go.Figure(data=[go.Surface(z=U_data.T, x=x_space, y=a_space, colorscale='blackbody')],
                  )

    fig.update_traces(contours_z=dict(show=True, usecolormap=False,
                                      project_z=True, start=np.amin(U_data), 
                                      end=np.amax(U_data), color='black', size=0.05))

    fig.add_scatter3d(x=m1coords[:,0], y=m1coords[:,1], z=m1coords[:,2]+0.02, mode='markers', showlegend=False,
                      marker=dict(size=4, color='blueviolet'))
    fig.add_scatter3d(x=m2coords[:,0], y=m2coords[:,1], z=m2coords[:,2]+0.02, mode='markers', showlegend=False,
                      marker=dict(size=4, color='dodgerblue'))

    fig.update_layout(title='Waddington Landscape', autosize=True, scene=dict(
                        xaxis = dict(
                            title='<b>x</b>', range=[np.amin(x_space), np.amax(x_space)], titlefont=dict(family='Cambria', size=22)),
                        yaxis = dict(
                            title='<b>\u03b1</b>', range=[np.amin(a_space), np.amax(a_space)], titlefont=dict(family='Cambria', size=22)),
                        zaxis = dict(
                            title='U', range=[np.amin(U_data)*1.05, 1.2], titlefont=dict(family='Cambria', size=10))),
                      width=650, height=650,
                      margin=dict(l=65, r=50, b=65, t=90),
                      font=dict(family='Cambria', size=16, color='#7f7f7f'),
                      
                     )

    skip = 10 
    fig.frames = [go.Frame(
                            data=[go.Scatter3d(
                            x=[m1coords[int(skip*k),0]],
                            y=[m1coords[int(skip*k),1]],
                            z=[m1coords[int(skip*k),2]+0.1],                            
                            mode="markers",
                            marker=dict(color="lime", size=38))])
                        for k in range(int(len(m1coords)/skip))]

    fig.layout.updatemenus = [dict(type="buttons", 
                                   buttons=[dict(label="Play", method="animate", args=[None])])]
        

    fig.show()
    # plot(fig)

    return
