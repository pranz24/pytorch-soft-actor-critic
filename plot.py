import plotly
from plotly.graph_objs import Scatter, Line
import torch


steps = []
def plot_line(xs, ys_population, algo):
    steps.append(xs)
    if algo == "SAC":
        colour = 'rgb(0, 172, 237)'
    elif algo == "SAC(GMM)":
        colour = 'rgb(0, 172, 12)'
    else:
        colour = 'rgb(172, 12, 0)'

    ys = torch.Tensor(ys_population)

    ys = ys.squeeze()

    trace = Scatter(x=steps, y=ys.numpy(), line=Line(color=colour), name='Reward')

    if algo == "SAC(GMM)":
        plotly.offline.plot({
            'data': [trace],
            'layout': dict(title='SAC(GMM)',
                    xaxis={'title': 'Steps'},
                    yaxis={'title': 'Reward'})
        }, filename='SAC(GMM).html', auto_open=False)
    elif algo == "SAC":
        plotly.offline.plot({
            'data': [trace],
            'layout': dict(title='SAC',
                           xaxis={'title': 'Steps'},
                           yaxis={'title': 'Reward'})
        }, filename='SAC.html', auto_open=False)
    else:
        plotly.offline.plot({
            'data': [trace],
            'layout': dict(title=algo,
                           xaxis={'title': 'Steps'},
                           yaxis={'title': 'Reward'})
        }, filename='{}.html'.format(algo), auto_open=False)


