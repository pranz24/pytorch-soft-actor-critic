import plotly
from plotly.graph_objs import Scatter, Line
import torch


steps = []
def plot_line(xs, ys_population, args):
    steps.append(xs)
    if args.deterministic == True and args.value_update > 1:
        colour = 'rgb(172, 12, 0)'
    elif args.deterministic == False and args.value_update <= 1:
        colour = 'rgb(0, 172, 237)'
    else:
        colour = 'rgb(0, 172, 12)'

    ys = torch.Tensor(ys_population)

    ys = ys.squeeze()

    trace = Scatter(x=steps, y=ys.numpy(), line=Line(color=colour), name='Reward')

    if args.deterministic == True and args.value_update > 1:
        plotly.offline.plot({
            'data': [trace],
            'layout': dict(title='SAC(Deterministic, Hard Update)',
                    xaxis={'title': 'Steps'},
                    yaxis={'title': 'Reward'})
        }, filename='SAC(Deterministic).html', auto_open=False)
    elif args.deterministic == False and args.value_update <= 1:
        plotly.offline.plot({
            'data': [trace],
            'layout': dict(title='SAC',
                           xaxis={'title': 'Steps'},
                           yaxis={'title': 'Reward'})
        }, filename='SAC.html', auto_open=False)
    else:
        plotly.offline.plot({
            'data': [trace],
            'layout': dict(title='SAC(Hard Update)',
                           xaxis={'title': 'Steps'},
                           yaxis={'title': 'Reward'})
        }, filename='SAC(Hard Update).html', auto_open=False)
