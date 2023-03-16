# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
cols = plotly.colors.DEFAULT_PLOTLY_COLORS
# %%
df = pd.read_csv('wandb_export_1_thresh_t.csv')
df
df = pd.read_csv('wandb_export_3_thresh_p.csv')
df

#kwargs["thresh_t"]:.03}-{kwargs["channel"]
list(df.columns)

# %% extract row names
#names = sorted(list(map(lambda x:x.split("_")[2].lower(),df['name'])))
#names = [names[i] for i in [1,0,3,2]]
#names
# %%
for channel in [None, 0, 1, 2, 3]:
    fig = go.Figure()
    for row in range(4):
        x = []
        y = []
        z = []
        for thresh in np.linspace(0,1,21)[1:-1]:
            x.append(df[f'recall_with_channel-{thresh:.03}-{channel}'].iloc[row])
            y.append(df[f'precision_with_channel-{thresh:.03}-{channel}'].iloc[row])
            z.append(df[f'f1score_with_channel-{thresh:.03}-{channel}'].iloc[row])
        draw = pd.DataFrame({'t':np.round(np.linspace(0,1,21),2)[1:-1],'x':x,'y':y,'z':z})
        name = df['name'].iloc[row].split("_")[2].lower()
        #fig = fig.add_trace(go.Line(x = draw['x'],y = draw['y'], name = name, text = draw['t']))
        fig = fig.add_trace(go.Line(x = draw['x'],y = draw['y'], name = name, text = draw['z']))

    #for f1 in np.round(np.linspace(0.2,0.7,8),8):
    # for f1 in [0.3,.57,0.62]:
    #     x = np.linspace(0.2,0.9,51)
    #     y = (f1 * x)/(2*x-f1)
    #     draw = pd.DataFrame({'x':x,'y':y})
    #     fig = fig.add_trace(go.Line(x = draw['x'],y = draw['y'],name=f'{f1}'))
    fig.update_layout(title=f'channel = {channel}',xaxis_title="recall",yaxis_title="precision",legend_title="Legend Title",font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))
    fig.update_layout(title={'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    fig.update_layout(autosize=False,width=1000,height=800,xaxis_range=[-0.1,1],yaxis_range=[-0.1,1])
    fig.show()


# %%
for channel in [None, 0, 1, 2, 3]:
    fig = go.Figure()
    for row in range(4):
        x = []
        y = []
        z = []
        for thresh in np.linspace(0,1,21)[1:-1]:
            x.append(df[f'recall_with_channel-{thresh:.03}-{channel}'].iloc[row])
            y.append(df[f'precision_with_channel-{thresh:.03}-{channel}'].iloc[row])
            z.append(df[f'f1score_with_channel-{thresh:.03}-{channel}'].iloc[row])
        draw = pd.DataFrame({'t':np.round(np.linspace(0,1,21),2)[1:-1],'x':x,'y':y,'z':z})
        name = df['name'].iloc[row].split("_")[2].lower()
        fig = fig.add_trace(go.Line(x = draw['t'],y = draw['z'], name = name))
    fig.update_layout(title=f'channel = {channel}',xaxis_title="threshold",yaxis_title="f1 score",legend_title="Legend Title",font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))
    fig.update_layout(title={'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    fig.update_layout(autosize=False,width=1100,height=500,)
    fig.show()

# %% calculated f1
for channel in [None, 0, 1, 2, 3]:
    fig = go.Figure()
    for row in range(4):
        x = []
        y = []
        z = []
        for thresh in np.linspace(0,1,21)[1:-1]:
            x.append(df[f'recall_with_channel-{thresh:.03}-{channel}'].iloc[row])
            y.append(df[f'precision_with_channel-{thresh:.03}-{channel}'].iloc[row])
            z.append(df[f'f1score_with_channel-{thresh:.03}-{channel}'].iloc[row])
        draw = pd.DataFrame({'t':np.round(np.linspace(0,1,21),2)[1:-1],'x':x,'y':y,'z':z})
        name = df['name'].iloc[row].split("_")[2].lower()
        f1 = list(map(lambda x,y:2*x*y/(x+y),draw['x'],draw['y']))
        fig = fig.add_trace(go.Line(x = draw['t'],y = f1, name = name))
    fig.update_layout(title=f'channel = {channel}',xaxis_title="threshold",yaxis_title="f1 score",legend_title="Legend Title",font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))
    fig.update_layout(title={'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    fig.update_layout(autosize=False,width=1100,height=500,)
    fig.show()
# %% drawn precision
for channel in [None, 0, 1, 2, 3]:
    fig = go.Figure()
    for row in range(4):
        x = []
        y = []
        z = []
        for thresh in np.linspace(0,1,21)[1:-1]:
            x.append(df[f'recall_with_channel-{thresh:.03}-{channel}'].iloc[row])
            y.append(df[f'precision_with_channel-{thresh:.03}-{channel}'].iloc[row])
            z.append(df[f'f1score_with_channel-{thresh:.03}-{channel}'].iloc[row])
        draw = pd.DataFrame({'t':np.round(np.linspace(0,1,21),2)[1:-1],'x':x,'y':y,'z':z})
        name = df['name'].iloc[row].split("_")[2].lower()
        fig = fig.add_trace(go.Line(x = draw['t'],y = draw['y'], name = name))
    fig.update_layout(title=f'channel = {channel}',xaxis_title="threshold",yaxis_title="f1 score",legend_title="Legend Title",font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))
    fig.update_layout(title={'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    fig.update_layout(autosize=False,width=1100,height=500,)
    fig.show()

# %% drawn recall
for channel in [None, 0, 1, 2, 3]:
    fig = go.Figure()
    for row in range(4):
        x = []
        y = []
        z = []
        for thresh in np.linspace(0,1,21)[1:-1]:
            x.append(df[f'recall_with_channel-{thresh:.03}-{channel}'].iloc[row])
            y.append(df[f'precision_with_channel-{thresh:.03}-{channel}'].iloc[row])
            z.append(df[f'f1score_with_channel-{thresh:.03}-{channel}'].iloc[row])
        draw = pd.DataFrame({'t':np.round(np.linspace(0,1,21),2)[1:-1],'x':x,'y':y,'z':z})
        name = df['name'].iloc[row].split("_")[2].lower()
        fig = fig.add_trace(go.Line(x = draw['t'],y = draw['x'], name = name))
    fig.update_layout(title=f'channel = {channel}',xaxis_title="threshold",yaxis_title="f1 score",legend_title="Legend Title",font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))
    fig.update_layout(title={'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    fig.update_layout(autosize=False,width=1100,height=500,)
    fig.show()


# %%
draw['f1'] = 2*draw['x']*draw['y']/(draw['x']+draw['y'])
draw


# %%
# %%
fig = make_subplots(rows=2, cols=2, subplot_titles=("leadtime = 15 min", "leadtime = 30 min", "leadtime = 45 min", "leadtime = 60 min"),shared_xaxes=True, shared_yaxes=True, horizontal_spacing = 0.05,vertical_spacing = 0.10)
for channel in [0, 1, 2, 3]:
    #fig = go.Figure()
    r,c = np.unravel_index(channel,(2,2))
    for row in range(3):# range(4) if persistance is needed
        x = []
        y = []
        z = []
        for thresh in np.linspace(0,1,21)[1:-1]:
            x.append(df[f'recall_with_channel-{thresh:.03}-{channel}'].iloc[row])
            y.append(df[f'precision_with_channel-{thresh:.03}-{channel}'].iloc[row])
            z.append(df[f'f1score_with_channel-{thresh:.03}-{channel}'].iloc[row])
        draw = pd.DataFrame({'t':np.round(np.linspace(0,1,21),2)[1:-1],'x':x,'y':y,'z':z})
        name = df['name'].iloc[row].split("_")[2].lower()
        fig = fig.add_trace(go.Line(x = draw['t'],y = draw['z'], name = name, line=dict(width=2, color=cols[row]), legendgroup = f'{row}',showlegend = True if channel == 0 else False), row = r+1, col = c+1)

    fig.update_xaxes(title_text="threshold", row=r+1, col=c+1, range=[0,1])
    fig.update_yaxes(title_text="f1 score", row=r+1, col=c+1, range=[-0.1,0.8])
#fig.update_layout(title=f'channel = {channel}',xaxis_title="threshold",yaxis_title="f1 score",legend_title="Legend Title",font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))
#fig.update_layout(title={'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
fig.update_layout(autosize=False,width=1400,height=900,)
fig.show()
fig.write_image('saved_images/f1_vs_threshold.png')

# %%
draw_contour = False
fig = make_subplots(rows=2, cols=2, subplot_titles=("leadtime = 15 min", "leadtime = 30 min", "leadtime = 45 min", "leadtime = 60 min"),shared_xaxes=True, shared_yaxes=True, horizontal_spacing = 0.05,vertical_spacing = 0.10)
for channel in [0, 1, 2, 3]:
    #fig = go.Figure()
    r,c = np.unravel_index(channel,(2,2))
    for row in [0,1,2]:
        x = []
        y = []
        z = []
        for thresh in np.linspace(0,1,21)[1:-1]:
            x.append(df[f'recall_with_channel-{thresh:.03}-{channel}'].iloc[row])
            y.append(df[f'precision_with_channel-{thresh:.03}-{channel}'].iloc[row])
            z.append(df[f'f1score_with_channel-{thresh:.03}-{channel}'].iloc[row])
        draw = pd.DataFrame({'t':np.round(np.linspace(0,1,21),2)[1:-1],'x':x,'y':y,'z':z})
        name = df['name'].iloc[row].split("_")[2].lower()
        #fig = fig.add_trace(go.Line(x = draw['x'],y = draw['y'], name = name,text = draw['z'], line=dict(width=2, color=cols[row]), legendgroup = f'{row}',showlegend = True if channel == 0 else False), row = r+1, col = c+1)
        fig = fig.add_trace(go.Line(x = draw['x'],y = draw['y'], name = name,text = draw['t'], line=dict(width=2, color=cols[row]), legendgroup = f'{row}',showlegend = True if channel == 0 else False), row = r+1, col = c+1)

    fig.update_xaxes(title_text="recall", row=r+1, col=c+1, range=[-0.1,1.1])
    fig.update_yaxes(title_text="precision", row=r+1, col=c+1, range=[-0.1,1.1])
    if draw_contour:
        for f1 in np.round(np.linspace(0.2,0.8,9),2):
        #for f1 in [0.3,.57,0.62]:
            x = np.linspace(0.01,0.9,51)
            y = (f1 * x)/(2*x-f1)
            y = y*(y>0)+2*(y<0)
            draw = pd.DataFrame({'x':x,'y':y})
            fig = fig.add_trace(go.Line(x = draw['x'],y = draw['y'],name=f'{f1}',showlegend=False,line=dict(color=f'hsv(0,{f1*100}%,100%)')), row = r+1, col = c+1)
#fig.update_layout(title=f'channel = {channel}',xaxis_title="threshold",yaxis_title="f1 score",legend_title="Legend Title",font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))
#fig.update_layout(title={'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
fig.update_layout(autosize=False,width=1400,height=900,)
fig.show()
fig.write_image('saved_images/ROC.png')

# %%
