# coding: utf-8

import numpy as np
import seaborn as sns
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
#plotly.offline.init_notebook_mode(connected=True)


def plot_score_probability(df, plot_path):
    
    score_probability = sns.displot(
                data=df,
                x='min_similarity_score',
                hue='real_label',
                #kind='kde',
                fill=True,
                #col='variable'
                stat="probability"
                )
                
    score_probability.savefig(plot_path)


def plot_similarity_scores(ts_test, similarity_scores, true_label, N, plot_path):
    
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Test time series", "Ground truth label", "Anomaly score")
                        )
    
    # plot the test time series
    fig.add_trace(go.Scatter(x=np.arange(N),
                             y=list(ts_test),
                             line=dict(color='#636EFA', width=2),
                             name="Test time series"),
                  row=1, col=1)
                  
    # plot the true labels
    fig.add_trace(go.Scatter(x=np.arange(N),
                             y=list(true_label),
                             line=dict(color='#00CC96', width=2),
                             name="Ground truth label"),
                  row=2, col=1)
                  
    #plot the predicted scores
    fig.add_trace(go.Scatter(x=np.arange(N),
                             y=list(similarity_scores),
                             line=dict(color='#EF553B', width=2),
                             name="Anomaly score"),
                  row=3, col=1)
    
    fig.update_annotations(font=dict(size=16, color='black'))
    
    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=14, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
        
    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=14), color='black',
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
     
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      width=1300, height=800,
                      )
    
    fig.write_image(plot_path, scale=3)


def plot_anomaly_regions(ts_test, subs_similarity_scores, anomaly_regions, threshold, N, plot_path):
    
    fig = make_subplots(shared_xaxes=True, rows=2, cols=1)
    
    fig.add_trace(go.Scatter(x=np.arange(N), y=list(ts_test), name="Test Time Series"), row=1, col=1)
    
    for i in range(len(anomaly_regions)):
        fig.add_trace(go.Scatter(x=np.arange(anomaly_regions[i][0], anomaly_regions[i][1]), y=list(ts_test[anomaly_regions[i][0]:anomaly_regions[i][1]]), line=dict(color='red'), showlegend=False), row=1, col=1)
    
    #fig.add_hline(y=0.9)
    
    fig.add_trace(go.Scatter(x=np.arange(N), y=list(subs_similarity_scores), line=dict(color='green'), name="Anomaly score"), row=2, col=1)
    fig.add_hrect(y0=threshold, y1=np.max(subs_similarity_scores), line_width=0, fillcolor="red", opacity=0.2, row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(N), y=[threshold]*N, line_width=3, line_dash="dash", line=dict(color='red'), name="Threshold"), row=2, col=1)
    #fig.add_hline(y=threshold, line_width=4, line_dash="dash", line_color="red", name="Threshold", row=2, col=1)

    fig.write_image(plot_path, scale=3)


def plot_discords(ts, mp, discords, n, m, N, discords_num, plot_path):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04)

    ts = np.asarray(ts)
    mp = np.asarray(mp)

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=list(ts),
            line=dict(color='#636EFA', width=2),
            name="Time Series"
        ),
        row=1, col=1
    )

    # выделяем найденные discord-области
    for idx in list(discords)[:discords_num]:
        idx = int(idx)
        left = max(0, idx)
        right = min(n, idx + int(m))

        fig.add_trace(
            go.Scatter(
                x=np.arange(left, right),
                y=list(ts[left:right]),
                line=dict(color='red', width=3),
                showlegend=False
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(mp)),
            y=list(mp),
            line=dict(color='#00CC96', width=2),
            name="Matrix Profile"
        ),
        row=2, col=1
    )

    for idx in list(discords)[:discords_num]:
        idx = int(idx)
        if 0 <= idx < len(mp):
            fig.add_trace(
                go.Scatter(
                    x=[idx],
                    y=[mp[idx]],
                    mode='markers',
                    marker=dict(symbol='star', color='red', size=9),
                    showlegend=False
                ),
                row=2, col=1
            )

    fig.update_annotations(font=dict(size=16, color='black'))

    fig.update_xaxes(
        showgrid=False,
        linecolor='#000',
        ticks='outside',
        tickfont=dict(size=14, color='black'),
        linewidth=1,
        tickwidth=1,
        mirror=True
    )

    fig.update_yaxes(
        showgrid=False,
        linecolor='#000',
        ticks='outside',
        tickfont=dict(size=14, color='black'),
        zeroline=False,
        linewidth=1,
        tickwidth=1,
        mirror=True
    )

    fig.update_layout(
        title_text="Top-k discords in the time series",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=1300,
        height=700,
    )

    fig.write_image(plot_path, scale=3)


def plot_comparison_similarity_scores(multi_ts, similarity_scores, true_label, n, N, d, plot_path):
    
    colors = ['#636EFA', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#8C564B', '#316395', '#BCBD22', '#7F7F7F', '#2CA02C', '#AF0038']
    
    fig = make_subplots(rows=d+2, cols=1,
                        shared_xaxes=True,
                        #vertical_spacing=0.05,
                        #row_heights=[0.5, 0.25, 0.25],
                        #subplot_titles=( "Ground truth label", "Anomaly score")
                        )
                    
    # plot time series
    for i in range(d):
        fig.add_trace(go.Scatter(x=np.arange(n),
                                 y=multi_ts[:,i],
                                 name="Time series #" + str(i),
                                 line=dict(color=colors[i], width=2)),
                      row=i+1, col=1)
        
    # plot the true labels
    fig.add_trace(go.Scatter(x=np.arange(N),
                             y=list(true_label),
                             line=dict(color='#00CC96', width=2),
                             name="Ground truth label"),
                  row=d+1, col=1)
    
    #plot the predicted scores
    fig.add_trace(go.Scatter(x=np.arange(N),
                             y=list(similarity_scores),
                             line=dict(color='#EF553B', width=2),
                             name="Anomaly score"),
                  row=d+2, col=1)
    
    fig.update_annotations(font=dict(size=16, color='black'))
    
    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=14, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    
    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=14), color='black',
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      width=1300, height=160*(d+2),
                      )
                        
    fig.write_image(plot_path, scale=3)
