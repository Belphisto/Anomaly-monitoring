# coding: utf-8

import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def _apply_axis_style(fig):
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


def _discrete_colorscale(colors):
    if not colors:
        colors = ['#636EFA']

    n = len(colors)
    if n == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    scale = []
    for i, color in enumerate(colors):
        left = i / n
        right = (i + 1) / n
        scale.append([left, color])
        scale.append([right, color])

    scale[0][0] = 0.0
    scale[-1][0] = 1.0
    return scale


def plot_score_probability(df, plot_path):
    score_probability = sns.displot(
        data=df,
        x='min_similarity_score',
        hue='real_label',
        fill=True,
        stat="probability"
    )
    score_probability.savefig(plot_path)


def plot_ts(ts, N, plot_path, title='Time Series'):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(
            x=np.arange(N),
            y=list(ts),
            line=dict(color='#636EFA', width=2),
            name=title
        ),
        row=1, col=1
    )

    _apply_axis_style(fig)
    fig.update_layout(
        title_text=title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=1300,
        height=450,
    )
    fig.write_image(plot_path, scale=3)


def plot_multivariate_ts(multi_ts, n, d, plot_path, title='Multivariate Time Series'):
    colors = ['#636EFA', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
              '#FF97FF', '#FECB52', '#8C564B', '#316395', '#BCBD22', '#7F7F7F',
              '#2CA02C', '#AF0038']

    fig = make_subplots(
        rows=d,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=tuple([f"Time series #{i}" for i in range(d)])
    )

    for i in range(d):
        fig.add_trace(
            go.Scatter(
                x=np.arange(n),
                y=multi_ts[:, i],
                name=f"Time series #{i}",
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            row=i + 1, col=1
        )

    _apply_axis_style(fig)
    fig.update_layout(
        title_text=title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=1300,
        height=max(300, 220 * d),
    )
    fig.write_image(plot_path, scale=3)


def plot_similarity_scores(ts_test, similarity_scores, true_label, N, plot_path):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Test time series", "Ground truth label", "Anomaly score")
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(N),
            y=list(ts_test),
            line=dict(color='#636EFA', width=2),
            name="Test time series"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(N),
            y=list(true_label),
            line=dict(color='#00CC96', width=2),
            name="Ground truth label"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(N),
            y=list(similarity_scores),
            line=dict(color='#EF553B', width=2),
            name="Anomaly score"
        ),
        row=3, col=1
    )

    _apply_axis_style(fig)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        width=1300,
        height=800,
    )

    fig.write_image(plot_path, scale=3)


def plot_anomaly_regions(ts_test, subs_similarity_scores, anomaly_regions, threshold, N, plot_path):
    fig = make_subplots(shared_xaxes=True, rows=2, cols=1)

    fig.add_trace(go.Scatter(x=np.arange(N), y=list(ts_test), name="Test Time Series"), row=1, col=1)

    for left, right in anomaly_regions:
        fig.add_trace(
            go.Scatter(
                x=np.arange(left, right),
                y=list(ts_test[left:right]),
                line=dict(color='red'),
                showlegend=False
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=np.arange(N),
            y=list(subs_similarity_scores),
            line=dict(color='green'),
            name="Anomaly score"
        ),
        row=2, col=1
    )
    fig.add_hrect(
        y0=threshold,
        y1=np.max(subs_similarity_scores),
        line_width=0,
        fillcolor="red",
        opacity=0.2,
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(N),
            y=[threshold] * N,
            line_width=3,
            line_dash="dash",
            line=dict(color='red'),
            name="Threshold"
        ),
        row=2, col=1
    )

    _apply_axis_style(fig)
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
    colors = ['#636EFA', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
              '#FF97FF', '#FECB52', '#8C564B', '#316395', '#BCBD22', '#7F7F7F',
              '#2CA02C', '#AF0038']

    fig = make_subplots(rows=d + 2, cols=1, shared_xaxes=True)

    for i in range(d):
        fig.add_trace(
            go.Scatter(
                x=np.arange(n),
                y=multi_ts[:, i],
                name="Time series #" + str(i),
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            row=i + 1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=np.arange(N),
            y=list(true_label),
            line=dict(color='#00CC96', width=2),
            name="Ground truth label"
        ),
        row=d + 1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(N),
            y=list(similarity_scores),
            line=dict(color='#EF553B', width=2),
            name="Anomaly score"
        ),
        row=d + 2, col=1
    )

    _apply_axis_style(fig)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        width=1300,
        height=160 * (d + 2),
    )

    fig.write_image(plot_path, scale=3)


def plot_multi_snippets(multi_train_ts, multi_snippets, train_label, n, d, m, snippets_num, plot_path):
    colors = ['#636EFA', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
              '#FF97FF', '#FECB52', '#8C564B', '#316395', '#BCBD22', '#7F7F7F',
              '#2CA02C', '#AF0038']

    rows = d + 2
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.22] * d + [0.10, 0.18],
        subplot_titles=tuple(
            [f"Train time series #{i}" for i in range(d)] + ["Train label", "Snippets"]
        )
    )

    x = np.arange(n)

    for dim in range(d):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=multi_train_ts[:, dim],
                name=f"Time series #{dim}",
                line=dict(color=colors[dim % len(colors)], width=2)
            ),
            row=dim + 1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(train_label)),
            y=list(train_label),
            line=dict(color='#00CC96', width=2),
            name="Train label"
        ),
        row=d + 1, col=1
    )

    snippet_indices = []
    if isinstance(multi_snippets, dict):
        snippet_indices = multi_snippets.get('indices', [])
    if snippet_indices is None:
        snippet_indices = []

    snippet_indices = [int(v) for v in snippet_indices[:snippets_num]]

    snippets_labels = np.full(n, -1, dtype=float)
    for cls, start in enumerate(snippet_indices):
        left = max(0, int(start))
        right = min(n, int(start) + int(m))
        snippets_labels[left:right] = cls

        for dim in range(d):
            seg_x = np.arange(left, right)
            seg_y = multi_train_ts[left:right, dim]
            fig.add_trace(
                go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    mode='lines',
                    line=dict(color=colors[cls % len(colors)], width=4),
                    showlegend=False
                ),
                row=dim + 1, col=1
            )

    heatmap_colors = colors[:max(1, len(snippet_indices))]
    colorscale = _discrete_colorscale(heatmap_colors)
    z_values = np.where(snippets_labels < 0, np.nan, snippets_labels)

    fig.add_trace(
        go.Heatmap(
            z=[z_values],
            x=np.arange(n),
            y=["snippets"],
            zmin=0,
            zmax=max(0, len(snippet_indices) - 1),
            colorscale=colorscale,
            showscale=False,
            hoverongaps=False
        ),
        row=d + 2, col=1
    )

    _apply_axis_style(fig)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        width=1400,
        height=max(700, 220 * rows),
    )

    fig.write_image(plot_path, scale=3)

def plot_snippets(ts, ts_snippets, n, m, snippets_num, plot_path):
    colors = ['#636EFA', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
              '#FF97FF', '#FECB52', '#8C564B', '#316395', '#BCBD22', '#7F7F7F',
              '#2CA02C', '#AF0038']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04)

    ts = np.asarray(ts)
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=list(ts),
            line=dict(color='#636EFA', width=2),
            name="Time Series"
        ),
        row=1, col=1
    )

    snippet_indices = []
    if isinstance(ts_snippets, dict):
        snippet_indices = ts_snippets.get('indices', [])
    if snippet_indices is None:
        snippet_indices = []

    snippet_indices = [int(v) for v in snippet_indices[:snippets_num]]

    snippet_labels = np.full(n, -1, dtype=float)

    for cls, start in enumerate(snippet_indices):
        left = max(0, int(start))
        right = min(n, int(start) + int(m))
        snippet_labels[left:right] = cls

        fig.add_trace(
            go.Scatter(
                x=np.arange(left, right),
                y=list(ts[left:right]),
                line=dict(color=colors[cls % len(colors)], width=4),
                showlegend=False
            ),
            row=1, col=1
        )

    heatmap_colors = colors[:max(1, len(snippet_indices))]
    colorscale = _discrete_colorscale(heatmap_colors)
    z_values = np.where(snippet_labels < 0, np.nan, snippet_labels)

    fig.add_trace(
        go.Heatmap(
            z=[z_values],
            x=np.arange(n),
            y=["snippets"],
            zmin=0,
            zmax=max(0, len(snippet_indices) - 1),
            colorscale=colorscale,
            showscale=False,
            hoverongaps=False
        ),
        row=2, col=1
    )

    _apply_axis_style(fig)
    fig.update_layout(
        title_text="Snippets in the time series",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=1300,
        height=650,
    )

    fig.write_image(plot_path, scale=3)

def plot_profiles(profiles, n, snippets_num, plot_path):
    colors = ['#636EFA', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
              '#FF97FF', '#FECB52', '#8C564B', '#316395', '#BCBD22', '#7F7F7F',
              '#2CA02C', '#AF0038']

    fig = make_subplots(rows=snippets_num, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    for i in range(snippets_num):
        if i >= len(profiles):
            break

        profile = np.asarray(profiles[i], dtype=float).ravel()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(profile)),
                y=list(profile),
                line=dict(color=colors[i % len(colors)], width=2),
                name=f"Profile #{i}"
            ),
            row=i + 1, col=1
        )

    _apply_axis_style(fig)
    fig.update_layout(
        title_text="MPdist profiles",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=1300,
        height=max(400, 220 * snippets_num),
    )

    fig.write_image(plot_path, scale=3)

def plot_annotation(ts, true_label, pred_label, N, plot_path, title="Annotation"):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Time series", "True label", "Predicted annotation")
    )

    ts = np.asarray(ts).ravel()
    true_label = np.asarray(true_label).ravel()
    pred_label = np.asarray(pred_label).ravel()

    n_ts = min(len(ts), int(N))
    n_true = min(len(true_label), int(N))
    n_pred = min(len(pred_label), int(N))

    fig.add_trace(
        go.Scatter(
            x=np.arange(n_ts),
            y=list(ts[:n_ts]),
            line=dict(color='#636EFA', width=2),
            name="Time series"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n_true),
            y=list(true_label[:n_true]),
            line=dict(color='#00CC96', width=2),
            name="True label"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n_pred),
            y=list(pred_label[:n_pred]),
            line=dict(color='#EF553B', width=2),
            name="Predicted annotation"
        ),
        row=3, col=1
    )

    _apply_axis_style(fig)
    fig.update_layout(
        title_text=title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=1300,
        height=800,
    )

    fig.write_image(plot_path, scale=3)