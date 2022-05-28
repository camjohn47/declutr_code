import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import os
from pathlib import Path

import math

import sys

# Default for histogram comparison.
XAXIS_RANGE = [0, 1000]

def get_default_xrange(series, percentile=.9):
    xmax = series.quantile(q=percentile)
    x_range = [0, int(xmax)]
    return x_range

def make_histogram(df, column, layout_args={},x_range=None):
    fig = px.histogram(df, x=column, range_x=x_range)
    fig.update_layout(layout_args)
    return fig

def make_3d_scatter(df, xyz_columns=["x", "y", "z"], layout_args={},color_column=None):
    '''
    Make a 3D Plotly scatter of columns in a dataframe <df>.
    '''

    xyz_columns = xyz_columns if xyz_columns else df.columns
    xyz_in_df = all([column in df.columns for column in xyz_columns])

    if not xyz_in_df:
        print(f'ERROR: xyz columns = {xyz_columns} not found in 3d scatter df columns'
              f'={df.columns}. ')
        sys.exit(1)

    x_column, y_column, z_column = xyz_columns
    fig = px.scatter_3d(data_frame=df, x=x_column, y=y_column, z=z_column, color=color_column)
    fig.update_layout(layout_args)
    return fig 

def make_histogram_comparison(hist_vals, rows, cols, subplot_titles=[], subplot_xaxis=[], subplot_yaxis=[], layout_args={},
                              histnorm=None, histfunc="count", xbins=None, xaxis_range=None):
    '''
    Compare histograms side-by-side in a single plot.
    '''

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    removed_docs = max(hist_vals[0])
    yaxis_range = [0, removed_docs]
    min_doc_size = min(hist_vals[1])
    xaxis_range = xaxis_range if xaxis_range else XAXIS_RANGE

    for i in range(len(hist_vals)):
        # Shift up row and column by 1 because plotly rows\cols are 1-indexed.
        row = math.floor(i / cols) + 1
        col = (i % cols) + 1
        fig.add_trace(go.Histogram(x=hist_vals[i], histfunc=histfunc, histnorm=histnorm, xbins=xbins), row=row, col=col)
        fig.update_xaxes(subplot_xaxis[i], row=row, col=col, range=xaxis_range)
        fig.update_yaxes(subplot_yaxis[i], row=row, col=col, range=yaxis_range)
        fig.add_vline(min_doc_size, row=row, col=col)

    fig.update_layout(layout_args, showlegend=False)
    return fig

def process_fig(fig, path=None):
    '''
    Show a Plotly figure and save it in html.
    '''

    fig.show()

    if path:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        print(f"UPDATE: Saving figure to {path}.")
        fig.write_html(path)



