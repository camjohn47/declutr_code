import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import math

import sys

def make_histogram(df, column, layout_args={}, save_path=None):
    fig = px.histogram(df, x=column)
    fig.update_layout(layout_args)
    fig.show()

    if save_path:
        print(f'UPDATE: Saving plotly histogram to {save_path}.')
        fig.savefig(save_path)

def make_3d_scatter(df, xyz_columns=[], layout_args={}, save_path=None, color_column=None):
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
    fig.show()

    if save_path:
        print(f'UPDATE: Saving plotly 3d-scatter to {save_path}.')
        fig.savefig(save_path)

def make_histogram_comparison(hist_vals, rows, cols, subplot_titles=[], subplot_xaxis=[], subplot_yaxis=[], layout_args={},
                              save_path=None, histnorm=None, xbins=None):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    min_doc_size = min(hist_vals[1])

    for i in range(len(hist_vals)):
        # Shift up row and column by 1 because plotly rows\cols are 1-indexed.
        row = math.floor(i / cols) + 1
        col = (i % cols) + 1
        fig.add_trace(go.Histogram(x=hist_vals[i], histnorm=histnorm, xbins=xbins), row=row, col=col)
        fig.update_xaxes(subplot_xaxis[i], row=row, col=col)
        fig.update_yaxes(subplot_yaxis[i], row=row, col=col)
        fig.add_vline(min_doc_size, row=row, col=col)

    fig.update_layout(layout_args)
    fig.show()

    if save_path:
        print(f'UPDATE: Saving histogram comparison')
        fig.savefig(save_path)



