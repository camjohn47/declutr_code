import plotly.express as px

def make_histogram(df, column, layout_args={}, save_path=None):
    fig = px.histogram(df, x=column)
    fig.show()
    fig.update_layout(layout_args)

    if save_path:
        print(f'UPDATE: Saving plotly histogram to {save_path}.')
        fig.savefig(save_path)