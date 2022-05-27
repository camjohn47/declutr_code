from dash import Dash
from dash.dcc import Graph
from dash.html import Button, Div

from learning_visualizer import LearningVisualizer

class ProcessingApp():
    '''
    A Plotly Dash app for allowing users to participate in self-supervised pre-processing. Specifically, enabling users
    to create anchors, positives, and negatives for a DeClutr document batch.
    '''

    #TODO: Integrate this with Website class in other repo.
    FONT_FAMILY = "Verdana"
    STYLE = {"font-family": FONT_FAMILY, "color": "grey", "background-color": "black", "margin-bottom": "00px",
             "margin:top":"0px", "height": "40px"}
    TAB_COLORS = dict(background="black", border="grey", primary="grey")
    SUB_TAB_STYLE = {**STYLE, "font-size": 10}
    FILLER_STYLE = {**STYLE, "height": "1000px"}
    GRAPH_STYLE = {**STYLE, "height": "300px"}
    BUTTON_STYLE = {**STYLE, "background-color": "white", "height": "50px", "margin-bottom": "0px", "margin-top": "20px"}
    SELECTED_STYLE = {**STYLE, "color": "blue"}

    def __init__(self, visualizer_args={}):
        self.learning_visualizer = LearningVisualizer(**visualizer_args)
        self.get_anchor_button_id = lambda doc_ind: f"anchor-button-{doc_ind}"
        self.button_ids = []

    def create_button(self, doc_ind):
        name = f"document {doc_ind + 1}"
        id = self.get_anchor_button_id(doc_ind)
        button = Button(name, id=id, style=self.BUTTON_STYLE)
        return button

    def create_app(self):
        app = Dash()
        start_figure, document_count = self.learning_visualizer.generate_processing_visuals().__next__()
        start_graph = Graph(figure=start_figure)
        buttons = [self.create_button(doc_ind) for doc_ind in range(document_count)]
        app.layout = Div([start_graph, *buttons])
        app.run_server(debug=True, port=7217)


if __name__ == "__main__":
    app = ProcessingApp()
    app.create_app()