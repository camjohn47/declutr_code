import dash
from dash import Dash
from dash.dcc import Graph
from dash.html import Button, Div
from dash.dependencies import Input, Output

from learning_visualizer import LearningVisualizer
import re

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
    ANCHOR_SELECTION_GRAPH_ID = "anchor-selection-graph"

    def __init__(self, visualizer_args={}):
        self.learning_visualizer = LearningVisualizer(**visualizer_args)
        self.get_anchor_button_id = lambda doc_ind: f"anchor-button-{doc_ind}"
        self.get_ind_from_button_id = lambda button_id: int(re.search(r'(?<=anchor-button-)\d+', button_id).group(0))
        self.button_ids = []

    def create_button(self, doc_ind):
        name = f"document {doc_ind + 1}"
        id = self.get_anchor_button_id(doc_ind)
        self.button_ids.append(id)
        button = Button(name, id=id, style=self.BUTTON_STYLE)
        return button

    def get_button_inputs(self):
        '''
        Get button inputs that can be used in Dash callbacks.
        '''

        inputs = [Input(component_id=button_id, component_property="n_clicks") for button_id in self.button_ids]
        return inputs

    def create_app(self):
        app = Dash()
        start_figure, document_count = self.learning_visualizer.generate_processing_visuals().__next__()
        start_graph = Graph(figure=start_figure, id=self.ANCHOR_SELECTION_GRAPH_ID)
        buttons = [self.create_button(doc_ind) for doc_ind in range(document_count)]
        button_ids = self.get_button_inputs()

        app.layout = Div([start_graph, *buttons])
        @app.callback(Output(component_id=self.ANCHOR_SELECTION_GRAPH_ID, component_property="figure"),
                      *button_ids)
        def select_anchor_figure(n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4):
            context = dash.callback_context
            triggered = context.triggered
            triggered_prop_ids = triggered[0]["prop_id"]
            clicked_button_id = [button_id for button_id in self.button_ids if button_id in triggered_prop_ids][0]
            clicked_doc_index = self.get_ind_from_button_id(clicked_button_id)
            print(f"UPDATE: Triggered = {triggered}, clicked document = {clicked_doc_index}")
            document = self.learning_visualizer.document_texts[clicked_doc_index]
            print(f"UPDATE: Clicked document = {document}")
            anchor_figure = self.learning_visualizer.build_anchor_document_fig(document, clicked_doc_index)
            return anchor_figure

        app.run_server(debug=True, port=7217)


if __name__ == "__main__":
    app = ProcessingApp()
    app.create_app()