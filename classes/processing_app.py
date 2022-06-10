import dash
from dash import Dash
from dash.dcc import Graph
from dash.html import Button, Div, H1
from dash.dependencies import Input, Output

from classes.learning_visualizer import LearningVisualizer
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
    TITLE_STYLE = {**STYLE, "font-size": 32}
    ANCHOR_SELECTION_GRAPH_ID = "anchor-selection-graph"
    PORT = 1389

    def __init__(self, visualizer_args={}):
        self.learning_visualizer = LearningVisualizer(**visualizer_args)
        self.get_anchor_button_id = lambda doc_ind: f"anchor-button-{doc_ind}"
        self.get_ind_from_button_id = lambda button_id: int(re.search(r'(?<=anchor-button-)\d+', button_id).group(0))
        self.get_start_button_id = lambda start_ind: f"start-button-{start_ind}"
        self.doc_button_ids = []
        self.start_button_ids = []

    def create_doc_button(self, doc_ind):
        id = self.get_anchor_button_id(doc_ind)
        self.doc_button_ids.append(id)
        button = Button(f"document processing {doc_ind + 1}", id=id, style=self.BUTTON_STYLE)
        return button

    def get_doc_button_inputs(self):
        '''
        Get button inputs that can be used in Dash callbacks.
        '''

        inputs = [Input(component_id=button_id, component_property="n_clicks") for button_id in self.doc_button_ids]
        return inputs

    def get_doc_button_outputs(self):
        '''
        Get button outputs that can be used in Dash callbacks.
        '''

        inputs = [Output(component_id=button_id, component_property="style") for button_id in self.doc_button_ids]
        return inputs

    def create_start_button(self, start_index):
        id = self.get_start_button_id(start_index)
        self.start_button_ids.append(id)
        button = Button(f"word {start_index + 1}", id=id, style=dict(display="none"))
        return button

    def get_start_button_inputs(self):
        '''
        Get anchor start button inputs for choosing where to start anchor span in anchor document.
        '''

        inputs = [Input(component_id=button_id, component_property="n_clicks") for button_id in self.start_button_ids]
        return inputs

    def get_start_button_outputs(self):
        '''
        Get button outputs that can be used in Dash callbacks.
        '''

        inputs = [Output(component_id=button_id, component_property="style") for button_id in self.start_button_ids]
        return inputs

    def init_anchor_buttons(self, max_doc_word_count):
        start_buttons = [self.create_start_button(doc_ind) for doc_ind in range(max_doc_word_count)]
        start_button_inputs = self.get_start_button_inputs()
        return start_buttons, start_button_inputs

    def create_app(self):
        app = Dash()
        processing_visuals = self.learning_visualizer.generate_processing_visuals().__next__()
        start_figure = processing_visuals["fig"]
        document_count = processing_visuals["doc_count"]
        max_doc_word_count = processing_visuals["max_doc_word_count"]
        start_graph = Graph(figure=start_figure, id=self.ANCHOR_SELECTION_GRAPH_ID)
        doc_buttons = [self.create_doc_button(doc_ind) for doc_ind in range(document_count)]
        doc_button_inputs = self.get_doc_button_inputs()
        start_buttons, start_button_inputs = self.init_anchor_buttons(max_doc_word_count=max_doc_word_count)
        title = H1("Step 1: Choose your anchor document!", id="title", style=self.TITLE_STYLE)
        app.layout = Div([title, start_graph, *doc_buttons, *start_buttons])

        # Get outputs for updating different button sets through callbacks.
        anchor_figure_output = Output(component_id=self.ANCHOR_SELECTION_GRAPH_ID, component_property="figure")
        doc_button_outputs = self.get_doc_button_outputs()
        start_button_outputs = self.get_start_button_outputs()
        #print(f"UPDATE: Doc button inputs = {doc_button_inputs}, start button outputs = {start_button_outputs}")
        title_output = Output(component_id="title", component_property="children")

        # Updates figure with a zoomed in selection of the anchor document chosen by user.
        @app.callback(anchor_figure_output, title_output, *doc_button_outputs, *start_button_outputs, *doc_button_inputs)
        def select_anchor_figure(*n_clicks):
            '''
            Focuses on the chosen anchor document in the subplots. Also hides anchor document buttons and shows anchor
            start buttons.
            '''

            #TODO: Check for beginning with n_clicks_i = 0, and return identity.

            context = dash.callback_context
            triggered = context.triggered
            triggered_prop_ids = triggered[0]["prop_id"]
            clicked_button_id = [button_id for button_id in self.doc_button_ids if button_id in triggered_prop_ids][0]
            clicked_doc_index = self.get_ind_from_button_id(clicked_button_id)
            document = self.learning_visualizer.document_texts[clicked_doc_index]
            anchor_figure = self.learning_visualizer.build_anchor_document_fig(document, clicked_doc_index)
            doc_button_styles = [dict(display="none") for button in doc_button_outputs]
            anchor_button_styles = [self.BUTTON_STYLE for button in start_button_outputs]
            print(f"UPDATE: Selecting anchor document figure. ")
            anchor_title = f"You've chosen document {clicked_doc_index + 1}. Select a start word!"
            return anchor_figure, anchor_title, *doc_button_styles, *anchor_button_styles

        # Updates button visibility according to figure mode. Mode can be either multiple anchor documents before selection,
        # or single focused anchor document.
        #@app.callback(Input(component_id=self.ANCHOR_SELECTION_GRAPH_ID, component_property="figure"),
        #              Output())
        #def change_buttons_from_figure():
         #   pass

        app.run_server(debug=True, port=self.PORT)

if __name__ == "__main__":
    app = ProcessingApp()
    app.create_app()