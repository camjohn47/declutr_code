import ast
import sys

from sequence_processor import SequenceProcessor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from common_funcs import process_fig, get_code_df

from plotly.graph_objects import Scatter3d, Figure
from plotly.subplots import make_subplots
import numpy as np
from math import ceil, floor

class LearningVisualizer:
    ANCHOR_COLOR = "red"
    COLORS = ["blue", "black", "yellow", "green", "grey", "pink", "purple", "orange"]
    NUM_COLORS = len(COLORS)
    X_DELTA = 200
    Z_DELTA = .05
    Y_DELTA = .02
    TEXT_SIZE = 8
    SUBPLOTS_PER_ROW = 8
    PAPER_X_DELTA = 1 / SUBPLOTS_PER_ROW
    PAPER_Y_DELTA = 1 / 2
    ANCHOR_COL = ceil(SUBPLOTS_PER_ROW / 2) + 1
    COLUMN_WIDTH = 3000
    seq_processor_args = dict(documents_per_batch=4)

    def __init__(self, seq_processor=None, seq_processor_args={}, words_per_line=10, num_visuals_at_a_time=1):
        self.seq_processor_args.update(seq_processor_args)
        self.seq_processor = seq_processor if seq_processor else SequenceProcessor(**self.seq_processor_args)
        self.words_per_line = words_per_line
        self.num_visuals_at_a_time = num_visuals_at_a_time
        self.is_anchor_col = lambda col: col == self.ANCHOR_COL
        self.get_code_segment_col = lambda col: col if col < self.ANCHOR_COL else col + 1
        self.get_column_title = lambda col: f"code segment {self.get_code_segment_col(col)}" if not self.is_anchor_col(col)  \
                                        else "anchor code segment"
        self.build_subplot_parameters()
        self.contrasted_trace_coordinates = dict()
        self.contrasted_count = self.seq_processor.batch_size
        self.contrasted_indices = range(self.contrasted_count)

    def build_subplot_parameters(self):
        contrasted_count = self.seq_processor.batch_size
        self.row_count = ceil(contrasted_count / self.SUBPLOTS_PER_ROW)
        self.subplot_specs = [[dict(type="scene") for col in range(self.SUBPLOTS_PER_ROW + 1)] for row in range(self.row_count + 1)]
        self.column_widths = [self.COLUMN_WIDTH for i in range(self.SUBPLOTS_PER_ROW + 1)]
        self.column_titles = [self.get_column_title(col + 1) for col in range(self.SUBPLOTS_PER_ROW + 1)]

    @staticmethod
    def pad_input_sequences(contrasted_id_seqs):
        max_token_count = max([len(seq) for seq in contrasted_id_seqs])
        contrasted_id_seqs = pad_sequences(contrasted_id_seqs, maxlen=max_token_count)
        return contrasted_id_seqs

    @staticmethod
    def pad_text_sequences(text_seqs):
        seq_lens = [len(seq) for seq in text_seqs]
        max_len = max(seq_lens)
        pad_text = lambda seq, i: seq + ['' for token in range(max_len - seq_lens[i])]
        padded_text_seqs = [pad_text(seq, i) for i, seq in enumerate(text_seqs)]
        return padded_text_seqs

    def get_text_seqs(self, inputs):
        '''
        Converts anchor token ids and contrasted token ids to text sequences for visualization.
        '''

        anchor_id_seq = [inputs['anchor_sequence'].numpy().tolist()]
        contrasted_id_sequences = inputs['contrasted_sequences'].numpy().tolist()
        contrasted_id_sequences = self.pad_input_sequences(contrasted_id_sequences)
        anchor_text_seq = self.seq_processor.convert_ids_to_tokens(anchor_id_seq)[0]
        contrasted_text_seqs = self.seq_processor.convert_ids_to_tokens(contrasted_id_sequences)
        return anchor_text_seq, contrasted_text_seqs

    def get_text_chunks(self, text):
        '''
        Returns pairs of text and z (height) coordinate for each line of text.
        '''

        text_chunks = []
        text_words = text.split()
        word_count = len(text_words)
        line_count = ceil(len(text_words) / self.words_per_line)
        z_max = floor(line_count / 2) * self.Z_DELTA
        z_min = -z_max
        z = np.linspace(z_max, z_min, line_count)
        lines = range(line_count)

        for i, line in enumerate(lines):
            text_range_end = min(self.words_per_line * (i + 1), word_count)
            text_range = range(self.words_per_line * i, text_range_end)
            line_text = " ".join([text_words[j] for j in text_range])
            line_z = z[i]
            chunk = [line_text, line_z]
            text_chunks.append(chunk)

        return text_chunks

    def build_anchor_traces(self, anchor_text, num_contrasted):
        anchor_traces = []
        center_x = int(num_contrasted * self.X_DELTA / 2)
        anchor_chunks = self.get_text_chunks(anchor_text)

        for i, chunk in enumerate(anchor_chunks):
            line_text, line_z = chunk
            anchor_trace = Scatter3d(x=[center_x], y=[0], z=[line_z], text=line_text, name="anchor text", mode="text",
                                     textfont=dict(color=self.ANCHOR_COLOR, size=self.TEXT_SIZE), textposition="top center")
            anchor_traces.append(anchor_trace)

        return anchor_traces

    def split_contrasted_trace(self, contrasted_text, index):
        line_traces = []
        text_chunks = self.get_text_chunks(contrasted_text)
        traces_coordinates = []

        for i, chunk in enumerate(text_chunks):
            line_text, line_z = chunk
            color_index = index % self.NUM_COLORS
            color = self.COLORS[color_index]
            x = index * self.X_DELTA
            y = 0 - (index * self.Y_DELTA)

            # Need to keep track of each contrasted text's trace coordinates.
            line_trace = Scatter3d(x=[x], y=[y], z=[line_z], text=line_text, name=f"contrasted text {index + 1}",
                                   mode="text", textfont=dict(color=color, size=self.TEXT_SIZE), textposition="top center")
            line_traces.append(line_trace)
            trace_coordinates = dict(x=x, y=y, z=line_z)
            traces_coordinates.append(trace_coordinates)

        self.contrasted_trace_coordinates[index] = traces_coordinates
        return line_traces

    def build_contrasted_traces(self, contrasted_texts):
        contrasted_traces = []

        for contrasted_index, contrasted_text in enumerate(contrasted_texts):
            traces = self.split_contrasted_trace(contrasted_text, contrasted_index)
            contrasted_traces += traces

        return contrasted_traces

    def learning_fig_from_texts(self, anchor_text, contrasted_texts):
        fig = Figure()
        anchor_traces = self.build_anchor_traces(anchor_text, len(contrasted_texts))
        fig.add_traces(anchor_traces)
        contrasted_traces = self.build_contrasted_traces(contrasted_texts)
        fig.add_traces(contrasted_traces)
        fig.update_xaxes(title="Proposed Text")
        fig.update_layout(title_text="DeClutr Code Learning Task", title_x=0.5)
        return fig

    def get_batch_fig(self, batch):
        inputs, labels = batch
        anchor_text_seq, contrasted_text_seqs = self.get_text_seqs(inputs)
        batch_fig = self.learning_fig_from_texts(anchor_text_seq, contrasted_text_seqs)
        return batch_fig

    def build_anchor_subplot(self, fig, anchor_text, contrasted_count):
        anchor_traces = self.build_anchor_traces(anchor_text, contrasted_count)
        fig.add_traces(anchor_traces, rows=1, cols=self.ANCHOR_COL)
        return fig

    def build_contrasted_traces_subplots(self, fig, contrasted_texts):
        for contrasted_index, contrasted_text in enumerate(contrasted_texts):
            traces = self.split_contrasted_trace(contrasted_text, contrasted_index)
            subplot_row = ceil(contrasted_index / self.SUBPLOTS_PER_ROW) + 1
            subplot_row = max(subplot_row, 2)
            subplot_col = (contrasted_index % self.SUBPLOTS_PER_ROW) + 1
            subplot_col = subplot_col + 1 if subplot_col >= self.ANCHOR_COL else subplot_col
            subplot_rows = [subplot_row] * len(traces)
            subplot_cols = [subplot_col] * len(traces)
            fig.add_traces(traces, rows=subplot_rows, cols=subplot_cols)

        return fig

    def learning_subplots_from_texts(self, anchor_text, contrasted_texts):
        contrasted_count = len(contrasted_texts)
        row_count = ceil(contrasted_count / self.SUBPLOTS_PER_ROW)
        fig = make_subplots(row_count + 1, self.SUBPLOTS_PER_ROW + 1, specs=self.subplot_specs,
                            column_widths=self.column_widths, column_titles=self.column_titles)
        fig = self.build_anchor_subplot(fig, anchor_text, contrasted_count)
        fig = self.build_contrasted_traces_subplots(fig, contrasted_texts)
        #fig.update_scenes(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
        fig.update_layout(title_text="Can you code better than an AI? Choose the matching code for the anchor code.", 
                          title_x=0.5, title_font=dict(size=32), showlegend=False)
        return fig

    def get_subplots_fig(self, batch):
        inputs, labels = batch
        anchor_text_seq, contrasted_text_seqs = self.get_text_seqs(inputs)
        batch_fig = self.learning_subplots_from_texts(anchor_text_seq, contrasted_text_seqs)
        return batch_fig

    @staticmethod
    def get_user_input():
        user_input = input("INPUT: Do you want more visuals? Please type 'yes' or 'no'. ")
        invalid_input = user_input not in ["yes", "no"]

        while invalid_input:
            user_input = input("INPUT: Sorry, that's not a valid input. Do you want more visuals? Please type 'yes' or 'no'. ")
            invalid_input = user_input not in ["yes", "no"]

        return user_input

    def build_button_shape(self, index):
        traces = self.contrasted_trace_coordinates[index]
        min_x = min([trace["x"] for trace in traces])
        max_x = max([trace["x"] for trace in traces])
        min_y = min([trace["y"] for trace in traces])
        max_y = max([trace["y"] for trace in traces])
        x0 = index * self.PAPER_X_DELTA
        x1 = (index + 1) * self.PAPER_X_DELTA
        button = [dict(type="rectangle", xref="paper", yref="paper", x0=x0, x1=x1, y0=0, y1=0.25, color="red")]
        return button

    def build_button(self, index, button_shape):
        button = dict(label=f"code segment {index + 1}", method="relayout", args = ["shapes", button_shape])
        return button

    def add_guessing_buttons(self, learning_fig):
        button_shapes = list(map(self.build_button_shape, self.contrasted_indices))
        buttons = [self.build_button(index, button_shape) for index, button_shape in enumerate(button_shapes)]
        print(f"UPDATE: Buttons = {buttons}")
        learning_fig.update_layout(updatemenus=[dict(type="buttons", buttons=buttons)])
        return learning_fig

    def build_learning_visuals(self):
        '''
        A plotly figure showing the input anchor text and contrasted text sequences.
        '''

        code_df = get_code_df()
        self.seq_processor.fit_tokenizer_in_chunks(code_df, text_column="code")
        code_df = self.seq_processor.tokenize_document_df(code_df, text_column="code")
        code_df = self.seq_processor.add_document_size_column(code_df)
        code_df = self.seq_processor.filter_documents_by_size(code_df)
        declutr_dataset = self.seq_processor.get_dataset(code_df)
        ask_user_about_stop = lambda batch_index: batch_index % self.num_visuals_at_a_time == 0 and batch_index != 0

        for i, batch in enumerate(declutr_dataset):
            if ask_user_about_stop(i):
                user_input = self.get_user_input()

                if user_input == "no":
                    print(f"EXIT: Stopping visuals as requested. ")
                    sys.exit(1)

            #batch_fig = self.get_batch_fig(batch)
            #fig_name = f"batch_{i}_learning_visual"
            #process_fig(batch_fig, fig_name)

            subplots_fig = self.get_subplots_fig(batch)
            subplots_fig = self.add_guessing_buttons(subplots_fig)
            fig_name = f"batch_{i}_learning_subplots"
            process_fig(subplots_fig, fig_name)

if __name__ == "__main__":
    # Simple script for visualizing the DeClutr code learning task.
    _, sampling = sys.argv
    sampling = ast.literal_eval(sampling) / 100
    visualizer = LearningVisualizer()
    visualizer.build_learning_visuals()

