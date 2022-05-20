import ast
import sys

from sequence_processor import SequenceProcessor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from common_funcs import process_fig, get_code_df

from plotly.graph_objects import Scatter3d, Figure
import numpy as np
import math

class LearningVisualizer:
    ANCHOR_COLOR = "red"
    COLORS = ["blue", "black", "yellow", "green", "grey", "pink", "purple", "orange"]
    NUM_COLORS = len(COLORS)
    X_DELTA = 200
    Z_DELTA = .2

    def __init__(self, seq_processor=None, words_per_line=10, num_visuals_at_a_time=1):
        self.seq_processor = seq_processor if seq_processor else SequenceProcessor()
        self.words_per_line = words_per_line
        self.num_visuals_at_a_time = num_visuals_at_a_time

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
        line_count = math.ceil(len(text_words) / self.words_per_line)
        z_max = math.floor(line_count / 2) * self.Z_DELTA
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
        center_x = int(num_contrasted / 2)
        anchor_chunks = self.get_text_chunks(anchor_text)

        for i, chunk in enumerate(anchor_chunks):
            line_text, line_z = chunk
            anchor_trace = Scatter3d(x=[center_x], y=[0], z=[line_z], text=line_text, name="anchor text", mode="text",
                                     textfont=dict(color=self.ANCHOR_COLOR, size=1))
            anchor_traces.append(anchor_trace)

        return anchor_traces

    def split_contrasted_trace(self, contrasted_text, index):
        line_traces = []
        text_chunks = self.get_text_chunks(contrasted_text)

        for i, chunk in enumerate(text_chunks):
            line_text, line_z = chunk
            color_index = index % self.NUM_COLORS
            color = self.COLORS[color_index]
            line_trace = Scatter3d(x=[index * self.X_DELTA], y=[-1], z=[line_z], text=line_text, name=f"contrasted text {i}",
                                   mode="text", textfont=dict(color=color))
            line_traces.append(line_trace)

        return line_traces

    def build_contrasted_traces(self, contrasted_texts):
        contrasted_traces = []

        for contrasted_index, contrasted_text in enumerate(contrasted_texts):
            print(f"UPDATE: Building trace for contrasted text = {contrasted_text}")
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
        fig.update_scenes(zaxis=dict(showticklabels=False))
        return fig

    def get_batch_fig(self, batch):
        inputs, labels = batch
        anchor_text_seq, contrasted_text_seqs = self.get_text_seqs(inputs)
        print(f"UPDATE: Retrieved CTS = {contrasted_text_seqs}")
        batch_fig = self.learning_fig_from_texts(anchor_text_seq, contrasted_text_seqs)
        return batch_fig

    @staticmethod
    def get_user_input():
        user_input = input("INPUT: Do you want more visuals? Please type 'yes' or 'no'. ")
        invalid_input = user_input not in ["yes", "no"]

        while invalid_input:
            user_input = input("INPUT: Sorry, that's not a valid input. Do you want more visuals? Please type 'yes' or 'no'. ")
            invalid_input = user_input not in ["yes", "no"]

        return user_input

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

            batch_fig = self.get_batch_fig(batch)
            fig_name = f"batch_{i}_learning_visual"
            process_fig(batch_fig, fig_name)

if __name__ == "__main__":
    # Simple script for visualizing the DeClutr code learning task.
    _, sampling = sys.argv
    sampling = ast.literal_eval(sampling) / 100
    visualizer = LearningVisualizer()
    visualizer.build_learning_visuals()

