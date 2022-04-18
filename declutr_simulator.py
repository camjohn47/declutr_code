import ast
import os
import sys

from sequence_processor import SequenceProcessor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from code_parser import CodeParser

import plotly.express as px

import numpy as np

from itertools import product

def pad_input_sequences(contrasted_id_seqs):
    max_token_count = max([len(seq) for seq in contrasted_id_seqs])
    contrasted_id_seqs = pad_sequences(contrasted_id_seqs, maxlen=max_token_count)
    return contrasted_id_seqs

def pad_text_sequences(text_seqs):
    seq_lens = [len(seq) for seq in text_seqs]
    max_len = max(seq_lens)
    pad_text = lambda seq, i: seq + ['' for token in range(max_len - seq_lens[i])]
    padded_text_seqs = [pad_text(seq, i) for i, seq in enumerate(text_seqs)]
    return padded_text_seqs

def plot_batch_text(inputs, labels, seq_processor):
    '''
    A plotly figure showing the input anchor text and contrasted text sequences.
    '''

    anchor_id_seq = [inputs['anchor_sequence'].numpy().tolist()]
    contrasted_id_sequences = inputs['contrasted_sequences'].numpy().tolist()
    contrasted_id_sequences = pad_input_sequences(contrasted_id_sequences)
    print(f'UPDATE: Before conversion anchor = {anchor_id_seq}, contrasted = {contrasted_id_sequences}')
    anchor_text_seq = seq_processor.convert_ids_to_tokens(anchor_id_seq)
    contrasted_text_seqs = seq_processor.convert_ids_to_tokens(contrasted_id_sequences)
    print(f'UPDATE: After conversion anchor = {anchor_text_seq}, contrasted = {contrasted_text_seqs}')

    # Get maximum text size and pad contrasted seqs to this amount.
    contrasted_text_seqs = [seq.split() for seq in contrasted_text_seqs]
    contrasted_text_seqs = pad_text_sequences(contrasted_text_seqs)
    seq_lengths = [len(seq) for seq in contrasted_text_seqs]
    contrasted_count = len(contrasted_text_seqs)
    y_axis = list(range(contrasted_count))
    max_text_size = max([len(seq) for seq in contrasted_text_seqs])
    x_axis = list(range(max_text_size))
    print(f'UPDATE: seq lengths {seq_lengths}, contrated count {contrasted_count}, mts={max_text_size}')
    xy = list(product(x_axis, y_axis))
    x_axis = [coord[0] for coord in xy]
    y_axis = [coord[1] for coord in xy]
    #print(f'UPDATE: xy = {xy}')
    hover_text = [contrasted_seq[x] for contrasted_seq in contrasted_text_seqs for x in range(len(contrasted_seq))]
    print(f'UPDATE: Contrasted seqs {contrasted_text_seqs}')
    print(f'UPDATE: Hover text size = {len(hover_text)}, sample = {hover_text[0]}')
    title = f"Negative samples for anchor={anchor_text_seq}"
    fig = px.scatter(x=y_axis, y=x_axis, text=hover_text, title=title, size=[0 for marker in range(len(hover_text))])
    fig.update_xaxes(dict(title='Place in Script'))
    fig.update_yaxes(dict(title='Samples'))
    fig.show()

# Simple script for visualizing the DeClutr code learning task.
_, sampling = sys.argv
sampling = ast.literal_eval(sampling) / 100
parser = CodeParser()
code_directory = os.path.join('/Users', 'calvinjohn', 'Documents')
code_df = parser.code_directory_to_df(code_directory)
code_df = code_df.sample(frac=sampling)
print(f'UPDATE: Code dataframe info. ')
code_df.info()

seq_processor = SequenceProcessor()
declutr_dataset = seq_processor.get_declutr_training_dataset(code_df)
for batch in declutr_dataset:
    inputs, labels = batch
    plot_batch_text(inputs, labels, seq_processor)

