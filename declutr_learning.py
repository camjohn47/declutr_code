from sequence_processor import SequenceProcessor
from declutr_trainer import DeClutrTrainer
from code_parser import CodeParser
from common_funcs import find_code_df_methods

import os
import sys
import ast

from visuals import make_histogram

import time

# Parse scripts and build a code dataframe from them.
_, sampling, encoder_model, loss_objective  = sys.argv
sampling = ast.literal_eval(sampling) / 100
code_parser = CodeParser(code_type='python')
code_directory = '/Users/calvinjohn/Documents'
start = time.time()
code_df = code_parser.code_directory_to_df(code_directory)
parsing_time = time.time() - start
print(f'UPDATE: Parsing time = {parsing_time}')
code_df = code_df.sample(frac=sampling)
print(f'UPDATE: Final code dataframe info. ')
code_df.info()

# Plot histogram of different PL's.
layout_args = dict(title_text="Programming Language Distribution, Train Subset", title_x=0.5)
make_histogram(code_df, column="programming_language", layout_args=layout_args)
code_df = find_code_df_methods(code_df)

# Change tokenizer type to tf or pytorch.
anchors_per_doc = 1

# Fit tokenizer in chunks on the code dataframe and then start model training.
# Initialize declutr processor, model and training wrapper.
max_anchor_length = 64
sequence_processor_args = dict(loss_objective=loss_objective, max_anchor_length=max_anchor_length)
declutr_trainer = DeClutrTrainer(sequence_processor_args=sequence_processor_args, encoder_model=encoder_model)
model_id = "_".join([encoder_model, loss_objective, f"max_anchor_length={max_anchor_length}"])
declutr_args = dict(encoder_model=encoder_model, model_id=model_id)
declutr_model = declutr_trainer.build_declutr_model(code_df, declutr_args=declutr_args)
declutr_trainer.train_model(declutr_model, code_df)

