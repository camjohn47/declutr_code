from sequence_processor import SequenceProcessor
from declutr_trainer import DeClutrTrainer
from code_parser import CodeParser
from transformers import TFAutoModel, AutoTokenizer
from transformers import ElectraTokenizerFast, ElectraModel, ElectraConfig

import sys

import ast

_, sampling, encoder_model = sys.argv
sampling = ast.literal_eval(sampling) / 100
code_parser = CodeParser()
code_directory = '/Users/calvinjohn/Documents'
code_df = code_parser.code_directory_to_df(code_directory)
code_df = code_df.sample(frac=sampling)
print(f'UPDATE: Final code dataframe info. ')
code_df.info()

code_df = code_parser.find_code_df_methods(code_df)

# Initialize declutr processor, model and training wrapper.
declutr_trainer = DeClutrTrainer()
# Change tokenizer type to tf or pytorch.
anchors_per_doc = 1
pad_sequences = True if encoder_model == 'transformer' else False
seq_processor = SequenceProcessor(documents_per_batch=16, chunk_size=int(1.0e3), anchors_per_document=anchors_per_doc,
                                  pad_sequences=pad_sequences)

# Fit tokenizer in chunks on the code dataframe and then start model training.
seq_processor.fit_tokenizer_in_chunks(code_df)
model_id = f'{encoder_model}_metal_test'
declutr_args = dict(encoder_model=encoder_model, model_id=model_id)
declutr_model = declutr_trainer.build_model_from_processor(seq_processor, declutr_args=declutr_args)
declutr_trainer.train_model(declutr_model, code_df)

