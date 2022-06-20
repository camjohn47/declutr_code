from classes.retrieval_models import QueryEncoderRetriever
from classes.sequence_processor import SequenceProcessor
from tensorflow.data import Dataset
from tensorflow import type_spec_from_value
from functools import partial

#TODO: Abstract or super Trainer class that DeClutr and Trainer can inherit from.
class TranslatorTrainer():
    def __init__(self, script_encoder_id, query_encoder_id, seq_processor_args={}, sampling=1):
        # 1. Build translator model.
        self.sampling = sampling
        self.script_encoder_id = script_encoder_id
        self.query_encoder_id = query_encoder_id
        self.retriever = QueryEncoderRetriever(script_encoder_id=self.script_encoder_id, query_encoder_id=self.query_encoder_id)
        self.translator = self.retriever.build_translator_nn()

        # 2. Build two sequence processors: one for docstring and the other for code.
        self.tokenizers_fitted = False
        self.script_seq_processor = SequenceProcessor(**seq_processor_args)
        self.docstring_seq_processor = SequenceProcessor(**seq_processor_args)

        # 3. a) Generate contrastive (non-declutr) batches of docstrings, with the anchor returned.
         #   b) Identify anchor

    def fit_tokenizers(self, code_df):
        self.script_seq_processor.fit_tokenizer_in_chunks(code_df, text_column="code")
        self.docstring_seq_processor.fit_tokenizer_in_chunks(code_df, text_column="docstring")

        if self.sampling == 1:
            self.script_seq_processor.cache_tokenizer("code")
            self.script_seq_processor.cache_tokenizer("docstring")

        self.tokenizers_fitted = True

    def tokenize_docstring(self, docstring):
        docstring_tokenized = self.docstring_seq_processor.texts_to_sequences([docstring])[0]
        return docstring_tokenized

    def generate_translator_batches(self, code_df):
        for translator_batch in self.script_seq_processor.yield_translator_batches(document_df=code_df):
            print(f"UPDATE: translator batch = {translator_batch}")
            batch_inputs, batch_labels = translator_batch
            anchor_docstring = batch_inputs["anchor_docstring"]
            contrasted_scripts = batch_inputs["contrasted_scripts"]
            anchor_docstring_tokenized = self.tokenize_docstring(anchor_docstring)
            translator_inputs = [contrasted_scripts, anchor_docstring_tokenized]
            yield translator_inputs, batch_labels

    def get_output_signature(self, code_df):
        sample = next(self.generate_translator_batches(code_df))
        sample_inputs, sample_labels = sample
        inputs_signature = type_spec_from_value(sample_inputs)
        labels_signature = type_spec_from_value(sample_labels)
        output_signature = (inputs_signature, labels_signature)
        return output_signature

    def train_translator(self, code_df):
        code_df = code_df.sample(frac=self.sampling)
        print(f"UPDATE: Translator code df has {len(code_df)} rows after sampling.")

        if not self.tokenizers_fitted:
            self.fit_tokenizers(code_df)

        generator_method = partial(self.generate_translator_batches, code_df=code_df)
        output_signature = self.get_output_signature(code_df)
        translator_data = Dataset.from_generator(generator_method, output_signature=output_signature)
        self.translator.fit(x=translator_data, epochs=1)




