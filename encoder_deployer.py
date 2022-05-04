import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import make_ndarray
from tensorflow.keras.utils import Progbar

import numpy as np

from sequence_models import DeClutrContrastive, TransformerEncoder, RNNEncoder

from common_funcs import get_sequence_processor, run_with_time, add_features_to_df, get_default_feature_cols, cast_tf_tokens

import os

class EncoderDeployer():
    ENCODER_TO_CUSTOM_OBJECTS = dict(transformer=dict(transformer_encoder=TransformerEncoder), rnn=dict(rnn_encoder=RNNEncoder))
    #TODO: Remove text column dependency when dealing with newer models\processors.

    def __init__(self, model_id, text_column, models_dir="models", output_label=None, encoder_model=None):
        if not encoder_model:
            raise ValueError(f"ERROR: No encoder model provided to EncoderDeployer.")

        self.model_id = model_id
        self.models_dir = models_dir
        self.model_dir = os.path.join(self.models_dir, self.model_id)
        self.initialize_encoder(encoder_model)
        input_layer = self.encoder.layers[0]
        print(f"UPDATE: Input spec for input layer: {input_layer.input_spec}")
        self.sequence_processor = get_sequence_processor(self.model_dir)
        vocab = self.sequence_processor.tokenizer.word_index
        print(f"UPDATE: Sequence processor vocab = {vocab}.")

        # Default output label prefix of each feature is model_id.
        self.output_label = output_label if output_label else model_id
        self.text_column = text_column
        self.feature_method = self.make_features_from_padded if self.sequence_processor.pad_sequences else self.make_features_from_ragged

    def initialize_encoder(self, encoder_model):
        self.encoder_dir = os.path.join(self.model_dir, "encoder")
        custom_objects = self.ENCODER_TO_CUSTOM_OBJECTS[encoder_model]
        print(f"UPDATE: ModelDeployer loading model from {self.encoder_dir}.")
        self.encoder = load_model(filepath=self.encoder_dir, custom_objects=custom_objects, compile=False)
        print(f"UPDATE: ModelDeployer model summary below.")
        self.encoder.summary()

    def make_features_from_ragged(self, token_sequences):
        '''
        Represent ragged/variable-length token sequences with a features matrix.
        '''

        token_sequences = tf.ragged.stack([cast_tf_tokens(tokens) for tokens in token_sequences])
        feature_matrix = []
        sequence_count = token_sequences.shape[0]
        print(f"UPDATE: Starting deployment tokenization of {sequence_count} sequences.")
        prog_bar = Progbar(target=sequence_count)

        for i, sequence in enumerate(token_sequences):
            features = self.encoder(sequence)
            feature_matrix.append(features)
            prog_bar.update(i + 1)

        feature_matrix = np.asarray(feature_matrix)
        return feature_matrix

    #TODO: Consider moving this and above to SequenceProcessor.
    def make_features_from_padded(self, token_sequences):
        '''
        Pad token sequences and produce features for them with the model. Padding is necessary for transformer encoders.
        '''

        pad_length = self.sequence_processor.max_anchor_length
        token_sequences = pad_sequences(token_sequences, maxlen=pad_length, padding="post")
        #TODO: Optimize this, compare to numpy concatenation.
        feature_matrix = []
        sequence_count = token_sequences.shape[0]
        print(f"UPDATE: Starting deployment tokenization of {sequence_count} sequences.")
        prog_bar = Progbar(target=sequence_count)

        for i, token_sequence in enumerate(token_sequences):
            inputs = tf.cast(token_sequence, tf.int32)
            features = self.encoder(inputs)
            print(f"UPDATE: inputs = {inputs}, features={features} \n")
            feature_matrix.append(features)
            prog_bar.update(i + 1)

        feature_matrix = np.asarray(feature_matrix)
        return feature_matrix

    def make_feature_matrix(self, inputs_df):
        '''
        Use trained model to build a feature matrix from the raw text found in <inputs_df>.
        '''

        # Preprocess the df.
        preprocess_args = dict(df=inputs_df)
        inputs_df = run_with_time(self.sequence_processor.preprocess_df, preprocess_args, "Deployment preprocessing.")

        # Tokenize df.
        tokenize_args = dict(document_df=inputs_df)
        tokenized_df = run_with_time(self.sequence_processor.tokenize_document_df, tokenize_args, "Deployment tokenization")
        token_sequences = tokenized_df["document_tokens"].values
        feature_matrix = self.feature_method(token_sequences)
        return feature_matrix


