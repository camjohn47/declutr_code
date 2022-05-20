import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import make_ndarray
from tensorflow.keras.utils import Progbar

import numpy as np

from sequence_models import DeClutrContrastive, TransformerEncoder, RNNEncoder

from common_funcs import get_sequence_processor, run_with_time, cast_tf_tokens, tokenize_df_wrapper, drop_nan_text, set_path_to_main

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
        self.model_dir = set_path_to_main(self.model_dir)
        self.initialize_encoder(encoder_model)
        self.embedding = self.encoder.layers[0]
        print(f"UPDATE: Embedding layer config: {self.embedding.get_config()}")
        self.sequence_processor = get_sequence_processor(self.model_dir)

        # Default output label prefix of each feature is model_id.
        self.output_label = output_label if output_label else model_id
        self.text_column = text_column
        self.prepare_method = self.prepare_padded_sequences if self.sequence_processor.pad_sequences else self.prepare_ragged_sequences

    def initialize_encoder(self, encoder_model):
        self.encoder_dir = os.path.join(self.model_dir, "encoder")
        custom_objects = self.ENCODER_TO_CUSTOM_OBJECTS[encoder_model]
        print(f"UPDATE: ModelDeployer loading model from {self.encoder_dir}.")
        self.encoder = load_model(filepath=self.encoder_dir, custom_objects=custom_objects, compile=False)
        print(f"UPDATE: ModelDeployer model summary below.")
        self.encoder.summary()

    def prepare_ragged_sequences(self, token_sequences):
        ragged_sequences = tf.ragged.stack([cast_tf_tokens(tokens) for tokens in token_sequences])
        return ragged_sequences

    def prepare_padded_sequences(self, token_sequences):
        pad_length = self.sequence_processor.max_anchor_length
        padded_sequences = pad_sequences(token_sequences, maxlen=pad_length, padding="post")
        return padded_sequences

    def make_features_from_sequences(self, token_sequences):
        '''
        Represent ragged/variable-length token sequences with a features matrix.
        '''

        feature_matrix = []
        sequence_count = token_sequences.shape[0]
        print(f"UPDATE: Starting deployment {sequence_count} sequences.")
        prog_bar = Progbar(target=sequence_count, unit_name="sequence")
        embedding_layer = self.encoder.layers[0]
        encoder_layer = self.encoder.layers[-1]

        for i, sequence in enumerate(token_sequences):
            if not len(sequence):
                continue

            sequence = tf.cast(sequence, tf.int32)
            embedding = embedding_layer(sequence)
            embedding = tf.expand_dims(embedding, axis=0) if tf.rank(embedding) == 2 else embedding

            if len(embedding) == 0:
                print(f"WARNING: Empty embedding = {embedding}")
                continue

            try:
                features = encoder_layer(embedding).numpy()
            except:
                print(f"WARNING: Failed to build features for {embedding}")
                continue

            features = np.squeeze(features) if tf.rank(features) == 3 else features

            if not features.shape[0]:
                print(f"UPDATE: Empty feature for embedding = {embedding}")
                continue

            #TODO: Load sequence summarization.
            average_features = np.mean(features, axis=0)

            if not average_features.shape:
                print(f"UPDATE: Empty mean feature for embedding = {embedding}, features = {features}")
                continue

            feature_matrix.append(average_features)
            prog_bar.update(i + 1)

        feature_matrix = np.stack(feature_matrix)
        return feature_matrix

    #TODO: Consider moving this and above to SequenceProcessor.
    def make_features_from_padded(self, token_sequences):
        '''
        Pad token sequences and produce features for them with the model. Padding is necessary for transformer encoders.

        Outputs
        A (N x feature_dims) numpy array. Each i-th row describes the i-th sequence with a <feature_dims>-dimensional vector.
        '''

        #TODO: Optimize this, compare to numpy concatenation.
        feature_matrix = []
        sequence_count = token_sequences.shape[0]
        print(f"UPDATE: Building features for {sequence_count} sequences.")
        prog_bar = Progbar(target=sequence_count)

        for i, token_sequence in enumerate(token_sequences):
            inputs = tf.cast(token_sequence, tf.int32)
            features = self.encoder(inputs)
            feature_matrix.append(features)
            prog_bar.update(i + 1)

        feature_matrix = np.stack(feature_matrix)
        return feature_matrix

    def make_feature_matrix(self, inputs_df):
        '''
        Use trained model to build a feature matrix from the raw text found in <inputs_df>.
        '''

        # Preprocess the df.
        preprocess_args = dict(df=inputs_df, text_column=self.text_column)
        inputs_df = run_with_time(drop_nan_text, preprocess_args, "Deployment preprocessing.")

        # Tokenize df.
        tokenize_args = dict(sequence_processor=self.sequence_processor, document_df=inputs_df, text_column=self.text_column)
        tokenized_df = run_with_time(tokenize_df_wrapper, tokenize_args, f"deployment {self.text_column} tokenization")
        token_sequences = tokenized_df["document_tokens"].values
        prepared_sequences = self.prepare_method(token_sequences)
        feature_matrix = self.make_features_from_sequences(prepared_sequences)
        return feature_matrix


