import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Progbar
from tensorflow import RaggedTensor

import numpy as np
from numpy import squeeze

from sequence_models import TransformerEncoder, RNNEncoder
from modules.common_funcs import get_sequence_processor, run_with_time, cast_tf_tokens, tokenize_df_wrapper, drop_nan_text, set_path_to_main, \
                         make_nested_dirs

from os.path import join, exists

from pandas import DataFrame, read_csv, concat

class EncoderDeployer():
    ENCODER_TO_CUSTOM_OBJECTS = dict(transformer=dict(transformer_encoder=TransformerEncoder), rnn=dict(rnn_encoder=RNNEncoder))
    CACHED_FEATURES_DIR = "cached_features"
    CACHED_FEATURES_PATH = "cached_<TEXT_COLUMN>_features.csv"
    CACHED_FEATURE_COL_TEMP = "cached_feature_<DIM>"

    def __init__(self, model_id, text_column, encoder_model, models_dir="models", output_label=None):
        self.model_id = model_id
        self.models_dir = models_dir
        self.model_dir = join(self.models_dir, self.model_id)
        self.model_dir = set_path_to_main(self.model_dir)
        self.text_column = text_column
        make_nested_dirs(join(self.model_dir, self.CACHED_FEATURES_DIR))
        self.CACHED_FEATURES_PATH = self.CACHED_FEATURES_PATH.replace("<TEXT_COLUMN>", self.text_column)
        self.cached_features_path = join(self.model_dir, self.CACHED_FEATURES_DIR, self.CACHED_FEATURES_PATH)
        self.load_cached_features()
        self.initialize_encoder(encoder_model)
        self.embedding = self.encoder.layers[0]
        print(f"UPDATE: Embedding layer config: {self.embedding.get_config()}")
        self.sequence_processor = get_sequence_processor(self.model_dir)

        # Default output label prefix of each feature is model_id.
        self.output_label = output_label if output_label else model_id
        self.prepare_method = self.prepare_padded_sequences if self.sequence_processor.pad_sequences else self.prepare_ragged_sequences

        # Load sub-models\layers.
        self.embedding_layer = self.encoder.layers[0]
        self.attention_layer = self.encoder.layers[1]
        self.encoder_layer = self.encoder.layers[-1]

        #TODO: Simplify using Transformer encoder config without resorting to layer configs.
        self.encoding_dims = self.get_encoding_dims(self.attention_layer)
        self.cached_feature_cols = [self.CACHED_FEATURE_COL_TEMP.replace("<DIM>", str(dim)) for dim in range(self.encoding_dims)]
        print(f"UPDATE: EncoderDeployer has encoding dims = {self.encoding_dims}.")

    @staticmethod
    def get_encoding_dims(encoder):
        encoder_config = encoder.get_config()

        if "value_dim" in encoder_config:
            encoding_dims = encoder_config["value_dim"]
        elif "units" in encoder_config:
            encoding_dims = encoder_config["units"]
        else:
            raise KeyError(f"KEY ERROR: Encoder config is missing 'value_dim' and 'units' keys. One is required.")

        return encoding_dims

    def load_cached_features(self):
        if exists(self.cached_features_path):
            self.cached_features_df = read_csv(self.cached_features_path)
            self.cached_features = set(self.cached_features_df[self.text_column].unique())
        else:
            self.cached_features_df = DataFrame()
            self.cached_features = set([])

    def initialize_encoder(self, encoder_model):
        self.encoder_dir = join(self.model_dir, "encoder")
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

    def extract_features_vector(self, cached_key):
        cached_features_df = self.cached_features_df[self.cached_features_df[self.text_column] == cached_key]
        feature_vector = cached_features_df[self.cached_feature_cols].values
        return feature_vector

    def load_features_from_cache(self, cached_key):
        cached_features = self.extract_features_vector(cached_key)
        cached_features = squeeze(cached_features)

        if len(cached_features.shape) == 2:
            print(f"WARNING: Multiple cached features for key = {cached_key}. Taking first.")
            cached_features = squeeze(cached_features[0, :])

        return cached_features

    def make_features_for_sequence(self, token_sequence, cached_key):
        if cached_key in self.cached_features:
            cached_features = self.load_features_from_cache(cached_key=cached_key)
            return cached_features

        sequence = tf.cast(token_sequence, tf.int32)
        embedding = self.embedding_layer(sequence)
        embedding = tf.expand_dims(embedding, axis=0) if tf.rank(embedding) == 2 else embedding
        features = self.encoder_layer(embedding).numpy()
        features = np.squeeze(features) if tf.rank(features) == 3 else features
        #TODO: Load sequence summarization.
        average_features = np.mean(features, axis=0)
        return average_features

    def make_features_from_sequences(self, token_sequences, cached_keys):
        '''
        Represent ragged/variable-length token sequences with a features matrix.
        '''

        feature_matrix = []
        sequence_count = len(token_sequences)
        print(f"UPDATE: Starting feature building for {sequence_count} sequences.")
        prog_bar = Progbar(target=sequence_count, unit_name="sequence")

        for i, sequence in enumerate(token_sequences):
            #TODO: Load sequence summarization.
            average_features = self.make_features_for_sequence(sequence, cached_keys[i])
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

    @staticmethod
    def remove_empty_seqs(token_sequences, cached_keys):
        nonempty_inds = [i for i, token_seq in enumerate(token_sequences) if len(token_seq)]
        nonempty_count = len(nonempty_inds)
        empty_count = len(token_sequences) - nonempty_count if not isinstance(token_sequences, RaggedTensor) else \
                        max([i for i, token_seq in enumerate(token_sequences)])

        if empty_count:
            print(f"WARNING: Removing {empty_count} empty token sequences and cached keys.")

        token_sequences = [token_sequences[i] for i in nonempty_inds]
        cached_keys = [cached_keys[i] for i in nonempty_inds]
        return token_sequences, cached_keys

    def update_cache(self, feat_mat, keys):
        feat_rows = feat_mat.tolist()
        keys_feat_rows = [[key, *feat_row] for feat_row, key in zip(feat_rows, keys)]
        new_feats_df = DataFrame(keys_feat_rows, columns=[self.text_column, *self.cached_feature_cols])
        self.cached_features_df = concat([self.cached_features_df, new_feats_df])
        print(f"UPDATE: Saving new cached features to {self.cached_features_path}.")
        self.cached_features_df.to_csv(self.cached_features_path, index=False)

    def get_new_feats_keys(self, feature_matrix, cached_keys):
        uncached_feature_inds = [i for i, key in enumerate(cached_keys) if key not in self.cached_features]
        print(f"UPDATE: Found {len(uncached_feature_inds)} uncached feature inds. ")
        uncached_keys = [cached_keys[i] for i in uncached_feature_inds]
        uncached_feature_mat = feature_matrix[uncached_feature_inds, :]
        print(f"UPDATE: Built uncached feature matrix with shape = {uncached_feature_mat.shape}. ")
        return uncached_feature_mat, uncached_keys

    def make_feature_matrix(self, inputs_df):
        '''
        Use trained model to build a feature matrix from the raw text found in <inputs_df>.
        '''

        # Preprocess the df.
        inputs_df = drop_nan_text(df=inputs_df, text_column=self.text_column)

        # Tokenize df.
        tokenized_df = tokenize_df_wrapper(sequence_processor=self.sequence_processor, document_df=inputs_df, text_column=self.text_column)
        token_sequences = tokenized_df["document_tokens"].values

        # Text column values used for caching\looking up features.
        cached_keys = tokenized_df[self.text_column].values
        prepared_sequences = self.prepare_method(token_sequences)
        prepared_sequences, cached_keys = self.remove_empty_seqs(prepared_sequences, cached_keys)
        feature_matrix = self.make_features_from_sequences(prepared_sequences, cached_keys)

        # Get new features and their keys for updating the cache. Then update the cache with these new vals.
        uncached_feats, uncached_keys = self.get_new_feats_keys(feature_matrix, cached_keys)
        self.update_cache(uncached_feats, uncached_keys)
        return feature_matrix

    #TODO: Improve training data scheme by i) creating a dir,
    #                                      ii) checking and filtering for redundancies during training,
    #                                     iii) join training data into single csv during calls like this.
    def get_training_data(self):
        training_data_path = join(self.model_dir, "training_data.csv")

        if not exists(training_data_path):
            raise FileNotFoundError(f"ERROR: Training data in {training_data_path} not found!")

        training_data = read_csv(training_data_path)
        return training_data

if __name__ == "__main__":
    encoder_deployer = EncoderDeployer(model_id="test", text_column="code", encoder_model="rnn")
