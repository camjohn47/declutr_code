import sys
import os

from pathlib import Path

from common_funcs import get_rank
from tensor_visualizer import TensorVisualizer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, MultiHeadAttention, LSTM, GRU, GlobalAvgPool1D, Softmax, \
                                    Dense, LayerNormalization, Dropout, LeakyReLU, Dot
from tensorflow.keras.layers.experimental import EinsumDense
from tensorflow.keras.models import load_model

tf.executing_eagerly()

class PWFF(Layer):
    '''
    Simple point-wise feed forward neural network implemented as a Keras layer. It linearly maps inputs to a <units>-dimensional
    space, after which leaky relu activation is applied.
    '''
    def __init__(self, units=100):
        super().__init__()
        self.units = units
        self.dense = Dense(units=self.units)
        self.leaky_relu_activation = LeakyReLU()

    def get_config(self):
        config = super().get_config()
        config['output_dim'] = self.units
        return config

    def call(self, inputs):
        transformed_inputs = self.dense(inputs)
        transformed_inputs = self.leaky_relu_activation(transformed_inputs)
        return transformed_inputs

#TODO: Experiment with Transformer XL architecture for sequence-level recurrence and longer context reception.
class TransformerEncoder(Model):
    self_attention_args = dict(num_heads=6, key_dim=100)
    dropout_args = dict(rate=.05)

    def __init__(self, input_dim, self_attention_args={}, normalization_args={}, dropout_args={}):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = Embedding(input_dim=self.input_dim, output_dim=100)
        self.self_attention_args.update(self_attention_args)
        self.self_attention = MultiHeadAttention(**self.self_attention_args)
        self.normalization_args = normalization_args
        self.normalization = LayerNormalization(**self.normalization_args)
        self.dropout_args.update(dropout_args)
        self.dropout = Dropout(**self.dropout_args)
        self.output_dims = self.self_attention_args["key_dim"]

    def get_config(self):
        config = super().get_config()
        config["input_dim"] = self.input_dim
        config['self_attention_args'] = self.self_attention_args
        config['normalization_args'] = self.normalization_args
        return config

    def call(self, inputs):
        '''
        inputs (Tensor): An integer token tensor describing text in a sequence. Can be one of the following shapes\types:
                         1. (sequence length) Tensor
                         2. (batch x None)-shaped RaggedTensor with different lengths in the outer dimension associated with
                            different text lengths.
        '''

        inputs = self.embedding(inputs)

        if get_rank(inputs) == 2:
            inputs = tf.expand_dims(inputs, axis=0)
        elif not get_rank(inputs):
            #print(f'WARNING: Empty inputs tensor fed to TransformerEncoder. ')
            return tf.zeros(self.output_dims)

        attention_encodings = self.self_attention(query=inputs, value=inputs, key=inputs)
        attention_encodings = self.dropout(attention_encodings)
        normalized_encodings = self.normalization(attention_encodings)
        return normalized_encodings

class TransformerDecoder(Layer):
    '''
    A layer representing the decoder of a simple Transformer architecture.
    '''

    embedding_args = dict(output_dim=100)
    self_attention_args = dict(num_heads=6, key_dim=100)
    encoder_attention_args = dict(num_heads=6, key_dim=100)
    dropout_args = dict(rate=.1)

    def __init__(self, input_dim, self_attention_args={}, normalization_args={}, dropout_args={}, encoder_attention_args={}):
        super().__init__()
        self.embedding_args["input_dim"] = input_dim
        self.embedding = Embedding(**self.embedding_args)
        self.self_attention_args.update(self_attention_args)
        self.self_attention = MultiHeadAttention(**self.self_attention_args)
        self.encoder_attention_args.update(encoder_attention_args)
        self.encoder_output_attention = MultiHeadAttention(**self.encoder_attention_args)
        self.pwff = PWFF()
        self.normalization_args = normalization_args
        self.normalization = LayerNormalization(**self.normalization_args)
        self.dropout_args.update(dropout_args)
        self.dropout = Dropout(**self.dropout_args)

    def get_config(self):
        config = super().get_config()
        config['self_attention_args'] = self.self_attention_args
        config['normalization_args'] = self.normalization_args
        return config

    def call(self, inputs, encoder_outputs):
        '''
        Inputs
        inputs: A (batch size x max seq length) int tensor containing padded tokens of each sequence.
        encoder_outputs: A (batch size x output
        '''

        inputs = self.embedding(inputs)
        inputs = tf.expand_dims(inputs, axis=0) if get_rank(inputs) == 2 else inputs
        self_attention_encodings = self.self_attention(query=inputs, value=inputs, key=inputs)
        self_attention_encodings = self.dropout(self_attention_encodings)
        self_attention_encodings = self.normalization(self_attention_encodings)

        attention_encodings = self.encoder_output_attention(query=encoder_outputs, value=encoder_outputs, key=self_attention_encodings)
        attention_encodings = self.pwff(attention_encodings)
        normalized_encodings = self.normalization(attention_encodings)
        normalized_encodings = self.dropout(normalized_encodings)
        return normalized_encodings

class Transformer(Model):
    '''
    Tf model that represents a basic transformer neural network. It has an encoder-decoder architecture that uses self-attention
    and linear transformations to encode sequences. This encoder
    '''

    output_dense_args = dict()

    def __init__(self, input_dim, output_dim=100, encoder_args={}, decoder_args={}, output_dense_args={}):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim=input_dim, **encoder_args)
        self.decoder = TransformerDecoder(input_dim=input_dim, **decoder_args)
        self.output_dense_args.update(output_dense_args)
        self.units = input_dim
        self.output_dense_args["units"] = output_dim
        self.output_dense = Dense(**self.output_dense_args)
        self.output_softmax = Softmax()

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        return config

    def call(self, inputs):
        if inputs.shape[0] == None:
            print(f'WARNING: Transformer input has no length! Returning zeros. Inputs: {inputs}')
            return tf.zeros(self.units)

        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(inputs, encoder_outputs)
        transformed_outputs = self.output_dense(decoder_outputs)
        probabilities = self.output_softmax(transformed_outputs)
        return probabilities

class RNNEncoder(Model):
    '''
    Wrapper for an RNN encoder.
    '''

    ARCHITECTURE_TO_MODEL = dict(lstm=LSTM, gru=GRU)
    model_args = dict(units=100)

    def __init__(self, input_dim, embedding_dims=100, architecture='lstm', model_args={}):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims
        embedding_args = dict(input_dim=self.input_dim, output_dim=self.embedding_dims)
        self.embedding_args = embedding_args
        self.embedding = Embedding(**self.embedding_args)
        print(f'UPDATE: Embedding config = {self.embedding.get_config()}.')

        self.model_args.update(model_args)
        # Returns the entire sequence of hidden states, rather than just the most recent one.
        self.model_args['return_sequences'] = True

        if architecture not in self.ARCHITECTURE_TO_MODEL:
            print(f"ERROR: Requested RNNEncoder architecture = {architecture} unavailable.")
            sys.exit(1)

        self.architecture = architecture
        self.model = self.ARCHITECTURE_TO_MODEL[self.architecture]
        self.model = self.model(**self.model_args)

        # Dimensions of LSTMEncoder's final output vector.
        self.output_dims = self.model_args["units"]

    def get_config(self):
        config = super().get_config()
        config['embedding_args'] = self.embedding_args
        config['model_args'] = self.model_args
        config["input_dim"] = self.input_dim
        config["embedding_dims"] = self.embedding_dims
        config["output_dims"] = self.output_dims
        config["architecture"] = self.architecture
        return config

    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            return cls(**config)
        else:
            print(f'WARNING: Config passed to RNNEncoder from_config is not a dictionary! {config}')
            return cls({})

    def call(self, inputs):
        embedded_inputs = self.embedding(inputs)

        if get_rank(embedded_inputs) == 2:
            embedded_inputs = tf.expand_dims(embedded_inputs, axis=0)

        elif not get_rank(embedded_inputs):
            print(f'WARNING: Flat embedded inputs fed to RNN layer! {embedded_inputs}')
            empty_outputs = tf.zeros(self.output_dims)
            return empty_outputs

        encoded_inputs = self.model(embedded_inputs)
        anchor_needs_reduction = get_rank(encoded_inputs) == 3 and not isinstance(inputs, tf.RaggedTensor)
        encoded_inputs = tf.squeeze(encoded_inputs, axis=0) if anchor_needs_reduction else encoded_inputs
        return encoded_inputs

class DeClutrContrastive(Model):
    '''
    Implementation of the DeClutr approach for self-supervised sentence and document embeddings:https://arxiv.org/abs/2006.03659.
    The main learning objective is to identify the associated document with an anchor document using their
    tokenized sequences. The associated document is called a positive sample, and is fed to the model
    with other tokenized text sequences (= negative samples) that have to be correctly distinguished.
    Because all positive and negative samples are algorithmically labeled for each anchor, the algorithm
    is self-supervised.

    The contrastive loss function, combined with DeClutr's nearby anchor positive sampling strategy,
    encourages aligned orientation and location for similar documents in the embedding space. Similarly,
    negative sample predictions are penalized. Embeddings between dissimilar documents
    separate and their angles diverge when they're incorrectly classified as a sequence pair.

    Each span begins with a randomly sampled starting point and randomly sampled length. Positive sampling
    covers a diverse range of three distinct positive sample types: contained, partially overlapped,
    and adjacent. Both hard and soft negative samples are included. Hard negative samples are from the
    same document, and soft negative samples are from different documents. The result is a comprehensive,
    continuous output representation.
    '''

    sequence_structure_args = dict()
    encoder_args = dict(units=100)
    SUPPORTED_ENCODERS = {'rnn': RNNEncoder, 'embedding': Embedding, 'transformer': TransformerEncoder}
    RNN_ENCODER_ARGS = dict(model_args=dict(units=100, return_sequences=True))
    TRANSFORMER_ARGS = dict()
    SUPPORTED_ENCODER_ARGS = {'rnn': RNN_ENCODER_ARGS, 'embedding': dict(output_dim=100), 'transformer': TRANSFORMER_ARGS}

    # Sequence summarization layers used to aggregate information along the time dimension. For example, each script
    # can have a variable amount of code. This layer determines how a sequence's word embeddings are summarized to produce
    # a single vector.
    MEAN_SUMMARIZATION = GlobalAvgPool1D()
    SEQUENCE_SUMMARIZATION_LAYERS = dict(average=GlobalAvgPool1D(), lstm=LSTM(return_sequences=False, units=100),
                                         gru=GRU(return_sequences=False, units=100))

    def __init__(self, batch_size, pretrained_encoder=None, pretrained_encoder_name=None, encoder_config={},
                 encoder_model='rnn', input_dim=None, models_directory="models", model_id='test', visualize_tensors=False,
                 sequence_summarization="average"):

        super().__init__()
        self.pretrained_encoder_name = pretrained_encoder_name

        if encoder_model not in self.SUPPORTED_ENCODERS:
            raise ValueError(f'ERROR: Unsupported encoder model {encoder_model} requested. Supported encoder models: '
                             f'{self.SUPPORTED_ENCODERS}')

        self.encoder_config = self.SUPPORTED_ENCODER_ARGS[encoder_model]
        self.encoder_config.update(encoder_config)
        self.encoder_config["input_dim"] = input_dim
        self.encoder, self.encoder_model = self.initialize_encoder(pretrained_encoder, encoder_model)

        # Pooler takes average over the embedding dimension.
        self.pooler = GlobalAvgPool1D(data_format='channels_last')
        self.batch_size = batch_size
        self.einsum_layer = self.build_einsum_layer()
        self.dot_product = Dot(axes=[1, 1], normalize=True)
        self.softmax = Softmax(axis=0)
        self.model_id = model_id
        self.build_directory(models_directory)
        self.visualize_tensors = visualize_tensors
        self.tensor_visualizer = TensorVisualizer(tf_model_dir=self.model_dir, num_axes=2) if self.visualize_tensors else None
        self.__sequence_summarization = self.get_sequence_summarization(sequence_summarization)
        print(f'UPDATE: Sequence summarization = {self.__sequence_summarization}')

    def build_directory(self, model_directory):
        self.model_dir = os.path.join(model_directory, self.model_id)
        print(f'UPDATE: Creating model directory {self.model_dir}.')
        Path(self.model_dir).mkdir(exist_ok=True, parents=True)

    def initialize_encoder(self, pretrained_encoder, encoder_model):
        if pretrained_encoder and not self.pretrained_encoder_name:
            raise ValueError(f'ERROR: Pre-trained encoder provided without name. ')

        if pretrained_encoder:
            encoder = pretrained_encoder
        else:
            encoder = self.SUPPORTED_ENCODERS[encoder_model](**self.encoder_config)

        return encoder, encoder_model

    def get_sequence_summarization(self, value):
        if value not in self.SEQUENCE_SUMMARIZATION_LAYERS:
            raise ValueError(f"ERROR: Requested unsupported sequence summarization {value}.")

        sequence_summarization = self.SEQUENCE_SUMMARIZATION_LAYERS[value]
        return sequence_summarization

    @property
    def sequence_summarization(self):
        return self.__sequence_summarization

    @sequence_summarization.setter
    def sequence_summarization(self, value):
        self.value = self.get_sequence_summarization(value)
        print(f'UPDATE: Set sequence summarization = {self.__sequence_summarization}')

    def build_einsum_layer(self):
        einsum_layer = EinsumDense(equation='k,ik->i', output_shape=(self.batch_size))
        return einsum_layer

    def get_config(self):
        config = super().get_config()
        config['embedding_args'] = self.embedding_args
        config['encoder_args'] = self.encoder_args
        config['decoder_args'] = self.decoder_args
        config['model_id'] = self.model_id
        config['batch_size'] = self.batch_size
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    def anchor_encoder_helper(self, anchor):
        '''
        Prepare anchor sequence for anchor encoding and then squeeze resulting encoding if it's 3D.
        '''

        if self.pretrained_encoder_name == 'roberta-base' and get_rank(anchor) == 1:
            anchor = tf.expand_dims(anchor, axis=0)

        anchor = anchor.numpy() if self.pretrained_encoder_name else anchor
        anchor_encoding = self.encoder(anchor)
        anchor_encoding = tf.squeeze(anchor_encoding, axis=0) if get_rank(anchor_encoding) == 3 else anchor_encoding
        return anchor_encoding

    def sequences_encoder_helper(self, contrasted_seqs):
        if self.pretrained_encoder_name == 'roberta-base' and get_rank(contrasted_seqs) == 3:
            contrasted_seqs = tf.squeeze(contrasted_seqs, axis=0)

        if self.pretrained_encoder_name:
            contrasted_seqs = contrasted_seqs.numpy()

        contrasted_seqs_encoding = self.encoder(contrasted_seqs)
        return contrasted_seqs_encoding

    def summarize_sequence(self, sequence):
        '''
        Summarize the sequence (should be a float tensor) along its time axis.
        '''

        # Expand tensor to have a batch axis for anchor sequence. This ensures proper mapping by summarization layer.
        sequence = tf.expand_dims(sequence, axis=0) if get_rank(sequence) == 2 else sequence
        summary = self.sequence_summarization(sequence)
        return summary

    def record_visual_outputs(self, output_probs):
        '''
        Save the outputs to the TensorVisualizer for building visuals, if specified.
        '''
        if self.tensor_visualizer:
            self.tensor_visualizer.record_training_outputs(output_probs, [])

    def call(self, inputs, *kwargs):
        '''
        Return output probabilities of DeclutrModel from mini-batch of document sequences.

        Inputs
        inputs: A dictionary containing an anchor sequence tensor and a contrasted sequences tensor.

        Outputs
        positive_sequence_probs (Tensor): Tensor containing the probabilities of each contrasted sequence
                                          being the positive sample for the anchor sequence.
        '''

        anchor = inputs['anchor_sequence']

        # Can happen at epoch start with tf 2.9 during warm up.
        if anchor.shape[0] == None:
            return tf.zeros(self.batch_size)

        contrasted_sequences = inputs['contrasted_sequences']

        # Can happen at epoch start with tf 2.9 during warm up.
        if contrasted_sequences.shape[0] == None:
            return tf.zeros(self.batch_size)

        encoded_anchor = self.anchor_encoder_helper(anchor)
        summarized_anchor = self.summarize_sequence(encoded_anchor)
        summarized_anchor = tf.squeeze(summarized_anchor)
        encoded_contrasted_sequences = self.sequences_encoder_helper(contrasted_sequences)

        # TODO: DeClutr summarizes each document's embedding with the average of its word embeddings.
        #       This has some drawbacks, such as ignoring word order and assuming equally important words.
        #       Experiment with RNN's and attention layers for improved sequence encodings.
        summarized_sequences = self.summarize_sequence(encoded_contrasted_sequences)

        # Project each contrasted sequence encoding onto anchor encoding. These values should correlate with
        # similarity, and will be fed into final softmax layer for positive sequence probabilities.
        scores = tf.einsum('k,ik->i', summarized_anchor, summarized_sequences)

        # Probabilities of each input sequence in the batch being the positive sample for the anchor.
        positive_sequence_probs = self.softmax(scores)
        positive_sequence_probs = tf.expand_dims(positive_sequence_probs, axis=0)
        self.record_visual_outputs(positive_sequence_probs)
        return positive_sequence_probs

class DeclutrMaskedLanguage(DeClutrContrastive):
    '''
    Generative version of DeclutrContrastive with a MMM learning objective. It tries to predict the words associated
    with masked out tokens from a sequence using its context.
    '''

    def __init__(self, masked_vocabulary_size, masked_token, pretrained_encoder_path=None, **declutr_args):
        self.declutr_args = declutr_args
        super().__init__(**self.declutr_args)
        self.masked_vocabulary_size = masked_vocabulary_size

        # Dense layer that expresses each masked word's probability as a linear combination of
        # the masked word embeddings.
        self.masked_vocabulary_dense = Dense(units=self.masked_vocabulary_size)

        # Integer denoting an input token has been masked.
        self.masked_token = masked_token
        self.default_output = [0 for i in range(self.masked_vocabulary_size)]

        if pretrained_encoder_path:
            encoder = self.extract_encoder(pretrained_encoder_path)
            self.encoder = encoder if encoder else self.encoder

    def load_model(self, model_path):
        print(f'UPDATE: Loading pre-trained encoder from {model_path}.')
        try:
            model = load_model(model_path)
            return model
        except:
            print(f'WARNING: Encoder failed to load from {model_path}! Building new one from scratch.')
            return None

    def extract_encoder(self, model_path):
        model = self.load_model(model_path)
        encoder = model.encoder if model else None
        return encoder

    def get_config(self):
        config = super().get_config()
        config['masked_vocabulary_size'] = self.masked_vocabulary_size
        config['masked_token'] = self.masked_token
        return config

    def find_masked_indices(self, tensor):
        masked_indices = tf.where(tensor == self.masked_token).numpy()
        masked_shape = masked_indices.shape
        masked_indices = masked_indices.tolist() if masked_shape != (1, 1) else masked_indices.tolist()[0]
        return masked_indices

    def gather_masked_embeddings(self, masked_sequence, embeddings):
        '''
        Inputs
        masked_sequence (Tensor): Original masked token sequence from which embeddings were made.
        embeddings (sequence length x embedding dim): Token embeddings of an anchor sequence.

        Outputs
        A (N x embedding dims)-shaped tensor with the embedding vectors for the N masked out tokens.
        '''

        masked_indices = self.find_masked_indices(masked_sequence)
        masked_embeddings = tf.gather(embeddings, masked_indices, axis=0)

        if get_rank(masked_embeddings) == 3:
            masked_embeddings = tf.squeeze(masked_embeddings, axis=1)

        return masked_embeddings

    def call(self, inputs):
        '''
        Return output word probabilities of a masked token sequence.

        Inputs
        inputs: A masked out, tokenized anchor span.
        '''

        masked_anchor = inputs

        if not masked_anchor.shape[0]:
            print(f'WARNING: MLM called on empty masked anchor. Returning zeros.')
            return self.default_output

        embedded_anchor = self.anchor_encoder_helper(masked_anchor)
        masked_embeddings = self.gather_masked_embeddings(masked_anchor, embedded_anchor)
        masked_vocabulary_probs = self.masked_vocabulary_dense(masked_embeddings)
        return masked_vocabulary_probs










