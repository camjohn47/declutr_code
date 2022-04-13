import sys

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow_probability.python.distributions import Beta, Sample

import numpy as np

import random

from functools import partial

class SequenceProcessor():
    # Default beta parameters taken from the original DeClutr paper. concentration1 = alpha, 0 = beta.
    anchor_args = dict(concentration1=4, concentration0=2)
    positive_sampling_args = dict(concentration1=2, concentration0=4)

    def __init__(self, tokenizer_args={}, min_anchor_length=32, max_anchor_length=112, anchor_args={},
                 positive_sampling_args={}, anchors_per_document=1, num_positive_samples=1, documents_per_batch=32,
                 chunk_size=int(1.0e3), max_document_length=512, pretrained_tokenizer=False, tokenizer_type=None,
                 tokenizer=None, sample_documents_with_replacement=False, pad_sequences=False):
        self.min_anchor_length = min_anchor_length
        self.max_anchor_length = max_anchor_length

        # This is chosen to allow for maximally extended positive sequences with maximum length and maximum start,
        # occuring after a maximally long anchor span.
        self.min_document_length = 2 * max_anchor_length
        self.max_document_length = max_document_length
        self.documents_per_batch = documents_per_batch
        self.anchors_per_document = anchors_per_document
        self.num_positive_samples = num_positive_samples

        # Anchors are the base document subset used in the declutr framework. For each batch of
        # input sequences to the model, positive and negative samples are defined with respect to anchors.
        # As in the paper, anchor length is sampled within the range using a beta distribution.
        self.anchor_args.update(anchor_args)
        self.anchor_length_distribution = Beta(**self.anchor_args)
        self.positive_sampling_args.update(positive_sampling_args)
        self.positive_sample_distribution = Beta(**self.positive_sampling_args)
        self.chunk_size = chunk_size
        self.batch_size = 2 * self.documents_per_batch * self.anchors_per_document - 1

        # Initialize tokenizer depending on whether a pre-trained one is used.
        self.pretrained_tokenizer = pretrained_tokenizer
        self.tokenizer_args = tokenizer_args
        self.tokenizer = self.initialize_tokenizer(tokenizer, tokenizer_type)
        self.tokenizer_type = tokenizer_type
        self.sample_documents_with_replacement = sample_documents_with_replacement
        self.pad_sequences = pad_sequences

    def initialize_tokenizer(self, tokenizer, tokenizer_type):
        # If pre-trained tokenizer is specified, the type and tokenizer itself must be provided.
        if self.pretrained_tokenizer:
            if not tokenizer_type:
                print(f'ERROR: Pretrained tokenizer requested but type unspecified! ')
                sys.exit(1)
            elif not tokenizer:
                print(f'ERROR: Pretrained tokenizer requested but no tokenizer provided! ')
                sys.exit(1)

            tokenizer = tokenizer
        # Otherwise, build a new Keras tokenizer.
        else:
            tokenizer = Tokenizer(**self.tokenizer_args)

        return tokenizer

    def generate_df_in_chunks(self, df, documents_per_chunk=None):
        '''
        Process the document df in smaller individual chunks for improved performance. This uses
        attribute <documents_per_batch> to determine the size of each chunk.
        '''

        # Default documents per chunk is set to be documents per batch.
        documents_per_chunk = documents_per_chunk if documents_per_chunk else self.documents_per_batch
        document_count = len(df)
        chunk_count = int(np.ceil(document_count / documents_per_chunk))
        get_chunk_end = lambda i: min(document_count, (i + 1) * documents_per_chunk)
        chunk_index_ranges = [range(documents_per_chunk * i, get_chunk_end(i)) for i in range(chunk_count)]

        for chunk_index_range in chunk_index_ranges:
            if (documents_per_chunk == self.documents_per_batch) and (chunk_index_range[-1] - chunk_index_range[0] + 1) < documents_per_chunk:
                print(f'WARNING: Incomplete chunk provided, skipping for generation.')
                continue

            df_chunk = df.take(chunk_index_range)
            yield df_chunk

    def generate_df_chunks(self, df, chunk_count):
        '''
        Yield the df in a fixed amount of chunks by taking random samples of the input df.
        '''

        document_count = len(df)

        for chunk in range(chunk_count):
            df_chunk = df.sample(n=self.documents_per_batch)
            yield df_chunk

    def fit_tokenizer_on_documents(self, document_df):
        documents = document_df['document'].values

        if self.pretrained_tokenizer:
            pass
        else:
            document_df['document_tokens'] = self.tokenizer.fit_on_texts(documents)

        return document_df

    def fit_tokenizer_in_chunks(self, document_df):
        for document_df_chunk in self.generate_df_in_chunks(document_df):
            self.fit_tokenizer_on_documents(document_df_chunk)

    def build_anchor_sequence_inds(self, document_tokens):
        '''
        Choose an anchor span for representing a document's token sequence.
        '''

        token_count = len(document_tokens)
        anchor_length_prob = self.anchor_length_distribution.sample([1])
        anchor_length = int(anchor_length_prob * (self.max_anchor_length - self.min_anchor_length)) + self.min_anchor_length
        start_domain = list(range(token_count - anchor_length))
        #print(f'UPDATE: Start domain = {start_domain}, length = {anchor_length}, tc = {token_count}')
        anchor_start = random.choice(start_domain)
        anchor_range = range(anchor_start, anchor_start + anchor_length)
        anchor_sequence = [document_tokens[i] for i in anchor_range]
        anchor_sequence = tf.cast(anchor_sequence, tf.int32)
        return anchor_sequence, anchor_range

    def build_positive_sequence(self, document_tokens, anchor_range):
        '''
        Build a positive sequence for a document with respect to an anchor span. Length of which is
        sampled with a beta distribution. Sequence starting point is uniformly sampled.

        :param document_tokens:
        :param anchor_range:
        :return:
        '''

        positive_length_prob = self.positive_sample_distribution.sample([1])

        # NOTE: Difference from sampling method described in DeClutr paper: Positive length is restricted
        # to a subsumed view in the event that anchor end > (document length - max span length). This
        # prevents the positive span from exceeding the document's bounds.

        # Sample the length of the positive span.
        proposed_positive_length = int(positive_length_prob * (self.max_anchor_length - self.min_anchor_length)) + self.min_anchor_length
        anchor_start, anchor_end = anchor_range[0], anchor_range[-1]
        max_positive_length = len(document_tokens) - anchor_end
        positive_length = min(proposed_positive_length, max_positive_length)
        positive_start_domain = list(range(anchor_start - positive_length, anchor_end))

        # Sample starting point of the positive span.
        positive_start = random.choice(positive_start_domain)
        positive_domain = range(positive_start, positive_start + positive_length)
        positive_sequence = [document_tokens[i] for i in positive_domain]
        positive_sequence = tf.cast(positive_sequence, tf.int32)
        return positive_sequence

    def filter_documents_by_size(self, document_df):
        '''
        Removes documents in the dataframe that have less than <self.min_document_length> tokens.
        '''

        document_df['document_size'] = document_df.apply(lambda row: len(row['document_tokens']), axis=1)
        valid_inds = document_df["document_size"] >= self.min_document_length
        document_count = len(document_df)
        document_df = document_df[valid_inds]
        invalid_document_count = document_count - len(document_df)

        if invalid_document_count:
            print(f'WARNING: {invalid_document_count} documents were dropped due to small length.')

        return document_df

    def build_document_input_sequences(self, document_tokens):
        document_input_sequences = []

        for anchor in range(self.anchors_per_document):
            anchor_tokens, anchor_range = self.build_anchor_sequence_inds(document_tokens=document_tokens)
            positive_token_sequences = [self.build_positive_sequence(document_tokens, anchor_range) \
                                        for i in range(self.num_positive_samples)]
            anchor_sequences = [anchor_tokens, *positive_token_sequences]
            document_input_sequences += anchor_sequences

        return document_input_sequences

    def build_batch_input_sequences(self, batch_document_tokens):
        '''
        Build anchor spans and their corresponding positive spans for each document in a batch.

        Inputs
        batch_document_tokens: (N x None)-shaped integer tensor containing tokens for each document.

        Outputs
        batch_input_sequences: ((N * A * P) x None)-shaped integer tensor containing tokens for each
                                anchor and positive span.
        '''

        batch_input_sequences = []

        for document_tokens in batch_document_tokens:
            document_input_sequences = self.build_document_input_sequences(document_tokens)
            batch_input_sequences += document_input_sequences

        return batch_input_sequences

    def build_batch_label_sequence(self, batch_input_sequences):
        '''
        Inputs
        batch_input_sequences

        Outputs
        batch_labels (array-like): 2*A*N-shaped integer tensor with binary labels for each input sequence.
        '''

        batch_sequence_count = len(batch_input_sequences)
        batch_document_count = int(batch_sequence_count / self.anchors_per_document)
        batch_document_inds = list(range(batch_document_count))

        positive_document_ind = random.choice(batch_document_inds)
        positive_sequence_inds = [positive_document_ind, positive_document_ind + 1]
        labels = [1 if i in positive_sequence_inds else 0 for i in range(batch_sequence_count)]

        # Remove anchor from labels and convert to tensor.
        labels.pop(positive_document_ind)
        batch_label_sequence = tf.constant(labels)
        return batch_label_sequence, positive_document_ind

    def sample_document_inds(self, document_df):
        document_count = len(document_df)
        document_inds = list(range(document_count))
        sample_count = min(self.documents_per_batch, document_count)
        document_inds = random.sample(document_inds, k=sample_count)
        return document_inds

    def extract_anchor_from_inputs(self, batch_input_sequences, positive_document_ind):
        anchor_sequence = batch_input_sequences.pop(positive_document_ind)
        return anchor_sequence, batch_input_sequences

    def convert_anchor_and_inputs(self, anchor_sequence, batch_input_sequences):
        batch_input_sequences = tf.ragged.stack(batch_input_sequences) if not self.pad_sequences else tf.stack(batch_input_sequences)
        batch_input_sequences = tf.cast(batch_input_sequences, tf.int32)
        anchor_sequence = tf.cast(anchor_sequence, tf.int32)
        return anchor_sequence, batch_input_sequences

    def process_batch_labels(self, batch_labels):
        batch_labels = tf.cast(batch_labels, tf.int32)
        batch_labels = tf.expand_dims(batch_labels, axis=0)
        return batch_labels

    #def shuffle_inputs_outputs(self, batch_input_sequences, label_sequence):

    def tokenize_document_df(self, document_df):
        '''
        Add a tokenized documents column to the input <document_df>. Each document is tokenized with
        the processor's tokenizer.
        '''

        documents = document_df['document'].values

        if self.pretrained_tokenizer:
            documents = documents.tolist()
            documents_tokenized = list(map(self.tokenizer.tokenize, documents))
            document_df['document_tokens'] = list(map(self.tokenizer.convert_tokens_to_ids, documents_tokenized))
        else:
            document_df['document_tokens'] = self.tokenizer.texts_to_sequences(documents)

        return document_df

    def pad_anchor_and_contrasted(self, anchor, contrasted_sequences):
        anchor = pad_sequences([anchor], padding='post', maxlen=self.max_anchor_length)[0]
       # print(f'UPDATE: Anchor after padding = {anchor}, contrasted seqs = {contrasted_sequences}')
        contrasted_sequences = pad_sequences(contrasted_sequences, padding='post', maxlen=self.max_anchor_length)
        return anchor, contrasted_sequences

    def determine_padding(self, anchor, input_sequences):
        # Pad sequences for pre-trained encoders.
        if self.pretrained_tokenizer or self.pad_sequences:
            anchor, input_sequences = self.pad_anchor_and_contrasted(anchor, input_sequences)

        return anchor, input_sequences

    # TO DO: Test training time and memory usage for batch yielding vs. dataset.
    def yield_declutr_contrastive_batches(self, document_df):
        '''
        Inputs
        document_df (dataframe): Dataframe containing documents.

        Outputs
        declutr_dataset (Dataset): Tf dataset with batches of positive and negative input token sequences
                                   used for training.
        '''

        self.fit_tokenizer_in_chunks(document_df)

        # Pre-process and tokenize the document df.
        document_df = self.tokenize_document_df(document_df)
        document_df = self.filter_documents_by_size(document_df)
        self.cardinality_estimate = len(document_df)

        for i, document_chunk in enumerate(self.generate_df_chunks(document_df, self.cardinality_estimate)):
            # Build input sequences and a label sequence describing their classes as generator output.
            tokens_chunk = document_chunk['document_tokens'].values
            batch_input_sequences = self.build_batch_input_sequences(tokens_chunk)
            #print(f'UPDATE: Initial batch inputs length = {len(batch_input_sequences)}')
            batch_paths = document_chunk['script_path'].values
            batch_labels, positive_document_ind = self.build_batch_label_sequence(batch_input_sequences)
            #print(f'UPDATE: After build batch label seq, batch inputs length = {len(batch_input_sequences)}')
            anchor, batch_input_sequences = self.extract_anchor_from_inputs(batch_input_sequences, positive_document_ind)
            #print(f'UPDATE: After anchor extraction, anchor length = {len(anchor)}, batch inputs length = {len(batch_input_sequences)}')
            anchor, batch_input_sequences = self.determine_padding(anchor, batch_input_sequences)
            #print(f'UPDATE: After padding determination, anchor length = {len(anchor)}, batch inputs length = {len(batch_input_sequences)}')
            anchor, batch_input_sequences = self.convert_anchor_and_inputs(anchor, batch_input_sequences)
            #print(f'UPDATE: After conversion, anchor shape = {anchor.shape}, batch inputs shape = {batch_input_sequences.shape}')
            batch_labels = self.process_batch_labels(batch_labels)

            #print(f'UPDATE: Final anchor shape = {anchor.shape}, batch inputs shape = {batch_input_sequences.shape},'
            #      f' outputs shape = {batch_labels.shape}')
            yield {'anchor_sequence':anchor, 'contrasted_sequences': batch_input_sequences}, batch_labels

    #def yield_declutr_mlm_batches(self, ):

    def get_declutr_training_dataset(self, document_df):
        '''
        Process and return a DeClutr dataset from the input document dataframe.
        '''

        # Get tensor spec from sampled batch.
        input_sample = next(self.yield_declutr_training_batches(document_df))[0]
        contrasted_sample = input_sample['contrasted_sequences']
        input_spec = dict(anchor_sequence=tf.TensorSpec(shape=(None,), dtype=tf.int32),
                          contrasted_sequences=tf.type_spec_from_value(contrasted_sample))
        output_signature = (input_spec, tf.TensorSpec(shape=(1, self.batch_size), dtype=tf.int32))
        print(f'UPDATE: Output signature = {output_signature}')
        gen = partial(self.yield_declutr_training_batches, document_df=document_df)
        dataset = Dataset.from_generator(gen, output_signature=output_signature)
        return dataset

    def get_vocab_size(self):
        '''
        Return size of tokenizer's vocabulary = number of unique words\tokens it's seen.
        '''

        if not self.pretrained_tokenizer:
            vocab_size =  len(self.tokenizer.index_word)
        else:
            vocab_size = len(self.tokenizer.get_vocab())

        return vocab_size

    def build_batch_inputs(self, document_df):
        '''
        Inputs
        document_df (DataFrame): Document-based dataframe.

        Outputs
        batch_inputs (List)    : List of token sequences representing the anchor spans and positive samples
                                 of a random document subset. The subset is taken from <document_df>.
                                 Intended for use as a DeClutr minibatch.
        '''

        # Make sure that each document index is used as a positive sample.
        document_inds = self.sample_document_inds(document_df)
        batch_df = document_df[document_df.index.isin(document_inds)]
        batch_tokens = batch_df['document_tokens'].values
        batch_anchors_inds = list(map(self.build_anchor_sequence_inds, batch_tokens))
        batch_anchor_tokens = [x[0] for x in batch_anchors_inds]
        batch_anchor_inds = [x[1] for x in batch_anchors_inds]
        positive_tokens = list(map(self.build_positive_sequence, batch_tokens, batch_anchor_inds))
        batch_inputs = [batch_anchor_tokens, positive_tokens]
        return batch_inputs

    def convert_ids_to_tokens(self, id_sequence):
        token_sequence = self.tokenizer.sequences_to_texts(id_sequence)
        return token_sequence


























