import sys
import os

import json
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow_probability.python.distributions import Beta, Sample
from tensorflow.keras.utils import Progbar
from tensorflow import type_spec_from_value

import random

from functools import partial
from itertools import chain

from common_funcs import find_code_df_methods, make_path_directories

import math

class SequenceProcessor():
    '''
    Class for building text sequences that can train and validate a DeClutr neural network.
    For more information on DeClutr or how it's implemented in this project, please view original paper
    https://arxiv.org/abs/2006.03659 or read the documentation in sequence_models.py.

    The main role of this class is to randomly sample documents from a collection of documents and
    sample subsets of these documents to build sequence batches. The sequences are tokenized and prepared
    for Tensorflow models. These text sequence batches include binary labels describing the positive
    or negative relationship between them and the anchor sequence. Declutr's learning task is to find the most similar
    sequence = positive in the batch to a reference anchor sequence.

    Default beta parameters taken from the original DeClutr paper concentration1 = alpha, 0 = beta.
    '''

    anchor_args = dict(concentration1=4, concentration0=2)
    positive_sampling_args = dict(concentration1=2, concentration0=4)

    # Where to save tokenizers and search for pre-trained ones.
    TOKENIZER_DIR = "tokenizers"

    ANIMATION_INPUT_KEYS = ["document_texts", "anchor_index", "anchor_start", "anchor_end"]

    def __init__(self, loss_objective="declutr_contrastive", tokenizer_args={}, min_anchor_length=32, max_anchor_length=64, anchor_args={},
                 positive_sampling_args={}, anchors_per_document=1, num_positive_samples=1, documents_per_batch=32,
                 chunk_size=int(1.0e3), max_document_length=512, use_pretrained_tokenizer=False, tokenizer_type=None,
                 pretrained_tokenizer=None, sample_documents_with_replacement=False, pad_sequences=False, mmm_sample=.1,
                 animate_batch_prep=False):
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
        self.use_pretrained_tokenizer = use_pretrained_tokenizer
        self.tokenizer_args = tokenizer_args
        self.tokenizer = None
        self.pretrained_tokenizer = pretrained_tokenizer
        self.tokenizer_type = tokenizer_type
        self.sample_documents_with_replacement = sample_documents_with_replacement
        self.pad_sequences = pad_sequences

        # Sampling for masked method model objective: frac of anchor methods masked out and predicted.
        self.mmm_sample = mmm_sample

        # Define the processor's loss function objective. Then find method for generating a dataset for
        # this objective.
        self.LOSS_OBJECTIVE_TO_DATASET_METHOD = dict(declutr_contrastive=self.get_declutr_contrastive_dataset,
                                                     declutr_masked_method=self.get_declutr_mmm_dataset)
        if loss_objective not in self.LOSS_OBJECTIVE_TO_DATASET_METHOD:
            print(f'ERROR: Loss objective {loss_objective} not in {self.LOSS_OBJECTIVE_TO_DATASET_METHOD}. ')
            sys.exit(1)

        self.loss_objective = loss_objective
        self.dataset_method = self.LOSS_OBJECTIVE_TO_DATASET_METHOD[self.loss_objective]
        self.method_vocabulary = None
        self.cardinality_estimate = 0

        # Lambdas for document -> DeClutr sequence index conversion.
        self.map_sequence_to_doc_ind = lambda i: math.floor(i / (2 * self.anchors_per_document * self.num_positive_samples))
        self.map_doc_to_contrasted_ind = lambda i: self.map_sequence_to_doc_ind(i)

        self.animate_batch_prep = animate_batch_prep

    def initialize_tokenizer(self):
        # If pre-trained tokenizer is specified, the type and tokenizer itself must be provided.
        if self.use_pretrained_tokenizer:
            if not self.tokenizer_type:
                print(f'ERROR: Pretrained tokenizer requested but type unspecified! ')
                sys.exit(1)
            elif not self.pretrained_tokenizer:
                print(f'ERROR: Pretrained tokenizer requested but no tokenizer provided! ')
                sys.exit(1)

            tokenizer = self.pretrained_tokenizer
        # Otherwise, build a new Keras tokenizer.
        else:
            tokenizer = Tokenizer(**self.tokenizer_args)

        return tokenizer

    def generate_df_in_batches(self, df, documents_per_chunk=None):
        '''
        Process the document df in smaller individual chunks for improved performance. This uses
        attribute <documents_per_batch> to determine the size of each chunk.
        '''

        # Shuffle the dataframe to reduce sampling bias in the chunks.
        df = df.sample(frac=1)

        # Default documents per chunk is set to be documents per batch.
        documents_per_chunk = documents_per_chunk if documents_per_chunk else self.documents_per_batch
        document_count = len(df)
        document_inds = range(document_count)

        # Yields at least one batch for each document.
        for document_ind in document_inds:
            negative_sample_count = min(document_count, documents_per_chunk - 1)

            if negative_sample_count < documents_per_chunk - 1:
                print(f'WARNING: Document df has {document_count} events -> less than requested negative '
                      f'sample count = {negative_sample_count}.')

            chunk_index_range = [document_ind] + random.sample(document_inds, k=negative_sample_count)
            df_batch = df.take(chunk_index_range)
            yield df_batch

    def partition_df(self, df, chunk_size):
        '''
        Yield the df in a fixed amount of chunks by partitioning its indices in windows of <chunk_size>.
        '''

        document_count = len(df)
        chunk_count = math.ceil(document_count/chunk_size)

        for i, chunk in enumerate(range(chunk_count)):
            chunk_start = int(i * chunk_size)
            chunk_end = int(min((i + 1) * chunk_size, document_count))
            chunk_inds = list(range(chunk_start, chunk_end))
            df_chunk = df.take(chunk_inds)
            yield df_chunk

    def get_tokenizer_path(self, text_column):
        tokenizer_path = os.path.join(self.TOKENIZER_DIR, f"{text_column}_tokenizer.json")
        return tokenizer_path

    def search_for_tokenizer(self, text_column):
        tokenizer_path = self.get_tokenizer_path(text_column)
        tokenizer = None

        if os.path.exists(tokenizer_path):
            print(f"UPDATE: Found tokenizer path {tokenizer_path}.")
            with open(tokenizer_path, "r") as file:
                try:
                    tokenizer_str = file.read()
                    tokenizer = tokenizer_from_json(tokenizer_str)
                except:
                    print(f"WARNING: Failed to load JSON Keras tokenizer found in {tokenizer_path}.")

        if not tokenizer:
            tokenizer = self.initialize_tokenizer()

        return tokenizer

    def search_for_tokenizer_pickle(self, text_column):
        tokenizer_path = self.get_tokenizer_path(text_column).replace(".json", ".pickle")
        tokenizer = None

        if os.path.exists(tokenizer_path):
            print(f"UPDATE: Found tokenizer path {tokenizer_path}.")
            with open(tokenizer_path, "rb") as file:
                try:
                    tokenizer_str = file.read()
                    tokenizer = pickle.loads(tokenizer_str)
                except:
                    print(f"WARNING: Failed to load JSON Keras tokenizer found in {tokenizer_path}.")

        if not tokenizer:
            tokenizer = self.initialize_tokenizer()

        return tokenizer

    def fit_tokenizer_on_documents(self, document_df, text_column):
        documents = document_df[text_column].values

        if self.use_pretrained_tokenizer:
            pass
        else:
            self.tokenizer.fit_on_texts(documents)

        return document_df

    def fit_tokenizer_in_chunks(self, document_df, text_column, chunk_size=1.0e3):
        if not self.tokenizer:
            self.tokenizer = self.search_for_tokenizer_pickle(text_column)

        document_count = len(document_df)
        chunk_count = math.ceil(document_count / chunk_size)
        print(f'UPDATE: Tokenization on {text_column} chunks in progress.')
        progress_bar = Progbar(target=chunk_count, stateful_metrics=["document_count", "vocabulary_size"])
        document_count = 0

        for i, document_df_chunk in enumerate(self.partition_df(document_df, chunk_size)):
            self.fit_tokenizer_on_documents(document_df_chunk, text_column=text_column)
            vocab_size = self.get_vocab_size()
            document_count += len(document_df_chunk)
            progress_values = [["document_count", document_count], ["vocabulary_size", vocab_size]]
            progress_bar.update(i + 1, values=progress_values)

    def build_anchor_sequence_inds(self, document_tokens):
        '''
        Choose an anchor span for representing a document's token sequence.
        '''

        token_count = len(document_tokens)
        anchor_length_prob = self.anchor_length_distribution.sample([1])
        anchor_length = int(anchor_length_prob * (self.max_anchor_length - self.min_anchor_length)) + self.min_anchor_length
        start_domain = list(range(token_count - anchor_length))
        anchor_start = random.choice(start_domain)
        anchor_range = range(anchor_start, anchor_start + anchor_length)
        anchor_sequence = [document_tokens[i] for i in anchor_range]
        anchor_sequence = tf.cast(anchor_sequence, tf.int32)
        return anchor_sequence, anchor_range

    def build_positive_sequence(self, document_tokens, anchor_range):
        '''
        Build a positive sequence for a document with respect to an anchor span. Length of which is sampled with a beta
        distribution. Its starting point is uniformly sampled.

        NOTE: Difference from sampling method described in DeClutr paper: Positive length is restricted to a subsumed view
        if anchor end > (document length - max span length). This prevents the positive span from exceeding the document's
        bounds.
        '''

        positive_length_prob = self.positive_sample_distribution.sample([1])

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

        if self.animate_batch_prep:
            self.anchor_start = anchor_start
            self.anchor_end = anchor_end

        return positive_sequence

    def add_document_size_column(self, document_df):
        document_df["document_size"] = document_df.apply(lambda row: len(row['document_tokens']), axis=1)
        return document_df

    def filter_documents_by_size(self, document_df):
        '''
        Removes documents in the dataframe that have less than <self.min_document_length> tokens.

        Inputs
        document_df (DataFrame): Tokenized document dataframe with a "document_tokens" column.
        '''

        document_df = self.add_document_size_column(document_df) if "document_size" not in document_df.columns else document_df
        valid_inds = document_df["document_size"] >= self.min_document_length
        document_df = document_df[valid_inds]
        return document_df

    def build_document_input_sequences(self, document_tokens):
        '''
        Inputs
        document_tokens (array-like): Tokens representing a document's text.

        Outputs
        List of the document's anchor sequences, and positive sequences for each anchor. Positive sequences are used as
        negative samples for different documents (soft negative samples), and hard negative samples for different anchors
        of the same document.
        '''

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

    def build_contrastive_label_sequence(self, batch_input_sequences):
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
        '''
        From an input document dataframe, return randomly sampled document indices.
        '''

        document_count = len(document_df)
        document_inds = list(range(document_count))
        sample_count = min(self.documents_per_batch, document_count)
        document_inds = random.sample(document_inds, k=sample_count)
        return document_inds

    def extract_anchor_from_inputs(self, batch_input_sequences, positive_document_ind):
        anchor_sequence = batch_input_sequences.pop(positive_document_ind)
        return anchor_sequence, batch_input_sequences

    def convert_anchor_and_inputs(self, anchor_sequence, batch_input_sequences):
        '''
        Convert anchor and contrasted sequences into tensors of types that vary by model architecture.
        '''

        batch_input_sequences = tf.ragged.stack(batch_input_sequences) if not self.pad_sequences else tf.stack(batch_input_sequences)
        batch_input_sequences = tf.cast(batch_input_sequences, tf.int32)
        anchor_sequence = tf.cast(anchor_sequence, tf.int32)
        return anchor_sequence, batch_input_sequences

    def process_batch_labels(self, batch_labels):
        batch_labels = tf.cast(batch_labels, tf.int32)
        batch_labels = tf.expand_dims(batch_labels, axis=0)
        return batch_labels

    def tokenize_document_df(self, document_df, text_column):
        '''
        Add a tokenized documents column to the input <document_df>. Each document is tokenized with the processor's tokenizer.
        '''

        documents = document_df[text_column].values

        if self.use_pretrained_tokenizer:
            documents = documents.tolist()
            documents_tokenized = list(map(self.tokenizer.tokenize, documents))
            document_df['document_tokens'] = list(map(self.tokenizer.convert_tokens_to_ids, documents_tokenized))
        else:
            document_df['document_tokens'] = self.tokenizer.texts_to_sequences(documents)

        return document_df

    def pad_anchor_and_contrasted(self, anchor, contrasted_sequences):
        '''
        Pad anchor and contrasted sequences to the right with zeros using <max_anchor_length> length.
        '''

        anchor = pad_sequences([anchor], padding='post', maxlen=self.max_anchor_length)[0]
        contrasted_sequences = pad_sequences(contrasted_sequences, padding='post', maxlen=self.max_anchor_length)
        return anchor, contrasted_sequences

    def determine_padding(self, anchor, input_sequences):
        '''
        Pad sequences for pre-trained encoders or if specified with processor instance.
        '''

        if self.use_pretrained_tokenizer or self.pad_sequences:
            anchor, input_sequences = self.pad_anchor_and_contrasted(anchor, input_sequences)

        return anchor, input_sequences

    def update_sequences_with_column(self, input_sequences, df, column, anchor_ind):
        '''
        Use a column from df to make other entries in the batch of input sequences. For example:

        column = "programming_language" produces two columns: "anchor_programming_language" and
        "contrasted_programming_languages," so that other information about the sequences is available
        in input_sequences.
        '''

        if column not in df.columns:
            print(f'ERROR! Tried to add column {column} not available in df columns = {df.columns}.')
            sys.exit(1)

        #TODO: Modularize this method.
        column_vals = df[column].values
        anchor_document_ind = self.map_sequence_to_doc_ind(anchor_ind)
        anchor_val = column_vals[anchor_document_ind]
        anchor_tensor_name = f"anchor_{column}"
        input_sequences[anchor_tensor_name] = anchor_val
        contrasted_tensor_name = f"contrasted_{column}s"
        row_count = input_sequences["contrasted_sequences"].shape[0] + 1
        contrasted_vals = [column_vals[self.map_doc_to_contrasted_ind(i)] for i in range(row_count)]
        contrasted_vals.pop(anchor_document_ind)
        input_sequences[contrasted_tensor_name] = contrasted_vals
        return input_sequences

    def add_columns(self, input_sequences, df, columns, anchor_ind):
        for column in columns:
            input_sequences = self.update_sequences_with_column(input_sequences, df, column, anchor_ind)

        return input_sequences

    # TO DO: Test training time and memory usage for batch yielding vs. dataset.
    def yield_declutr_contrastive_batches(self, document_df, add_columns=[]):
        '''
        Inputs
        document_df (dataframe): Dataframe containing documents. Must be tokenized with "document_tokens" column before.
        add_columns (list):      Columns from document_df to include with each batch of sequences.

        Outputs
        declutr_dataset (dict): Dictionary with batches of positive and negative input token sequences used for training.
        '''

        # Pre-process and tokenize the document df.
        self.cardinality_estimate = len(document_df)

        for i, document_chunk in enumerate(self.generate_df_in_batches(document_df)):
            # Build input sequences and a label sequence describing their classes as generator output.
            tokens_chunk = document_chunk['document_tokens'].values
            contrastive_inputs = self.build_batch_input_sequences(tokens_chunk)
            contrastive_labels, anchor_index = self.build_contrastive_label_sequence(contrastive_inputs)
            anchor, contrastive_inputs = self.extract_anchor_from_inputs(contrastive_inputs, anchor_index)
            anchor, contrastive_inputs = self.determine_padding(anchor, contrastive_inputs)
            anchor, contrastive_inputs = self.convert_anchor_and_inputs(anchor, contrastive_inputs)
            batch_labels = self.process_batch_labels(contrastive_labels)
            input_sequences = dict(anchor_sequence=anchor, contrasted_sequences=contrastive_inputs)
            input_sequences = self.add_columns(input_sequences, document_chunk, add_columns, anchor_index)

            # Provide intermediate variables\steps of the process for making visuals.
            if self.animate_batch_prep:
                document_texts = document_chunk["docstring"].values
                input_sequences["document_texts"] = document_texts
                input_sequences["anchor_index"] = anchor_index
                input_sequences["anchor_start"] = self.anchor_start
                input_sequences["anchor_end"] = self.anchor_end

            yield input_sequences, batch_labels

    def yield_declutr_mmm_batches(self, document_df):
        '''
        Add masked anchor to inputs and labels of masked tokens to labels. This allows for masked method modeling.
        '''

        if not self.method_vocabulary:
            print(f'WARNING: Declutr MMM batches called without method vocabulary. Building method vocab now.')
            self.build_method_vocabulary(document_df)

        for inputs, labels in self.yield_declutr_contrastive_batches(document_df):
            anchor = inputs["anchor_sequence"]
            mmm_inputs, mmm_labels = self.build_mmm_inputs(anchor)
            no_methods_found = mmm_labels.shape[0] == 0

            # This is fine, as some spans won't contain any methods. Especially true for docstrings because they're
            # written in natural language.
            if no_methods_found:
                continue

            if not mmm_inputs:
                print(f'WARNING: Empty masked anchor sequence. Skipping this MMM batch.')
                continue

            yield mmm_inputs, mmm_labels

    def count_declutr_mmm_batches(self, document_df):
        '''
        Add masked anchor to inputs and labels of masked tokens to labels. This allows for masked method
        modeling.
        '''

        batch_count = 0

        for inputs, labels in self.yield_declutr_mmm_batches(document_df):
            batch_count += 1

        return batch_count

    def build_method_vocabulary(self, code_df):
        if 'methods' not in code_df.columns:
            print(f'WARNING: No methods column found in code dataframe. Attempting to build one. ')
            code_df = find_code_df_methods(code_df)

        elif not self.tokenizer.word_index:
            print(f"WARNING: Processor tokenizer hasn't fit on code df during method vocab building.")
            self.fit_tokenizer_in_chunks(code_df, text_column="code")

        methods = list(chain.from_iterable(code_df['methods'].values.tolist()))
        self.method_vocabulary = [method for method in methods if method in self.tokenizer.word_index]
        self.MASKED_INDEX = self.get_vocab_size()
        self.method_vocabulary.append(self.MASKED_INDEX)
        self.tokenizer.fit_on_texts(["MASKED_TOKEN"])
        print(f'UPDATE: Sequence processor method vocab size: {len(self.method_vocabulary)}, masked index = {self.MASKED_INDEX}.')

    def build_mmm_inputs(self, anchor_sequence):
        '''
        Inouts
        anchor_sequence: A T-length integer tensor containing T tokens describing the anchor span of a document.

        Outputs
        masked_anchor_sequence: The same anchor_sequence tensor, but with a random portion of the tokens associated with
                                methods masked out.
        masked_method_tokens: A M-sized integer tensor with the correct token for each masked out method token.
        '''

        self.method_tokens = [self.tokenizer.word_index[method] for method in self.method_vocabulary if method != self.MASKED_INDEX]
        anchor_method_inds = [i for i, token in enumerate(anchor_sequence) if token in self.method_tokens]
        masked_out_method_count = int(self.mmm_sample * len(anchor_method_inds))
        masked_out_inds = random.sample(anchor_method_inds, k=masked_out_method_count)

        # Replace masked out tokens with -1.
        masked_out_anchor_sequence = [token if i not in masked_out_inds else self.MASKED_INDEX for i, token in enumerate(anchor_sequence)]
        masked_out_tokens = [anchor_sequence[i] for i in masked_out_inds]
        masked_out_one_hot = tf.one_hot(indices=masked_out_tokens, depth=self.get_method_vocab_size())
        return masked_out_anchor_sequence, masked_out_one_hot

    def get_declutr_contrastive_dataset(self, document_df):
        '''
        Process and return a DeClutr dataset from the input document dataframe.
        '''

        # Get tensor spec from sampled batch.
        input_sample = next(self.yield_declutr_contrastive_batches(document_df))[0]
        contrasted_sample = input_sample['contrasted_sequences']
        input_spec = dict(anchor_sequence=tf.TensorSpec(shape=(None,), dtype=tf.int32),
                          contrasted_sequences=type_spec_from_value(contrasted_sample))

        # Add batch animation data to the specs.
        if self.animate_batch_prep:
            for animation_key in self.ANIMATION_INPUT_KEYS:
                print(f"UPDATE: Adding animation key {animation_key} spec. ")
                animation_sample = input_sample[animation_key]
                input_spec[animation_key] = type_spec_from_value(animation_sample)

        output_signature = (input_spec, tf.TensorSpec(shape=(1, self.batch_size), dtype=tf.int32))
        gen = partial(self.yield_declutr_contrastive_batches, document_df=document_df)
        dataset = Dataset.from_generator(gen, output_signature=output_signature)
        return dataset

    def get_declutr_mmm_dataset(self, document_df):
        '''
        Process and return a DeClutr dataset from the input document dataframe.
        '''

        # Get tensor spec from sampled batch.
        input_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32)
        method_count = self.get_method_vocab_size()
        output_signature = (input_spec, tf.TensorSpec(shape=(None, method_count), dtype=tf.int32))
        print(f'UPDATE: Output signature = {output_signature}')
        gen = partial(self.yield_declutr_mmm_batches, document_df=document_df)
        dataset = Dataset.from_generator(gen, output_signature=output_signature)
        return dataset
    
    def get_dataset(self, document_df):
        return self.dataset_method(document_df)

    def get_vocab_size(self):
        '''
        Return tokenizer's vocabulary size.
        '''
        
        vocab = self.tokenizer.index_word if not self.use_pretrained_tokenizer else self.tokenizer.get_vocab()
        vocab_size = len(vocab) 
        return vocab_size

    def get_method_vocab_size(self):
        return len(self.method_vocabulary)

    def convert_ids_to_tokens(self, id_sequence):
        token_sequence = self.tokenizer.sequences_to_texts(id_sequence)
        return token_sequence

    def texts_to_sequences(self, documents):
        return self.tokenizer.texts_to_sequences(documents)

    def cache_tokenizer(self, text_column):
        tokenizer_path = self.get_tokenizer_path(text_column)
        make_path_directories(tokenizer_path)
        json_tokenizer = self.tokenizer.to_json()

        with open(tokenizer_path, "w") as file:
            print(f"UPDATE: Caching tokenizer to {tokenizer_path}.")
            file.write(json_tokenizer)

        pickle_path = tokenizer_path.replace(".json", ".pickle")
        pickle_obj = pickle.dumps(self.tokenizer)

        with open(pickle_path, "wb") as file:
            print(f"UPDATE: Caching pickled tokenizer to {pickle_path}.")
            file.write(pickle_obj)



