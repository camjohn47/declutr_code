import os
import sys

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorboard.plugins.hparams import api as hp
from tensorflow.data.experimental import cardinality
from tensorflow.keras.models import save_model
from tensorflow.keras.losses import CategoricalCrossentropy as cross_entropy

from common_funcs import run_with_time, drop_nan_text
from code_parser import CodeParser
from sequence_models import DeClutrContrastive, DeclutrMaskedLanguage
from sequence_processor import SequenceProcessor
from loss_functions import ContrastiveLoss, MaskedMethodLoss
from tensor_visualizer import TensorVisualizer, VisualizerCallBack
from visuals import make_histogram, process_fig, make_histogram_comparison

from itertools import product

from pathlib import Path

import time

import math
import numpy as np

import dill

tf.executing_eagerly()

# Make sure Tensorflow can find your GPU and is using it properly.
tf.config.list_physical_devices()

class DeClutrTrainer(SequenceProcessor):
    # Number top categories included for top k categorical accuracy. For ex, if model predicts true
    # class as 2nd or 3rd highest, these would both be considered correct with TOP_K=3.
    TOP_K = 3
    #TODO: Build top k label in class and express as fraction of batch size.
    metrics = [CategoricalAccuracy(), TopKCategoricalAccuracy(k=TOP_K, name=f"top_{TOP_K}_categorical_accuracy")]

    # Hyper-parameters and domains optimized with tensorboard hparams.
    EMBEDDING_DIM_DOMAIN = [100]
    HP_EMBEDDING_DIM = hp.HParam('embedding_dimension', hp.Discrete(EMBEDDING_DIM_DOMAIN))
    HPARAMS = [HP_EMBEDDING_DIM]
    DECLUTR_MODEL_CLASSES = dict(declutr_contrastive=DeClutrContrastive, declutr_masked_language=DeclutrMaskedLanguage)
    DECLUR_MODEL_LOSSES = dict(declutr_contrastive=ContrastiveLoss, declutr_masked_language=MaskedMethodLoss)
    code_parser_args = dict(programming_language="all")
    SUBPLOT_XAXIS = [dict(title_text="Token Count") for i in range(2)]
    SUBPLOT_YAXIS = [dict(title_text='Scripts') for i in range(2)]
    LAYOUT_ARGS = dict(title_text="Document Size Distribution Before and After Filter", title_x=0.5)
    XAXIS_RANGE = [0, 5000]
    models_dir = "models"
    tensorboard_dir = "tensorboard_logs"

    def __init__(self, sequence_processor_args={}, code_parser_args={}, chunk_size=1000, epoch_count=3, train_split=.75,
                 metrics=[], declutr_model="declutr_contrastive", encoder_model='lstm', save_format="tf", models_dir=None,
                 tensorboard_dir="tensorboard_logs", save_training_data=True, visualize_tensors=False, sampling=1, text_column="code"):

        super().__init__(**sequence_processor_args)
        declutr_model = self.loss_objective

        if declutr_model not in self.DECLUTR_MODEL_CLASSES:
            print(f'ERROR: Requested declutr model {declutr_model} not in available classes: '
                  f'{self.DECLUTR_MODEL_CLASSES}.')
            sys.exit(1)

        self.declutr_model = declutr_model
        self.declutr_model_class = self.DECLUTR_MODEL_CLASSES[self.declutr_model]
        sequence_processor_args["loss_objective"] = self.declutr_model
        self.loss = cross_entropy(name=self.declutr_model)
        self.chunk_size = chunk_size
        self.epoch_count = epoch_count
        self.train_split = train_split

        # Overwrite default training metrics if they're provided.
        self.metrics = metrics if metrics else self.metrics
        self.model_path = None
        self.encoder_model = encoder_model
        is_transformer_architecture = self.encoder_model in ["transformer", "transformer_encoder"]

        if is_transformer_architecture and not self.pad_sequences:
            print(f'WARNING: Pad sequences not set and transformer architecture is requested! Setting pad sequences = True.')
            self.pad_sequences = True

        self.save_format = save_format
        self.tensorboard_dir = tensorboard_dir if tensorboard_dir else self.tensorboard_dir
        print(f"UPDATE: Setting Trainer's TensorBoard directory to {self.tensorboard_dir}.")
        self.save_training_data = save_training_data

        # Whether to use TensorVisualizer for visualizing training outputs and labels.
        self.visualize_tensors = visualize_tensors

        # Parser for finding scripts and building text dataframes from which sequences are made.
        self.models_dir = models_dir if models_dir else self.models_dir
        self.code_parser_args.update(code_parser_args)
        self.code_parser = CodeParser(**self.code_parser_args)
        self.sampling = sampling

        # Column of text used for training. Either "code," or "docstring."
        self.text_column = text_column

    def build_code_df(self, code_directory):
        func = self.code_parser.code_directory_to_df
        kwargs = dict(script_directory=code_directory)
        code_df = run_with_time(func=func, kwargs=kwargs, name="CodeParser df building")
        code_df = code_df.sample(frac=self.sampling)
        print(f'UPDATE: DeClutrTrainer has built a code df with sampling={self.sampling}. Info shown below.')
        code_df.info()
        return code_df

    def make_programming_language_hist(self, code_df):
        layout_args = dict(title_text="Programming Language Distribution, Train Subset", title_x=0.5)
        hist = make_histogram(code_df, column="language", layout_args=layout_args)
        path = os.path.join(self.model_dir, self.model_id, "programming_language_histogram.html")
        process_fig(fig=hist, path=path)

    def start_training_from_directory(self, code_directory, declutr_args={}):
        '''
        Parse through a code directory to extract a text dataframe. Then, build sequences from this text and train a
        Declutr model with them.
        '''

        if "model_id" not in declutr_args:
            print(f'ERROR: Model is must be specified in declutr_args! ')
            sys.exit(1)

        self.model_id = declutr_args["model_id"]
        code_df = self.build_code_df(code_directory)
        code_df = drop_nan_text(code_df, text_column=self.text_column)
        self.fit_tokenizer_in_chunks(code_df, text_column=self.text_column)

        #TODO: Re-implement full sampling requirement after testing.
        if self.sampling == 1:
            print(f"UPDATE: Saving tokenizer because text is fully sampled.")
            self.cache_tokenizer(self.text_column)

        declutr_model = self.build_declutr_model(code_df=code_df, declutr_args=declutr_args)
        self.make_programming_language_hist(code_df)
        self.train_model(declutr_model=declutr_model, document_df=code_df)

    def save_to_model_dir(self):
        '''
        Save this trainer instance with Dill to the tensorflow model directory. That way, when loading the model later,
        you can easily produce sequences with the same processor used for training.
        '''

        if not self.model_dir:
            print(f'ERROR: Tried to save DeClutr Trainer before building model directory! ')
            sys.exit(1)

        serialized_self = dill.dumps(self)
        self.path = os.path.join(self.model_dir, "processor.dill")

        with open(self.path, "wb") as file:
            print(f'UPDATE: Saving DeClutrTrainer to {self.path}.')
            file.write(serialized_self)

    def build_hparam_callback(self, log_dir, hparam_vals):
        '''
        hparam_vals (dict): Mapping from hparam to the specific value it's taking for this experiment.
        '''

        hparam_val_dic = {hparam: hparam_vals[i] for i, hparam in enumerate(self.HPARAMS)}
        hparam_callback = hp.KerasCallback(log_dir, hparam_val_dic)
        return hparam_callback

    def update_declutr_encoder_args(self, declutr_args, vocab_size):
        # Increment input dimension to account for the masked token if using a MLM architecture.
        declutr_args['input_dims'] = vocab_size if self.declutr_model_class == 'declutr_contrastive' else vocab_size + 1
        declutr_args['batch_size'] = self.batch_size
        declutr_args["visualize_tensors"] = self.visualize_tensors

        # If positional encodings are used for Transformer, provide sequence length.
        if declutr_args["encoder_config"]["use_positional_encodings"]:
            declutr_args["encoder_config"]["sequence_length"] = self.max_anchor_length

        return declutr_args

    def set_directories(self, declutr_model):
        self.model_dir = declutr_model.model_dir
        print(f"UPDATE: Setting Trainer model's directory to {self.model_dir}.")
        self.model_path = self.model_dir

        if self.save_format == 'h5':
            self.model_path = os.path.join(self.model_path, "declutr_model.h5")

        self.document_length_hist_path = os.path.join(self.model_dir, "document_length_histogram.html")

    def save_encoder(self, model):
        self.encoder_path = os.path.join(self.model_dir, "encoder")
        self.encoder_path = os.path.join(self.encoder_path, ".h5") if self.save_format == 'h5' else self.encoder_path
        encoder = model.encoder
        print(f"UPDATE: Saving encoder to {self.encoder_path}.")
        save_model(model=encoder, filepath=self.encoder_path, save_format=self.save_format)

    def build_tensor_visualizer(self, model_dir):
        tensor_visualizer = TensorVisualizer(tf_model_dir=model_dir) if self.visualize_tensors else None
        return tensor_visualizer

    def build_declutr_model(self, code_df, declutr_args={}):
        '''
        Build and return a Declutr model for training using code_df for vocabulary.

        Inputs
        code_df (DataFrame): A script-based dataframe supplying text and other attributes for training sequences.
        declutr_args (dict): Parameter-value mapping with configuration for building a Declutr model.
        '''

        vocab_size = self.get_vocab_size()
        declutr_args = self.update_declutr_encoder_args(declutr_args, vocab_size)

        if self.declutr_model == 'declutr_contrastive':
            print(f'UPDATE: Building contrastive DeClutr model with declutr args = {declutr_args}.')
            declutr_model = self.declutr_model_class(**declutr_args)
        else:
            print(f'UPDATE: DeClutr trainer building method vocabulary from code df.')
            self.build_method_vocabulary(code_df)
            masked_vocabulary_size = self.get_method_vocab_size()
            print(f'UPDATE: Building MML model with masked vocab size = {masked_vocabulary_size}, '
                  f'masked index = {self.MASKED_INDEX} declutr args = {declutr_args}.')
            declutr_model = self.declutr_model_class(masked_vocabulary_size=masked_vocabulary_size, masked_token=self.MASKED_INDEX,
                                                     **declutr_args)

        declutr_model.compile(optimizer=Adam(), loss=self.loss, metrics=self.metrics, run_eagerly=True)
        self.set_directories(declutr_model)

        # Save trainer to the model directory for easy coordination during later usage\loading.
        self.save_to_model_dir()
        self.tensor_visualizer = declutr_model.tensor_visualizer
        return declutr_model

    def update_callbacks_visuals(self, callbacks):
        if self.visualize_tensors:
            visualize_callback = VisualizerCallBack(self.tensor_visualizer)
            callbacks.append(visualize_callback)

        return callbacks

    def get_model_callbacks(self, hparam_tuning=True, hparam_vals=None):
        '''
        Get keras callbacks for helping with model training and providing analysis data for it.

        Inputs
        hparam_tuning (bool): Whether to add HParam callback for hyperparameter tuning.
        hparam_vals (list): List of values each hyperparam is taking.
        '''

        # Set CSV log directory to be chunk, epoch dependent to avoid overwriting when training over
        # different chunks and epochs.
        log_path = os.path.join(self.log_dir, 'fit_log.csv')
        csv_logger = CSVLogger(filename=log_path, append=True)
        tensorboard = TensorBoard(log_dir=self.tensorboard_dir, write_images=True, update_freq='batch', embeddings_freq=1,
                                  histogram_freq=1000)
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_dir, 'checkpoint'))
        callbacks = [csv_logger, checkpoint]

        if hparam_tuning:
            if not hparam_vals:
                print(f'ERROR: No hparam vals provided for hparam tuning! ')
                sys.exit(1)

            hparam_callback = self.build_hparam_callback(self.model_dir, hparam_vals)
            callbacks.append(hparam_callback)

        callbacks = self.update_callbacks_visuals(callbacks)
        return callbacks

    def train_over_grid_search(self, declutr, dataset_train, dataset_val, train_steps, val_steps):
        '''
        Train and validate over all values in an HParam domain with the current chunk's datasets.
        '''

        hparam_domains = [hparam.domain.values for hparam in self.HPARAMS]
        hparam_combos = list(product(hparam_domains)) if len(hparam_domains) > 1 else [[i] for i in hparam_domains[0]]

        # Only one epoch fed into fit call at a time to refresh negative samples. If we used epochs=self.epoch_count instead,
        # the same negative samples would be used which would likely result in lower quality embeddings.
        #TODO: Run above experiment.
        for hparam_combo in hparam_combos:
            callbacks = self.get_model_callbacks(hparam_vals=hparam_combo)
            print(f'UPDATE: Beginning training with HParam combo {hparam_combo}. Training steps = {train_steps},'
                  f' validation steps = {val_steps}.')
            start = time.time()
            declutr.fit(x=dataset_train, epochs=1, callbacks=callbacks, validation_data=dataset_val,
                        steps_per_epoch=train_steps, validation_steps=val_steps)
            print(f'UPDATE: Train time = {time.time() - start}.')
            print(f'UPDATE: Saving full model to {self.model_path}.')
            save_model(declutr, filepath=self.model_path, save_format=self.save_format)
            self.save_encoder(declutr)

    def build_fit_directory(self):
        self.log_dir = os.path.join(self.model_dir, f'chunk_{self.chunk}')
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def update_indices(self, chunk, epoch):
        self.chunk = chunk
        self.epoch = epoch

    def get_batch_count(self, document_df):
        if self.declutr_model == 'declutr_contrastive':
            # By default, one batch of negative samples for each document as an anchor sample.
            return self.cardinality_estimate
        else:
            return self.count_declutr_mmm_batches(document_df)

    @staticmethod
    def get_max_bin_height(hist_vals, bin_width):
        max_hist_vals = hist_vals[0]
        first_bin_end = bin_width
        num_vals_in_bin = len(max_hist_vals[max_hist_vals < first_bin_end])
        return num_vals_in_bin

    def filter_document_df(self, document_df, make_visuals=True):
        '''
        Filter documents by length and build/save Plotly histogram for it.
        '''

        document_count = len(document_df)
        document_df = self.add_document_size_column(document_df)
        document_sizes_before = document_df["document_size"].values
        document_df = self.filter_documents_by_size(document_df)
        document_sizes_after = document_df["document_size"].values
        invalid_document_count = document_count - len(document_df)

        if invalid_document_count:
            print(f'WARNING: {invalid_document_count} documents were dropped due to small length.')
            document_df.info()

        if not make_visuals:
            return document_df

        # Make a side by side comparison of the document size distribution before and after filtering.
        print(f'UPDATE: Making before and after document size histogram. ')
        hist_vals = [document_sizes_before, document_sizes_after]
        subplot_titles = [f"Before Filter ({len(hist_vals[0])} Scripts)", f"After Filter ({len(hist_vals[1])} Scripts)"]
        xbins = dict(start=0, end=np.max(document_sizes_after), size=self.min_document_length)
        yaxis_range = [0, self.get_max_bin_height(hist_vals, bin_width=self.min_document_length)]

        #TODO: Simplify this by moving constant arguments to a class dictionary.
        fig = make_histogram_comparison(hist_vals=hist_vals, rows=1, cols=2, subplot_titles=subplot_titles,
                                        subplot_xaxis=self.SUBPLOT_XAXIS, subplot_yaxis=self.SUBPLOT_YAXIS,
                                        layout_args=self.LAYOUT_ARGS, xbins=xbins, xaxis_range=self.XAXIS_RANGE,
                                        yaxis_range = yaxis_range)
        process_fig(fig, self.document_length_hist_path)
        return document_df

    def train_model(self, declutr_model, document_df):
        '''
        For each epoch, create new anchor spans + positive and negative samples from the document df.
        Then use these sequences as inputs and outputs for training the declutr model.
        '''

        document_df = self.tokenize_document_df(document_df, text_column=self.text_column)
        document_df = self.filter_document_df(document_df)
        self.chunk_count = math.ceil(len(document_df) / self.chunk_size)

        if self.save_training_data:
            training_data_path = os.path.join(self.model_dir, "training_data.csv")
            print(f"UPDATE: Saving training document df to {training_data_path}.")
            document_df.to_csv(training_data_path, index=False)

        for chunk, chunk_df in enumerate(self.partition_df(document_df, chunk_size=self.chunk_size)):
            for epoch in range(self.epoch_count):
                print(f'UPDATE: Beginning DeClutr training for chunk {chunk}/{self.chunk_count}, '
                      f'epoch {epoch}, chunk size = {len(chunk_df)}.')
                self.update_indices(chunk, epoch)
                self.build_fit_directory()
                dataset = self.get_dataset(chunk_df)
                self.batch_count = cardinality(dataset) if cardinality(dataset) != tf.data.experimental.UNKNOWN_CARDINALITY\
                                   else self.get_batch_count(chunk_df)
                train_steps = int(self.batch_count * self.train_split)
                val_steps = self.batch_count - train_steps
                dataset_train = dataset.take(train_steps)
                dataset_val = dataset.skip(train_steps)
                self.train_over_grid_search(declutr_model, dataset_train, dataset_val, train_steps, val_steps)

