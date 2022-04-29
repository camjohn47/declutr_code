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

from common_funcs import run_with_time
from code_parser import CodeParser
from sequence_models import DeClutrContrastive, DeclutrMaskedLanguage
from sequence_processor import SequenceProcessor
from loss_functions import ContrastiveLoss, MaskedMethodLoss
from tensor_visualizer import TensorVisualizer, VisualizerCallBack
from visuals import make_histogram, process_fig

from itertools import product

from pathlib import Path

import time

import math

import dill

tf.executing_eagerly()
tf.config.list_physical_devices()

class DeClutrTrainer(SequenceProcessor):
    # Number top categories included for top k categorical accuracy. For ex, if model predicts true
    # class as 2nd or 3rd highest, these would both be considered correct with TOP_K=3.
    TOP_K = 3
    metrics = [CategoricalAccuracy(), TopKCategoricalAccuracy(k=TOP_K, name=f"top_{TOP_K}_categorical_accuracy")]

    # Hyper-parameters and domains optimized with tensorboard hparams.
    EMBEDDING_DIM_DOMAIN = [10, 100]
    HP_EMBEDDING_DIM = hp.HParam('embedding_dimension', hp.Discrete(EMBEDDING_DIM_DOMAIN))
    HPARAMS = [HP_EMBEDDING_DIM]
    DECLUTR_MODEL_CLASSES = dict(declutr_contrastive=DeClutrContrastive, declutr_masked_language=DeclutrMaskedLanguage)
    DECLUR_MODEL_LOSSES = dict(declutr_contrastive=ContrastiveLoss, declutr_masked_language=MaskedMethodLoss)
    code_parser_args = dict(programming_language="all")

    def __init__(self, sequence_processor_args={}, code_parser_args={}, chunk_size=1000, epoch_count=3, train_split=.75,
                 metrics=[], declutr_model="declutr_contrastive", encoder_model='lstm', save_format="tf", model_dir="models",
                 tensorboard_dir="tensorboard_logs", save_training_data=True, visualize_tensors=False, sampling=1):

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

        if self.encoder_model == 'transformer' and not self.pad_sequences:
            print(f'WARNING: Pad sequences not set and transformer encoder is requested! Setting pad'
                  f' sequences = True. ')
            self.pad_sequences = True

        self.save_format = save_format
        self.tensorboard_dir = tensorboard_dir
        self.save_training_data = save_training_data

        # Whether to use TensorVisualizer for visualizing training outputs and labels.
        self.visualize_tensors = visualize_tensors

        # Parser for finding scripts and building text dataframes from which sequences are made.
        self.model_dir = model_dir
        self.code_parser_args.update(code_parser_args)
        self.code_parser = CodeParser(**self.code_parser_args)
        self.sampling = sampling

    def build_code_df(self, code_directory):
        func = self.code_parser.code_directory_to_df
        kwargs = dict(script_directory=code_directory)
        code_df = run_with_time(func=func, kwargs=kwargs, name="CodeParser df building")
        code_df = code_df.sample(frac=self.sampling)
        print(f'UPDATE: DeClutrTrainer has built code df with sampling={self.sampling}. Info shown below.')
        code_df.info()
        return code_df

    def make_programming_language_hist(self, code_df):
        layout_args = dict(title_text="Programming Language Distribution, Train Subset", title_x=0.5)
        hist = make_histogram(code_df, column="language", layout_args=layout_args)
        path = os.path.join(self.model_dir, self.model_id, "programming_language_histogram.png")
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
        code_df = self.preprocess_df(code_df)
        self.make_programming_language_hist(code_df)
        self.fit_tokenizer_in_chunks(code_df)
        declutr_model = self.build_declutr_model(code_df=code_df, declutr_args=declutr_args)
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
        declutr_args['input_dim'] = vocab_size if self.declutr_model_class == 'declutr_contrastive' else vocab_size + 1
        declutr_args['batch_size'] = self.batch_size
        declutr_args["visualize_tensors"] = self.visualize_tensors
        return declutr_args

    def update_from_model(self, declutr_model):
        self.model_dir = declutr_model.model_dir
        print(f'UPDATE: Setting Trainer model directory to {self.model_dir}.')
        self.model_path = self.model_dir

        if self.save_format == 'h5':
            self.model_path = os.path.join(self.model_path, "declutr_model.h5")

    def save_encoder(self, model):
        self.encoder_path = os.path.join(self.model_dir, "encoder")
        self.encoder_path = os.path.join(self.encoder_path, ".h5") if self.save_format == 'h5' else self.encoder_path
        encoder = model.encoder
        print(f"UPDATE: Saving encoder to {self.encoder_path}.")
        save_model(encoder, filepath=self.encoder_path, save_format=self.save_format)

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
            declutr = self.declutr_model_class(**declutr_args)
        else:
            print(f'UPDATE: DeClutr trainer building method vocabulary from code df.')
            self.build_method_vocabulary(code_df)
            masked_vocabulary_size = self.get_method_vocab_size()
            print(f'UPDATE: Building MML model with masked vocab size = {masked_vocabulary_size}, '
                  f'masked index = {self.MASKED_INDEX} declutr args = {declutr_args}.')
            declutr = self.declutr_model_class(masked_vocabulary_size=masked_vocabulary_size, masked_token=self.MASKED_INDEX,
                                               **declutr_args)

        declutr.compile(optimizer=Adam(), loss=self.loss, metrics=self.metrics, run_eagerly=True)
        self.update_from_model(declutr)

        # Save trainer to the model directory for easy coordination during later usage\loading.
        self.save_to_model_dir()
        self.tensor_visualizer = declutr.tensor_visualizer
        return declutr

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
        tensorboard = TensorBoard(log_dir=self.tensorboard_dir, write_images=True, update_freq='batch')
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_dir, 'checkpoint'))
        callbacks = [csv_logger, tensorboard, checkpoint]

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
                  f' validation steps = {val_steps}, callbacks = {callbacks}')
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

    def train_model(self, declutr_model, document_df):
        '''
        For each epoch, create new anchor spans + positive and negative samples from the document df.
        Then use these sequences as inputs and outputs for training the declutr model.
        '''

        document_df = self.tokenize_document_df(document_df)
        document_df = self.filter_documents_by_size(document_df)
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
                print(f'UPDATE: Batch count = {self.batch_count}, train count = {train_steps}, val steps = {val_steps}')
                self.train_over_grid_search(declutr_model, dataset_train, dataset_val, train_steps, val_steps)

