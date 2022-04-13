import os
import sys

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorboard.plugins.hparams import api as hp

from declutr_model import DeClutrModel

from itertools import product

from pathlib import Path

tf.executing_eagerly()
tf.config.list_physical_devices()

class DeClutrTrainer():
    metrics = [CategoricalAccuracy(), TopKCategoricalAccuracy(k=3)]
    # Hyper-parameters to be optimized with tensorboard hparams.
    EMBEDDING_DIM_DOMAIN = [10, 100]
    HP_EMBEDDING_DIM = hp.HParam('embedding_dimension', hp.Discrete(EMBEDDING_DIM_DOMAIN))
    HPARAMS = [HP_EMBEDDING_DIM]

    def __init__(self, epoch_size=1000, epoch_count=3, train_split=.75, metrics=[]):
        self.epoch_size = epoch_size
        self.epoch_count = epoch_count
        self.train_split = train_split

        # Overwrite default training metrics if they're provided.
        self.metrics = metrics if metrics else self.metrics

    def build_hparam_callback(self, log_dir, hparam_vals):
        '''
        hparam_vals (dict): Mapping from hparam to the specific value it's taking for this experiment.
        '''

        hparam_val_dic = {hparam: hparam_vals[i] for i, hparam in enumerate(self.HPARAMS)}
        print(f'UPDATE: hparam val dic {hparam_val_dic}')
        hparam_callback = hp.KerasCallback(log_dir, hparam_val_dic)
        return hparam_callback

    def set_seq_processor(self, seq_processor):
        print(f'UPDATE: setting Trainers sequence processor to {seq_processor}.')
        self.seq_processor = seq_processor

    def update_declutr_encoder_args(self, declutr_args, vocab_size):
        declutr_args['input_dim'] = vocab_size
        print(f'UPDATE: Declutr args after updating embedding encoder dims: {declutr_args}.')

        return declutr_args

    def build_model_from_processor(self, seq_processor, declutr_args={}):
        self.set_seq_processor(seq_processor)
        vocab_size = self.seq_processor.get_vocab_size()
        print(f'UPDATE: Vocab size for code df = {vocab_size}.')
        declutr_args = self.update_declutr_encoder_args(declutr_args, vocab_size)
        batch_size = self.seq_processor.batch_size
        declutr = DeClutrModel(batch_size=batch_size, **declutr_args)
        declutr.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=self.metrics, run_eagerly=True)
        print(f'UPDATE: Setting Trainer model directory to {declutr.model_dir}.')
        self.model_dir = declutr.model_dir
        return declutr

    def get_model_callbacks(self, log_dir='logs', hparam_tuning=True, hparam_vals=None):
        '''
        Get keras callbacks for helping with model training and providing analysis data for it.

        Inputs
        model_dir:
        log_dir:
        hparam_tuning:
        '''

        # Set CSV log directory to be chunk, epoch dependent to avoid overwriting when training over
        # different chunks and epochs.
        log_path = os.path.join(self.log_dir, 'fit_log.csv')
        csv_logger = CSVLogger(filename=log_path, append=True)
        tensorboard = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_dir, 'checkpoint'))
        callbacks = [csv_logger, tensorboard, checkpoint]

        if hparam_tuning:
            if not hparam_vals:
                print(f'ERROR: No hparam vals provided for hparam tuning! ')
                sys.exit(1)

            hparam_callback = self.build_hparam_callback(log_dir, hparam_vals)
            callbacks.append(hparam_callback)

        return callbacks

    def train_over_grid_search(self, declutr, dataset_train, dataset_val, train_steps, val_steps):
        '''
        Train and validate over all values in an HParam domain with the current chunk's datasets.
        '''

        hparam_domains = [hparam.domain.values for hparam in self.HPARAMS]
        hparam_combos = list(product(hparam_domains)) if len(hparam_domains) > 1 else [[i] for i in hparam_domains[0]]

        # Only one epoch fed into fit call at a time to refresh negative samples. If we used epochs=self.epoch_count instead,
        # the same negative samples would be used which would almost surely result in lower quality embeddings.
        # LATER: Quantify this improvement.
        for hparam_combo in hparam_combos:
            callbacks = self.get_model_callbacks(hparam_vals=hparam_combo)
            print(f'UPDATE: Beginning training with HParam combo {hparam_combo}. Training steps = {train_steps},'
                  f' validation steps = {val_steps}')
            declutr.fit(x=dataset_train, epochs=1, callbacks=callbacks, validation_data=dataset_val,
                        steps_per_epoch=train_steps, validation_steps=val_steps)

    def build_fit_directory(self):
        self.log_dir = os.path.join(self.model_dir, f'chunk_{self.chunk}')
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def update_indices(self, chunk, epoch):
        self.chunk = chunk
        self.epoch = epoch

    def train_model(self, declutr, document_df):
        '''
        For each epoch, create new anchor spans + positive and negative samples from the document df. Then
        use these sequences as inputs and outputs for training the declutr model.
        '''

        for chunk, chunk_df in enumerate(self.seq_processor.generate_df_in_chunks(document_df, documents_per_chunk=self.epoch_size)):
            for epoch in range(self.epoch_count):
                print(f'UPDATE: Beginning DeClutr training for chunk {chunk}, epoch {epoch}, chunk size = {len(chunk_df)}.')
                self.update_indices(chunk, epoch)
                self.build_fit_directory()

                # Cardinality of -2 means that the cardinality couldn't be computed.
                dataset = self.seq_processor.get_declutr_training_dataset(chunk_df)
                batch_count = self.seq_processor.cardinality_estimate
                train_steps = int(batch_count * self.train_split)
                val_steps = batch_count - train_steps
                dataset_train = dataset.take(train_steps)
                dataset_val = dataset.skip(train_steps)
                print(f'UPDATE: Batch count = {batch_count}, train count = {train_steps}, val steps = {val_steps}')
                self.train_over_grid_search(declutr, dataset_train, dataset_val, train_steps, val_steps)

