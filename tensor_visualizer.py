from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

import os
import sys

import tensorflow as tf
import plotly.express as px
from visuals import make_3d_scatter

import pandas as pd
from pathlib import Path

class TensorVisualizer():
    '''
    A class for visualizing layer outputs from a tensorflow model and de-bugging final outputs.
    '''

    def __init__(self, tf_model_dir, num_axes=3, save_visuals=False):
        self.tf_model_dir = tf_model_dir
        self.tf_model = load_model(self.tf_model_dir)
        self.num_axes = num_axes
        self.columns = ["sequence", "batch", "dimension", "value"] if self.num_axes ==3 else ["sequence", "batch", "value"]
        self.visuals_df = pd.DataFrame([], columns=self.columns)
        self.record_method = self.record_3D_outputs if self.num_axes == 3 else self.record_2D_outputs
        self.visuals_directory = os.path.join(tf_model_dir, "tensor_visuals")
        Path(self.visuals_directory).mkdir(exist_ok=True, parents=True)
        self.batch_count = 0
        self.prev_visuals_count = 0
        self.save_visuals = save_visuals

    def update_visuals_df(self, update_df, batch_count):
        self.visuals_df = pd.concat([self.visuals_df, update_df])
        self.batch_count += batch_count

    def record_2D_outputs(self, outputs, labels):
        '''
        Convert the tensor into plotting coordinates and add these coordinates to the visuals dataframe.
        '''

        batch_count, sequence_count = outputs.shape
        visuals_df_rows = []

        for i in range(batch_count):
            for j in range(sequence_count):
                value = outputs[i, j].numpy()
                visuals_df_row = dict(sequence=j, batch=i + self.batch_count, value=value)
                visuals_df_rows.append(visuals_df_row)

        visuals_df = pd.DataFrame(visuals_df_rows, columns=self.columns)
        self.update_visuals_df(visuals_df, batch_count)

    # TODO: Use correct label information for each batch in the visuals.
    def record_3D_outputs(self, outputs, labels):
        '''
        Convert the tensor into plotting coordinates and add these coordinates to the visuals dataframe.
        '''

        batch_count, sequence_count, dims = outputs.shape
        visuals_df_rows = []

        for i in range(batch_count):
            for j in range(sequence_count):
                for dim in range(dims):
                    value = outputs[i, j, dim].numpy()
                    visuals_df_row = dict(sequence=j, batch=i + self.batch_count, dimension=dim, value=value)
                    visuals_df_rows.append(visuals_df_row)

        visuals_df = pd.DataFrame(visuals_df_rows, columns=self.columns)
        visuals_df.info()
        self.update_visuals_df(visuals_df, batch_count)

    def record_training_outputs(self, outputs, labels):
        outputs_rank = tf.rank(outputs)

        if outputs_rank != self.num_axes:
            print(f'WARNING: Outputs rank = {outputs_rank} != TensorVisualizer num axes. Skipping batch.')
            return

        self.record_method(outputs, labels)

    def get_save_path(self):
        save_path = os.path.join(self.visuals_directory, f"output_scatter_{self.prev_visuals_count}.png")
        return save_path

    def make_visuals(self):
        '''
        Create a Plotly 3D scatter of the output tensor history found in visuals df.
        '''

        save_path = self.get_save_path() if self.save_visuals else None
        xyz_columns = self.columns
        print(f'UPDATE: Visuals df before scatter')
        self.visuals_df.info()
        make_3d_scatter(self.visuals_df, xyz_columns=xyz_columns, save_path=save_path)

    def reset_visuals_df(self):
        if self.visuals_df.any:
            self.make_visuals()

        self.prev_visuals_count += 1
        self.visuals_df = pd.DataFrame([], columns=self.columns)
        self.batch_count = 0

    def visualize_outputs(self, batch_count, visuals_dir, layer):
        if layer not in self.tf_model.layers:
            print(f'ERROR: Requested layer {layer} not found in tf model layers = {self.tf_model.layers}')
            sys.exit(1)

        return []

class VisualizerCallBack(Callback):
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer

    def on_train_end(self, logs=None):
        print(f'UPDATE: Train end for visualizer callback. Resetting visuals df.')
        self.visualizer.reset_visuals_df()

    def on_train_epoch_end(self, epoch, logs=None):
        print(f'UPDATE: Epoch {epoch} end for visualizer callback. Resetting visuals df.')
        self.visualizer.reset_visuals_df()






