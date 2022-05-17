from experiment import Experiment
from common_funcs import get_argparse_name

import subprocess

import os
import shutil
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from plotly.express import line

from model_organizer import ModelOrganizer
from declutr_trainer import DeClutrTrainer
from visuals import process_fig

import re

class DeclutrExperiment(Experiment):
    EXPERIMENTS_DIRECTORY = Experiment.EXPERIMENTS_DIRECTORY
    DECLUTR_EXPERIMENTS_DIRECTORY = os.path.join(EXPERIMENTS_DIRECTORY, "declutr")
    TENSORBOARD_ARG = "-td"
    LOSS_FIG_TITLE = f"DeClutr <LOSS> for <VARIABLE_ARG> = <VARIABLE_DOMAIN_VS>"
    models_dir = DeClutrTrainer.models_dir
    DEFAULT_FIG_LAYOUT = dict(paper_bgcolor="grey", plot_bgcolor="black")
    TITLE_SUBSTITUTIONS = {"Rnn": "RNN", "Val": "Validation"}

    def __init__(self, variable_arg, variable_domain, constant_arg_vals={}, script="declutr_learning.py",
                 id_template="VARIABLE=VALUE", models_dir=None):
        self.variable_arg = variable_arg
        # Used to represent variable during script call.
        self.variable_argparse_name = get_argparse_name(self.variable_arg)
        self.variable_domain = variable_domain
        self.variable_domain_vs = " vs. ".join([var.capitalize() for var in self.variable_domain])
        self.constant_arg_vals = constant_arg_vals
        self.script = script
        self.id_template = id_template
        self.experiment_id = self.build_experiment_id()
        self.experiment_directory = os.path.join(self.EXPERIMENTS_DIRECTORY, self.experiment_id)
        Path(self.experiment_directory).mkdir(parents=True, exist_ok=True)
        self.build_config()
        self.models_dir = models_dir if models_dir else self.models_dir

    def build_experiment_id(self):
        experiment_id = f"{self.variable_arg}={'_'.join(self.variable_domain)}"
        return experiment_id

    def reset_experiment_directory(self):
        if os.path.exists(self.experiment_directory):
            print(f"WARNING: Experiment directory already exists in {self.experiment_directory}. Removing the existing data. ")
            shutil.rmtree(self.experiment_directory)

    def build_tensorboard_directory(self, model_id):
        tensorboard_directory = os.path.join(self.experiment_directory, "tensorboard_logs")
        Path(tensorboard_directory).mkdir(parents=True, exist_ok=True)
        return tensorboard_directory

    def build_config(self):
        self.config = dict(variable=self.variable_arg, variable_domain=self.variable_domain, script=self.script)
        print(f"UPDATE: Config = {self.config}")

    def get_config(self):
        return self.config

    def model_directory_from_id(self, model_id):
        model_dir = os.path.join(self.models_dir, model_id)
        return model_dir

    def collect_models_results(self):
        model_ids = self.get_model_ids()
        model_directories = list(map(self.model_directory_from_id, model_ids))
        model_id_to_value = self.get_model_id_to_variable_val()
        organizer = ModelOrganizer()
        models_results = DataFrame([])

        for i, model_directory in enumerate(model_directories):
            print(f"UPDATE: Experiment collecting model results in {model_directory}.")
            results = organizer.collect_results(model_directory)
            results[self.variable_arg] = model_id_to_value[model_ids[i]]
            models_results = pd.concat([models_results, results])

        return models_results

    def get_metric_title(self, metric):
        loss_metric_cleaned = re.sub("_", " ", metric)
        loss_metric_cleaned = loss_metric_cleaned.title()

        for word, substitution in self.TITLE_SUBSTITUTIONS.items():
            loss_metric_cleaned = re.sub(word, substitution, loss_metric_cleaned)

        return loss_metric_cleaned

    def build_loss_fig_title(self, loss_metric):
        variable_arg_title = self.get_metric_title(self.variable_arg)
        title = re.sub("<VARIABLE_ARG>", variable_arg_title, self.LOSS_FIG_TITLE)
        title = re.sub("<VARIABLE_DOMAIN_VS>", self.variable_domain_vs, title)
        loss_metric = self.get_metric_title(loss_metric)
        title = re.sub("<LOSS>", loss_metric, title)
        return title

    def update_metric_fig_layout(self, loss_fig, loss_metric):
        title = self.build_loss_fig_title(loss_metric)
        loss_fig.update_layout(title_text=title, title_x=0.5, **self.DEFAULT_FIG_LAYOUT)
        # TODO: Get exact iteration count from experiment's model training chunk size.
        loss_fig.update_xaxes(title="Chunks of 1000 Batches")
        return loss_fig

    def build_figs(self, results):
        '''
        Build and process Plotly line figure HTML's for loss and accuracy metrics.
        '''

        loss_metrics = self.get_metrics(results, "loss")
        accuracy_metrics = self.get_metrics(results, "accuracy")
        metrics = loss_metrics + accuracy_metrics

        for metric in metrics:
            loss_fig = line(results, x="chunk", y=metric, color=self.variable_arg)
            loss_fig = self.update_metric_fig_layout(loss_fig, metric)
            loss_fig_path = os.path.join(self.experiment_directory, "results", f"{metric}.html")
            Path(os.path.dirname(loss_fig_path)).mkdir(exist_ok=True, parents=True)
            process_fig(loss_fig, loss_fig_path)

    def get_variable_val_to_model_id(self):
        variable_call_args = list(map(self.get_script_call_args, self.variable_domain))
        variable_model_ids = list(map(self.get_model_id_from_call_args, variable_call_args))
        variable_to_model_id = dict(zip(self.variable_domain, variable_model_ids))
        return variable_to_model_id

    def get_model_id_to_variable_val(self):
        variable_to_model_id = self.get_variable_val_to_model_id()
        model_id_to_variable = {model_id: val for val, model_id in variable_to_model_id.items()}
        return model_id_to_variable

    @staticmethod
    def get_model_id_from_call_args(call_args):
        model_id_arg_index = call_args.index("-mi")
        model_id = call_args[model_id_arg_index + 1]
        return model_id

    def add_tensorboard_arg(self, call_args):
        model_id = self.get_model_id_from_call_args(call_args)
        tensorboard_dir = self.build_tensorboard_directory(model_id)
        call_args += [self.TENSORBOARD_ARG, tensorboard_dir]
        return call_args

    def run(self):
        self.reset_experiment_directory()

        for variable_value in self.variable_domain:
            call_args = self.get_script_call_args(variable_value=variable_value)
            call_args = list(map(str, call_args))
            call_args = self.add_tensorboard_arg(call_args)
            print(f"\nNEW EXPERIMENT: Running Declutr experiment with {self.variable_arg} = {variable_value}, "
                    f"call args = {call_args}.")
            subprocess.run(call_args)




#%%
