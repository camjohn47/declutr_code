from experiment import Experiment
from modules.common_funcs import get_argparse_name

import subprocess

import os
import shutil
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from plotly.express import line

from model_organizer import ModelOrganizer
from declutr_trainer import DeClutrTrainer
from modules.visuals import process_fig

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
                 id_template="VARIABLE=VALUE", models_dir=None, add_constants_to_id=None):
        self.variable_arg = variable_arg
        # Used to represent variable during script call.
        self.variable_argparse_name = get_argparse_name(self.variable_arg)
        self.variable_domain = [str(var) for var in variable_domain]
        self.variable_domain_vs = " vs. ".join([var.capitalize() for var in self.variable_domain])
        self.constant_arg_vals = constant_arg_vals
        self.script = script
        self.id_template = id_template
        self.models_dir = models_dir if models_dir else self.models_dir
        self.post_query_script = self.get_full_execution_path("codesearch_evaluation.py")
        self.add_constants_to_id = add_constants_to_id
        self.initialize_experiment_directory(experiment_class="DeClutr")

    def build_tensorboard_directory(self, model_id):
        tensorboard_directory = os.path.join(self.experiment_directory, "../models", model_id, "tensorboard_logs")
        Path(tensorboard_directory).mkdir(parents=True, exist_ok=True)
        return tensorboard_directory

    def build_config(self):
        '''
        Build dictionary with experiment parameters.
        '''
        self.config = dict(variable=self.variable_arg, variable_domain=self.variable_domain, script=self.script,
                           constant_arg_vals=self.constant_arg_vals)
        print(f"UPDATE: Declutr Experiment built config = {self.config}.")

    def save_config(self):
        '''
        Save config to a readable txt file in experiment dir.
        '''

        self.config_path = os.path.join(self.experiment_directory, "config.txt")

        with open(self.config_path, "w") as file:
            print(f"UPDATE: Declutr Experiment writing config to {self.config_path}.")
            file.write(str(self.config))

    def collect_models_results(self):
        '''
        Outputs
        models_results (DataFrame): A df with performance metrics for different experiment trials.
        '''

        model_ids = self.get_model_ids()
        model_directories = self.get_model_directories()
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

    def build_figs(self, results, metric):
        '''
        Build and process Plotly line figure HTML's for loss and accuracy metrics.
        '''

        metrics = self.get_metrics(results, metric)

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
        '''
        Add experiment's tensorboard directory to the call args. This ensures tensorboard visuals for different model runs
        will be available in same directory.
        '''

        model_id = self.get_model_id_from_call_args(call_args)
        tensorboard_dir = self.build_tensorboard_directory(model_id)
        call_args += [self.TENSORBOARD_ARG, tensorboard_dir]
        return call_args

    def add_experiment_id_arg(self, call_args):
        call_args += ["-ei", self.experiment_id]
        return call_args

    def run(self):
        '''
        Train different DeClutr models over the dependent variable values\domain.
        '''

        for variable_value in self.variable_domain:
            call_args = self.get_script_call_args(variable_value=variable_value)
            call_args = list(map(str, call_args))
            call_args = self.add_tensorboard_arg(call_args)
            call_args = self.add_experiment_id_arg(call_args)
            print(f"\nNEW EXPERIMENT: Running Declutr experiment with {self.variable_arg} = {variable_value}, call args"
                  f" = {call_args}.")
            subprocess.run(call_args, check=True)









#%%
