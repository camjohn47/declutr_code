from experiment import Experiment
from common_funcs import get_argparse_name

import subprocess

import os
import shutil
from pathlib import Path

class DeclutrExperiment(Experiment):
    EXPERIMENTS_DIRECTORY = Experiment.EXPERIMENTS_DIRECTORY
    DECLUTR_EXPERIMENTS_DIRECTORY = os.path.join(EXPERIMENTS_DIRECTORY, "declutr")
    TENSORBOARD_ARG = "-td"

    def __init__(self, variable_arg, variable_domain, constant_arg_vals={}, script="declutr_learning.py",
                 id_template="VARIABLE=VALUE"):
        self.variable_arg = variable_arg
        # Used to represent variable during script call.
        self.variable_argparse_name = get_argparse_name(self.variable_arg)
        self.variable_domain = variable_domain
        self.constant_arg_vals = constant_arg_vals
        self.script = script
        self.id_template = id_template
        self.experiment_id = self.build_experiment_id()
        self.build_experiment_directory()
        self.build_config()

    def build_experiment_id(self):
        experiment_id = f"{self.variable_arg}={'_'.join(self.variable_domain)}"
        return experiment_id

    def build_experiment_directory(self):
        self.experiment_directory = os.path.join(self.EXPERIMENTS_DIRECTORY, self.experiment_id)

        if os.path.exists(self.experiment_directory):
            print(f"WARNING: Experiment directory already exists in {self.experiment_directory}. Removing the existing data. ")
            shutil.rmtree(self.experiment_directory)

        Path(self.experiment_directory).mkdir(parents=True, exist_ok=True)

    def build_tensorboard_directory(self, model_id):
        tensorboard_directory = os.path.join(self.experiment_directory, "tensorboard_logs", model_id)
        Path(tensorboard_directory).mkdir(parents=True, exist_ok=True)
        return tensorboard_directory

    def build_config(self):
        self.config = dict(variable=self.variable_arg, variable_domain=self.variable_domain, script=self.script)
        print(f"UPDATE: Config = {self.config}")

    def get_config(self):
        return self.config

    def collect_results(self):
        pass

    def add_tensorboard_arg(self, call_args):
        model_id_arg_index = call_args.index("-mi")
        model_id = call_args[model_id_arg_index + 1]
        tensorboard_dir = self.build_tensorboard_directory(model_id)
        call_args += [self.TENSORBOARD_ARG, tensorboard_dir]
        return call_args

    def run(self):
        for variable_value in self.variable_domain:
            call_args = self.get_script_call_args(variable_value=variable_value)
            call_args = list(map(str, call_args))
            call_args = self.add_tensorboard_arg(call_args)
            print(f"\nNEW EXPERIMENT: Running Declutr experiment with {self.variable_arg} = {variable_value}, "
                    f"call args = {call_args}.")
            subprocess.run(call_args)


