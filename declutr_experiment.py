from experiment import Experiment
from common_funcs import get_argparse_name

import subprocess

import re

class DeclutrExperiment(Experiment):
    def __init__(self, variable_arg, variable_domain, constant_arg_vals={}, script="declutr_learning.py",
                 id_template="declutr_experiment_VARIABLE=VALUE"):
        print(f"Declutr experiment.")
        self.variable_arg = variable_arg
        # Used to represent variable during script call.
        self.variable_argparse_name = get_argparse_name(self.variable_arg)
        self.variable_domain = variable_domain
        self.constant_arg_vals = constant_arg_vals
        self.script = script
        self.id_template = id_template
        self.build_config()

    def build_config(self):
        self.config = dict(variable=self.variable_arg, variable_domain=self.variable_domain, script=self.script)
        print(f"UPDATE: Config = {self.config}")

    def get_config(self):
        return self.config

    def collect_results(self):
        pass

    def run(self):
        for variable_value in self.variable_domain:
            call_args = self.get_script_call_args(variable_value=variable_value)
            call_args = list(map(str, call_args))
            print(f"\nNEW EXPERIMENT: Running Declutr experiment with {self.variable_arg} = {variable_value}, "
                    f"call args = {call_args}.")
            subprocess.run(call_args)


