import abc
from abc import abstractmethod
import sys
import os
sys.path.append(os.path.join("..", ".."))
from common_funcs import get_argparse_name, mix_lists

import re

#TODO: Experiment abstract class that can be used for NEL, NER, CodeSearch, ... experiments.
class Experiment(abc.ABC):
    EXPERIMENTS_DIRECTORY = "experiments"

    def __int__(self, variable_arg, variable_domain, constant_arg_vals, script, constant_args, id_template="VARIABLE=VALUE"):
        self.variable_arg = variable_arg
        self.variable_domain = variable_domain
        self.constant_arg_vals = constant_arg_vals
        self.script = script
        self.id_template = id_template
        self.constant_args = constant_args

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def collect_results(self):
        pass

    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def build_experiment_id(self):
        pass

    # Methods used for generating subprocess arguments of different experiment trials/values.
    def get_model_id(self, variable_value):
        model_id = re.sub("VARIABLE", self.variable_arg, self.id_template)
        model_id = re.sub("VALUE", variable_value, model_id)
        return model_id

    def get_argparse_constants(self):
        constant_args = list(self.constant_arg_vals.keys())
        argparse_constants = list(map(get_argparse_name, constant_args))
        return argparse_constants

    def get_argparse_names(self, constant_argparse_names):
        variable_argparse_name = get_argparse_name(self.variable_arg)
        argparse_names = constant_argparse_names + [variable_argparse_name]
        return argparse_names

    def get_argparse_values(self, variable_value):
        constant_values = list(self.constant_arg_vals.values())
        argparse_values = constant_values + [variable_value]
        return argparse_values

    def add_prefix(self, name_values):
        script_call_args = ["python", self.script] + name_values
        return script_call_args

    def add_model_id(self, script_call_args, variable_value):
        model_id = self.get_model_id(variable_value)
        script_call_args += ["-mi", model_id]
        return script_call_args

    def get_script_call_args(self, variable_value):
        '''
        Inputs
        variable_value: Current value of the experiment's variable.

        Outputs
        Argument list for executing subprocess of this experiment trial.
        '''

        constant_argparse_names = self.get_argparse_constants()
        argparse_names = self.get_argparse_names(constant_argparse_names)
        argparse_values = self.get_argparse_values(variable_value)
        name_values = mix_lists([argparse_names, argparse_values])
        script_call_args = self.add_model_id(name_values, variable_value)
        script_call_args = self.add_prefix(script_call_args)
        return script_call_args

    def get_model_ids(self):
        model_ids = list(map(self.get_model_id, self.variable_domain))
        return model_ids





