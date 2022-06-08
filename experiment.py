import abc
from abc import abstractmethod
import sys
import os
sys.path.append(os.path.join("..", ".."))
from common_funcs import get_argparse_name, mix_lists

import re

class Experiment(abc.ABC):
    EXPERIMENTS_DIRECTORY = "experiments"
    MODELS_DIR = "models"

    def __int__(self, variable_arg, variable_domain, constant_arg_vals, script, models_dir=None, id_template="VARIABLE=VALUE",
                add_constants_to_id=None):
        self.variable_arg = variable_arg
        self.variable_domain = variable_domain
        self.constant_arg_vals = constant_arg_vals
        self.constant_args = list(self.constant_arg_vals.keys())
        self.script = script
        self.id_template = id_template
        self.models_dir = models_dir if models_dir else self.MODELS_DIR

        # Constants to include in experiment id.
        self.add_constants_to_id = add_constants_to_id
        self.check_constants()

        #TODO: Config building and saving in experiment directory.
        self.build_config()
        self.save_config()

    @abstractmethod
    def run(self):
        pass

    def check_constants(self):
        exists_unlisted_constant = lambda constants: any([constant not in self.constant_arg_vals for constant in constants])

        # Checks if there's any requested constant to include in experiment id that's not specified in <constant_arg_vals>.
        if self.add_constants_to_id and exists_unlisted_constant(self.add_constants_to_id):
            raise ValueError("ERROR: Constants in add constants to id that aren't present in constant arg vals!")

    def collect_models_results(self):
        pass

    @abstractmethod
    def build_config(self):
        pass

    @abstractmethod
    def save_config(self):
        pass

    def build_experiment_id(self):
        experiment_id = f"{self.variable_arg}={'_vs_'.join(self.variable_domain)}"

        if self.add_constants_to_id:
            for constant in self.add_constants_to_id:
                constant_val = self.constant_arg_vals[constant]
                experiment_id = "_".join([experiment_id, f"{constant}={constant_val}"])

        return experiment_id

    # Methods used for generating subprocess arguments of different experiment trials/values.
    def get_model_id(self, variable_value):
        model_id = re.sub("VARIABLE", self.variable_arg, self.id_template)
        model_id = re.sub("VALUE", variable_value, model_id)
        return model_id

    def model_directory_from_id(self, model_id):
        model_dir = os.path.join(self.models_dir, model_id)
        return model_dir

    def get_model_directories(self):
        model_ids = self.get_model_ids()
        model_directories = list(map(self.model_directory_from_id, model_ids))
        return model_directories

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

    @staticmethod
    def get_metrics(results, keyword):
        loss_metrics = [col for col in results.columns if keyword in col]
        return loss_metrics

    @staticmethod
    #TODO: Remove this and replace usages with set_path_to_main.
    def get_full_execution_path(execution_script):
        '''
        Ensure that the execution script is executed in declutr_code main directory.
        '''

        curr_dir = os.getcwd()
        curr_dir_head, curr_dir_tail = os.path.split(curr_dir)

        if curr_dir_tail == "declutr_code":
            full_execution_path = os.path.join(curr_dir, execution_script)
        # Shift up in the working directory tree if not currently in main directory.
        else:
            full_execution_path = os.path.join(curr_dir_head, execution_script)

        return full_execution_path






