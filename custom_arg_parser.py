from argparse import ArgumentParser

import sys

class CustomArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()

    def add_arguments(self, arguments_args, arguments_kwargs):
        args_kwargs = zip(arguments_args, arguments_kwargs)
        for args, kw_args in args_kwargs:
            self.add_argument(*args, **kw_args)

    def get_argument_subsets(self, arg_name_subsets):
        self.args = vars(self.parse_args())
        arg_subsets = []
        subset_found = lambda subset: all([arg_name in self.args for arg_name in subset])

        for arg_name_subset in arg_name_subsets:
            if not subset_found(arg_name_subset):
                print(f'ERROR: Requested arg name subset = {arg_name_subset} not found in '
                      f'CustomArgParser args = {(self.args)}.')
                sys.exit(1)

            arg_subset = {arg: self.args[arg] for arg in arg_name_subset}
            arg_subsets.append(arg_subset)

        return arg_subsets
