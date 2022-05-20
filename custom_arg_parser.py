from argparse import ArgumentParser
import sys

class CustomArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def join_dicts(dicts):
        joined_dict = {}

        for x in dicts:
            joined_dict = {**x, **joined_dict}

        return joined_dict

    def parse_kwargs(self, argument_args):
        kwargs = [x for x in argument_args if isinstance(x, dict)]
        kwargs = self.join_dicts(kwargs)
        argument_args.remove(kwargs)
        return argument_args, kwargs

    def get_arg_kwarg_pairs(self, arguments_args):
        arg_kwarg_pairs = []

        for argument_args in arguments_args:
            argument_args, argument_kwargs = self.parse_kwargs(argument_args)
            arg_kwarg_pair = [argument_args, argument_kwargs]
            arg_kwarg_pairs.append(arg_kwarg_pair)

        return arg_kwarg_pairs

    def add_arguments(self, arguments_args):
        argument_pairs = self.get_arg_kwarg_pairs(arguments_args)

        for args, kw_args in argument_pairs:
            print(f"UPDATE: Adding args = {args}, kwargs={kw_args}")
            self.add_argument(*args, **kw_args)

    #TODO: Single list flexibility for uniform argument subsets.
    def get_argument_subsets(self, arg_name_subsets):
        self.args = vars(self.parse_args())
        arg_subsets = []
        subset_found = lambda subset: all([arg_name in self.args for arg_name in subset])

        for arg_name_subset in arg_name_subsets:
            if not subset_found(arg_name_subset):
                print(f'ERROR: Requested arg name subset = {arg_name_subset} not found in '
                      f'CustomArgParser args = {(self.args)}.')
                sys.exit(1)

            arg_count = len(arg_name_subset)
            arg_subset = [self.args[arg] for arg in arg_name_subset] if arg_count > 1 else self.args[arg_name_subset[0]]
            arg_subsets.append(arg_subset)

        return arg_subsets
