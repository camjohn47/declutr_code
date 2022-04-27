import os

from declutr_trainer import DeClutrTrainer
from code_parser import CodeParser
import argparse

#TODO: Simplify this using CustomArgParser.

def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    programming_languages = CodeParser.PROGRAMMING_LANGUAGE_TO_EXTENSION.keys()
    arg_parser.add_argument("-mi", "--model_id", type=str, required=True)
    arg_parser.add_argument("-pl", "--programming_language", choices=programming_languages, default="all",
                            help="The programming language whose scripts will be ingested in pipeline.")
    arg_parser.add_argument("-lo", "--loss_objective", choices=["declutr_contrastive", "declutr_masked_"],
                            default="declutr_contrastive", help="The learning objective of the NN"
                                                                " that determines its architecture.")
    arg_parser.add_argument('-e', "--encoder_model", choices=["lstm", "transformer"], default='lstm')
    arg_parser.add_argument("-s", "--sampling", type=float, default=.01)
    arg_parser.add_argument("-sf", "--save_format", type=str, default="tf")
    return arg_parser

def get_args():
    arg_parser = get_arg_parser()
    args = vars(arg_parser.parse_args())
    code_parser_args = dict(programming_language=args["programming_language"])
    sequence_processor_args = dict(loss_objective=args["loss_objective"])
    declutr_trainer_args = dict(sequence_processor_args=sequence_processor_args, encoder_model=args["encoder_model"],
                                save_format=args["save_format"])
    declutr_model_args = dict(encoder_model=args["encoder_model"], model_id=args["model_id"])
    sampling = args["sampling"]
    declutr_trainer_args["code_parser_args"] = code_parser_args
    return sequence_processor_args, declutr_trainer_args, declutr_model_args, sampling

# Parse scripts and build a code dataframe from them.
sequence_processor_args, declutr_trainer_args, declutr_model_args, sampling = get_args()

# Initialize declutr processor, model and training helper.
declutr_trainer = DeClutrTrainer(**declutr_trainer_args)
curr_directory = os.getcwd()
declutr_trainer.start_training_from_directory(code_directory=curr_directory, declutr_args=declutr_model_args)

