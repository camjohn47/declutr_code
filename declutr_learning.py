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
    arg_parser.add_argument('-em', "--encoder_model", choices=["rnn", "transformer"], default='rnn')
    arg_parser.add_argument("-s", "--sampling", type=float, default=1)
    arg_parser.add_argument("-sf", "--save_format", type=str, default="tf")
    arg_parser.add_argument("-vt", "--visualize_tensors", type=bool, default=False)
    arg_parser.add_argument("-ea", "--encoder_architecture", choices=["lstm", "gru"], required=False, default="lstm",
                            help="RNN architecture of the DeClutr model's encoder.")
    arg_parser.add_argument("-ss", "--sequence_summarization", choices=["average", "lstm", "gru"], required=False, default="average",
                            help="Method of summarizing sequence along time\word placement dimension. Original DeClutr paper"
                                 "uses averaging.")
    arg_parser.add_argument("-nw", "--num_words", required=False, type=int, default=None,
                            help="Number of most popular words in vocabulary to keep.")
    #TODO: Add cross functionality support for scripts and docstrings. For starters, through a QA architecture.
    arg_parser.add_argument("-tc", "--text_column", required=False, default="code", choices=["code", "docstring"],
                            help="Column in code dataframe whose text will be analyzed. ")
    arg_parser.add_argument("-ed", "--embedding_dimensions", required=False, default=100, type=int,
                            help="Length of each embedding vector fed to encoder layer.")
    arg_parser.add_argument("-td", "--tensorboard_dir", required=False, default=DeClutrTrainer.tensorboard_dir,
                            help="Directory where tf model fit/validation logs are saved for later analysis.")
    return arg_parser

def get_encoder_config(args):
    encoder_config = dict(embedding_args=dict(output_dim=args["embedding_dimensions"]))

    if args["encoder_model"] == "rnn":
        encoder_config["architecture"] = args["encoder_architecture"]

    return encoder_config

def get_args():
    arg_parser = get_arg_parser()
    args = vars(arg_parser.parse_args())
    code_parser_args = dict(programming_language=args["programming_language"])
    tokenizer_args = dict(num_words=args["num_words"])
    sequence_processor_args = dict(loss_objective=args["loss_objective"], tokenizer_args=tokenizer_args)
    sampling = args["sampling"]
    encoder_config = get_encoder_config(args)
    declutr_trainer_args = dict(sequence_processor_args=sequence_processor_args, encoder_model=args["encoder_model"],
                                save_format=args["save_format"], visualize_tensors=args["visualize_tensors"], sampling=sampling,
                                tensorboard_dir=args["tensorboard_dir"], text_column=args["text_column"])
    declutr_model_args = dict(encoder_model=args["encoder_model"], model_id=args["model_id"], encoder_config=encoder_config,
                              sequence_summarization=args["sequence_summarization"])
    declutr_trainer_args["code_parser_args"] = code_parser_args
    return sequence_processor_args, declutr_trainer_args, declutr_model_args, sampling

# Parse scripts and build a code dataframe from them.
sequence_processor_args, declutr_trainer_args, declutr_model_args, sampling = get_args()

# Initialize declutr processor, model and training helper.
declutr_trainer = DeClutrTrainer(**declutr_trainer_args)
curr_directory = os.getcwd()
declutr_trainer.start_training_from_directory(code_directory=curr_directory, declutr_args=declutr_model_args)

