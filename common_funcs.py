import re
import time

import os

import dill

import tensorflow as tf

import pandas as pd

from code_parser import CodeParser

# NOTE: get_rank() will fail if not running eagerly.
tf.executing_eagerly()

get_default_feature_cols = lambda feature_count: [f"feature_{i}" for i in range(feature_count)]
cast_tf_tokens = lambda tokens: tf.cast(tokens, dtype=tf.int32)

# Get arg parse abbreviation of an arg by joining each word's first letter.
get_argparse_name = lambda arg: "-" + "".join([word[0].lower() for word in arg.split("_")])

TRAINING_DF_PATH = os.path.join("processed_data", "training_data.csv")

def find_code_df_methods(code_df):
    '''
    Builds a new "methods" column that contains methods found in each script.

    code_df (DataFrame):
    '''

    # This works for Python and Java, but not all languages. TODO: Expand this to work for all PL's.
    get_script_methods = lambda row: re.findall(r'(?<=[/(])\w+', row['script_path'])
    code_df['methods'] = code_df.apply(get_script_methods, axis=1)
    return code_df

def run_with_time(func, kwargs, name):
    '''
    Decorator for timing a labeled function and returning its output.
    '''

    print(f"UPDATE: Starting {name}. ")
    start = time.time()
    out = func(**kwargs)
    run_time = round(time.time() - start, 1)
    print(f'UPDATE: {name} took {run_time} seconds.')
    return out

def get_rank(tensor):
    '''
    Wrapper for tf's rank method. Returns a scalar = tensor's rank = axes count.
    '''

    try:
        rank = tf.rank(tensor).numpy()
    # Fails during stateful partition call when rank should be zero.
    except:
        rank = 0
    return rank

def get_sequence_processor(model_dir, suffix="processor.dill"):
    processor_path = os.path.join(model_dir, suffix)

    if not os.path.exists(processor_path):
        print(f"WARNING: Sequence processor path {processor_path} doesn't exist!")
        return None

    with open(processor_path, "rb") as file:
        processor = dill.load(file)
        return processor

def add_features_to_df(feature_matrix, df, feature_names=None):
    feature_count, sample_count = feature_matrix.shape

    if sample_count != len(df):
        raise ValueError(f"ERROR: Feature matrix sample count = {sample_count} != df rows = {len(df)}.")

    feature_names = feature_names if feature_names else get_default_feature_cols(feature_count)
    feature_df = pd.DataFrame(feature_matrix.tolist(), columns=feature_names)
    df = pd.concat([df, feature_df], axis=1)
    return df

def mix_lists(lists):
    '''
    Returns a list made by taking pairs from each list <a> and <b>.
    '''

    equal_length = lambda list_a, list_b: len(list_a) == len(list_b)
    lists_have_equal_length = all([equal_length(list_a, lists[0]) for list_a in lists])

    if not lists_have_equal_length:
        raise ValueError(f"ERROR: Tried to mix lists with uneqal lengths = {[len(x) for x in lists]}!")

    list_count = len(lists)
    item_count = len(lists[0])
    mixed_lists = [lists[i][j] for j in range(item_count) for i in range(list_count)]
    return mixed_lists

def get_code_df(sampling=.1, use_cached=True):
    if use_cached:
        code_df = pd.read_csv(TRAINING_DF_PATH)
    else:
        code_parser = CodeParser(programming_language="all")
        code_df = code_parser.code_directory_to_df(os.getcwd())

    code_df = code_df.sample(frac=sampling)
    return code_df
