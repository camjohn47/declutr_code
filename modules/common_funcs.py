import re
import time

import os
from os.path import join, dirname
from ast import literal_eval

import dill

import tensorflow as tf

import pandas as pd

from classes.code_parser import CodeParser

from sklearn.decomposition import PCA

from pathlib import Path

# NOTE: get_rank() will fail if not running eagerly.
tf.executing_eagerly()

get_default_feature_cols = lambda feature_count: [f"feature_{i}" for i in range(feature_count)]
cast_tf_tokens = lambda tokens: tf.cast(tokens, dtype=tf.int32)

# Get arg parse abbreviation of an arg by joining each word's first letter.
get_argparse_name = lambda arg: "-" + "".join([word[0].lower() for word in arg.split("_")])
make_nested_dirs = lambda dir: Path(dir).mkdir(exist_ok=True, parents=True)

def process_fig(fig, path):
    fig.show()
    make_path_directories(path)
    print(f"UPDATE: Saving html to {path}")
    fig.write_html(path)

def process_df(df, df_name, csv_path):
    print(f"UPDATE: Here's the info for df={df_name}. Saving to {csv_path}.")
    df.info()
    make_path_directories(csv_path)
    df.to_csv(csv_path, index=False)

def set_path_to_main(path):
    '''
    Ensure that script is executed in declutr_code main.
    '''

    curr_dir = os.getcwd()
    curr_dir_head, curr_dir_tail = os.path.split(curr_dir)

    # Shift up in the working directory tree if not currently in main directory.
    full_execution_path = join(curr_dir, path) if curr_dir_tail == "declutr_code" else os.path.join(curr_dir_head, path)

    return full_execution_path

MAIN_DIR = dirname(os.getcwd())
TRAINING_DF_PATH = join(MAIN_DIR, "processed_data", "training_data.csv")
TRAINING_DF_PATH = set_path_to_main(TRAINING_DF_PATH)

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

def have_equal_shapes(tensor_a, tensor_b):
    all_dims_are_equal = lambda x, y: all([x.shape[i] == y.shape[i] for i in range(get_rank(y))])

    if get_rank(tensor_a) != get_rank(tensor_b):
        return False
    elif all_dims_are_equal(tensor_a, tensor_b):
        return True
    else:
        return False

def get_sequence_processor(model_dir, suffix="processor.dill"):
    processor_path = os.path.join(MAIN_DIR, model_dir, suffix)

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

def convert_tokens_to_int(document_df):
    '''
    Converts each string list of tokens into a list of ints. Ie "[1, 2, 3, ...]" -> [1, 2, 3, ...]. Needed when using
    tokens in code df loaded from CSV.
    '''

    document_df["document_tokens"] = document_df.apply(lambda row: [int(token) for token in literal_eval(row["document_tokens"])], axis=1)
    return document_df

def drop_nans(df, columns):
    df.dropna(subset=columns, inplace=True)
    return df

def get_code_df(sampling=.1, use_cached=True):
    '''
    Return a code-based dataframe with natural language description and script columns.
    '''

    if use_cached:
        code_df = pd.read_csv(TRAINING_DF_PATH)
    else:
        code_parser = CodeParser(programming_language="all")
        code_df = code_parser.code_directory_to_df(os.getcwd())

    code_df = drop_nans(code_df, columns=["code", "docstring"])
    code_df = code_df.sample(frac=sampling)
    code_df = convert_tokens_to_int(code_df)
    return code_df

def reduce_matrix(matrix, target_dims):
    matrix = PCA(n_components=target_dims).fit_transform(matrix)
    return matrix

def drop_nan_text(df, text_column):
    '''
    Check text availability and drop rows with nan text.
    '''

    if text_column not in df.columns:
        raise ValueError(f"ERROR: {text_column} not in df columns = {df.columns}.")

    total_docs = len(df)
    df.dropna(subset=[text_column], inplace=True)
    nan_docs = total_docs - len(df)

    if nan_docs:
        print(f'WARNING: {nan_docs} documents dropped with nan.')

    return df

#TODO: Add text column argument to tokenize_df() in SequenceProcessor so this can be removed.
def tokenize_df_wrapper(sequence_processor, document_df, text_column):
    '''
    Add a tokenized documents column to the input <document_df>. Each document is tokenized with the processor's tokenizer.
    '''

    documents = document_df[text_column].values

    if sequence_processor.pretrained_tokenizer:
        documents = documents.tolist()
        documents_tokenized = list(map(sequence_processor.tokenizer.tokenize, documents))
        document_df['document_tokens'] = list(map(sequence_processor.tokenizer.convert_tokens_to_ids, documents_tokenized))
    else:
        document_df['document_tokens'] = sequence_processor.tokenizer.texts_to_sequences(documents)

    return document_df

def make_path_directories(path):
    '''
    Robustly makes nested directories found in a path.
    '''

    dir_name = os.path.dirname(path)
    Path(dir_name).mkdir(exist_ok=True, parents=True)




