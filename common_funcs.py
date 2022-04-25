import re
import time

import tensorflow as tf

# NOTE: get_rank() will fail if not running eagerly.
tf.executing_eagerly()

def find_code_df_methods(code_df):
    '''
    Returns tokens containing methods that were invoked and the indices where they occurred in
    respective sequences.
    '''

    # This works for Python and Java, but not
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

