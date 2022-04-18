import re

def find_code_df_methods(code_df):
    '''
    Returns tokens containing methods that were invoked and the indices where they occurred in
    respective sequences.
    '''

    # This works for Python and Java, but not
    get_script_methods = lambda row: re.findall(r'(?<=[/(])\w+', row['script_path'])
    code_df['methods'] = code_df.apply(get_script_methods, axis=1)
    print(code_df['methods'].value_counts())
    return code_df
