from glob import glob
import os
import sys

import json
import pandas as pd

import re

class CodeParser():
    '''
    A class for searching and parsing through scripts containing code to build dataframes for DeClutr
    model sequence processing. It's designed for compatibility with the wonderful CodeSearchNet project
    (https://github.com/github/CodeSearchNet), which provides clean JSON data for millions of scripts
    taken from Github. The datasets include code from 6 different programming languages, including Python,
    Java and C++.
    '''

    CODE_TYPE_TO_EXTENSION = dict(python='.py', java='.java')

    def __init__(self, code_type='python'):
        self.code_type = code_type
        codesearch_prefix = f'{self.code_type}_train_'
        self.is_codesearch_payload = lambda path: codesearch_prefix in path

        if self.code_type not in self.CODE_TYPE_TO_EXTENSION:
            print(f'ERROR: Code type {self.code_type} not in code type to extension {self.CODE_TYPE_TO_EXTENSION}.')
            sys.exit(1)

        self.code_extension = self.CODE_TYPE_TO_EXTENSION[self.code_type]

    def get_code_search_paths(self, code_directory):
        json_paths = glob(os.path.join(code_directory, '*.jsonl'))
        codesearch_paths = list(filter(self.is_codesearch_payload, json_paths))
        return codesearch_paths

    def code_directory_to_df(self, script_directory, shuffle=True):
        '''
        Finds scripts of the parser's code type in input <script_directory>. Then, read these scripts
        into a document-based dataframe for use by a DeClutr model.
        '''

        script_paths = glob(os.path.join(script_directory, f'*.{self.code_extension}'))
        script_df = []

        for script_path in script_paths:
            script_code = open(script_path, 'r').read()
            script_df.append(dict(document=script_code, script_path=script_path, script_directory=script_directory))

        script_df = pd.DataFrame(script_df)
        next_script_directories = [x for x in glob(os.path.join(script_directory, '*')) if os.path.isdir(x)]

        if next_script_directories:
           #print(f'UPDATE: Building code dataframes for newly found code directories: {next_script_directories}.')
            for next_script_directory in next_script_directories:
                new_script_df = self.code_directory_to_df(next_script_directory)
                script_df = pd.concat([script_df, new_script_df])

        code_search_paths = self.get_code_search_paths(script_directory)
        if code_search_paths:
            print(f'UPDATE: CodeSearch paths = {code_search_paths} found. Beginning parsing.')
            code_search_dfs = [self.parse_codesearch_payload(code_search_path) for code_search_path in code_search_paths]
            code_search_df = pd.concat(code_search_dfs)
            print(f'UPDATE: Codesearch paths produced {len(code_search_df)} documents.')
            script_df = pd.concat([script_df, code_search_df])

        # Shuffle the dataframe so that chunks taken from it are unbiased.
        script_df = script_df.sample(frac=1)  if shuffle else script_df
        return script_df

    @staticmethod
    def parse_codesearch_payload(codesearch_payload):
        '''
        Reads script text, script path, and script directory info from the payloads loaded from a
        CodeSearch jsonl file.

        Outputs
        code_search_df (DataFrame): Document-based dataframe containing scripts and other meta data for
                                    the scripts found in input JSONL file.
        '''

        file = open(codesearch_payload, 'r')
        payloads = file.read()
        payloads = [json.loads(payload) for payload in payloads.splitlines()]
        code_payloads = [payload['code'] for payload in payloads]
        code_paths = [payload['path'] for payload in payloads]
        code_urls = [payload['url'] for payload in payloads]
        parsed_payloads = list(zip(code_payloads, code_paths, code_urls))
        codesearch_rows = [dict(document=payload[0], script_path=payload[1], script_directory=payload[2])
                           for payload in parsed_payloads]
        code_search_df = pd.DataFrame(codesearch_rows)
        return code_search_df

    def find_code_df_methods(self, code_df):
        '''
        Returns tokens containing methods that were invoked and the indices where they occurred in respective sequences.
        '''

        # This works for Python and Java, but not
        get_script_methods = lambda row: re.findall(r'(?<=[/(])\w+', row['script_path'])
        code_df['methods'] = code_df.apply(get_script_methods, axis=1)
        print(code_df['methods'].value_counts())
        return code_df

   # def unzip_codesearch_payload(self, codesearch_payload):

    #def build_method_vocabulary(self, code_df, ):






