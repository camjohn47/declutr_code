import shutil
from glob import glob
import os
import sys

import json
import pandas as pd

import re
import gzip

from tensorflow.keras.utils import Progbar

class CodeParser():
    '''
    A class for searching and parsing through scripts containing code to build dataframes for DeClutr
    model sequence processing. It's designed for compatibility with the wonderful CodeSearchNet project
    (https://github.com/github/CodeSearchNet), which provides clean JSON data for millions of scripts
    taken from Github. The datasets include code from 6 different programming languages, including Python,
    Java and C++.
    '''

    PROGRAMMING_LANGUAGE_TO_EXTENSION = dict(python='.py', java='.java', c=".c", all='.*')
    EXTENTSION_TO_PROGRAMING_LANGUAGE = {extension: language for language, extension in PROGRAMMING_LANGUAGE_TO_EXTENSION.items()}
    CODE_EXTENSIONS = [".py", ".java", ".c"]

    def __init__(self, code_type='all', subset='train'):
        self.code_type = code_type
        codesearch_prefix = f'{self.code_type}_{subset}_' if self.code_type != 'all' else f'_{subset}_'
        self.is_codesearch_payload = lambda path: codesearch_prefix in path

        if self.code_type not in self.PROGRAMMING_LANGUAGE_TO_EXTENSION:
            print(f'ERROR: Code type {self.code_type} not in code type to extension {self.PROGRAMMING_LANGUAGE_TO_EXTENSION}.')
            sys.exit(1)

        self.code_extension = self.PROGRAMMING_LANGUAGE_TO_EXTENSION[self.code_type]

    def get_code_search_paths(self, code_directory, extension):
        json_paths = glob(os.path.join(code_directory, f'*{extension}'))
        codesearch_paths = list(filter(self.is_codesearch_payload, json_paths))
        return codesearch_paths

    def unzip_payload_gzip(self, payload_path):
        if '.jsonl' not in payload_path:
            print(f'WARNING: Non-jsonl file passed to unzip method. ')
            return
        elif '.gz' not in payload_path:
            print(f'WARNING: Non-gzip file passed to unzip method. ')
            return

        jsonl_path = payload_path.replace('.gz', '')

        # Write to jsonl path only if the file doesn't already exist.
        if not os.path.exists(jsonl_path):
            content = gzip.open(payload_path, 'r').read()
            print(f'UPDATE: Writing content to jsonl path = {jsonl_path} from gzip path = {payload_path}.')
            file = open(jsonl_path, 'wb')
            file.write(content)

    def get_all_script_paths(self, script_directory):
        script_path_pattern = os.path.join(script_directory, f'*{self.code_extension}')
        script_paths = []

        if self.code_type != 'all':
            script_paths = glob(script_path_pattern)
        else:
            for extension in self.CODE_EXTENSIONS:
                extension_paths = glob(os.path.join(script_directory, f"*{extension}"))
                script_paths += extension_paths

        return script_paths

    def unpack_gzip_paths(self, gzip_paths):
        for gzip_path in gzip_paths:
            self.unzip_payload_gzip(gzip_path)

    def code_directory_to_df(self, script_directory, shuffle=True):
        '''
        Finds scripts of the parser's code type in input <script_directory>. Then, read these scripts
        into a document-based dataframe for use by a DeClutr model.
        '''

        script_paths = self.get_all_script_paths(script_directory)
        script_df = []
        script_count = len(script_paths)
        print(f'UPDATE: CodeParser reading scripts in directory = {script_directory}')
        progress_bar = Progbar(target=script_count)

        for i, script_path in enumerate(script_paths):
            script_code = open(script_path, 'r').read()
            extension = "." + script_path.split('.')[-1]
            programming_language = self.EXTENTSION_TO_PROGRAMING_LANGUAGE[extension]
            script_df.append(dict(document=script_code, script_path=script_path, script_directory=script_directory,
                                  programming_language=programming_language))
            progress_bar.update(i)

        script_df = pd.DataFrame(script_df)
        next_script_directories = [x for x in glob(os.path.join(script_directory, '*')) if os.path.isdir(x)]

        for next_script_directory in next_script_directories:
            new_script_df = self.code_directory_to_df(next_script_directory)
            script_df = pd.concat([script_df, new_script_df])

        code_search_gzip_paths = self.get_code_search_paths(script_directory, ".gz")
        self.unpack_gzip_paths(code_search_gzip_paths)
        code_search_paths = self.get_code_search_paths(script_directory, ".jsonl")

        if code_search_paths:
            print(f'UPDATE: CodeSearch paths = {code_search_paths} found. Beginning parsing.')
            code_search_dfs = [self.parse_codesearch_payload(code_search_path) for code_search_path in code_search_paths]
            code_search_df = pd.concat(code_search_dfs)
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
        code_search_df (DataFrame): Document-based dataframe containing scripts and other meta data
                                    for the scripts found in input JSONL file.
        '''

        file = open(codesearch_payload, 'r')
        payloads = file.read()
        payloads = [json.loads(payload) for payload in payloads.splitlines()]
        codesearch_rows = [dict(document=payload['code'], script_path=payload['path'], script_directory=payload["url"],
                                programming_language=payload["language"]) for payload in payloads]
        code_search_df = pd.DataFrame(codesearch_rows)
        return code_search_df

    #def build_method_vocabulary(self, code_df, ):






