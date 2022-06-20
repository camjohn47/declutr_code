from glob import glob
import os
import sys

import json
import pandas as pd

import gzip

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
    codesearch_columns = ["code", "path", "url", "language", "docstring"]

    # Essential columns whose nan values will be dropped from parsed df.
    DROP_NAN_COLS = ["code", "docstring"]

    def __init__(self, programming_language='all', subset='train', codesearch_columns=[]):
        self.programming_language = programming_language
        codesearch_prefix = f'{self.programming_language}_{subset}_' if self.programming_language != 'all' else f'_{subset}_'
        self.is_codesearch_payload = lambda path: codesearch_prefix in path

        if self.programming_language not in self.PROGRAMMING_LANGUAGE_TO_EXTENSION:
            print(f'ERROR: Code type {self.programming_language} not in code type to extension {self.PROGRAMMING_LANGUAGE_TO_EXTENSION}.')
            sys.exit(1)

        self.code_extension = self.PROGRAMMING_LANGUAGE_TO_EXTENSION[self.programming_language]
        self.codesearch_columns = codesearch_columns if codesearch_columns else self.codesearch_columns

    def get_code_search_paths(self, code_directory, extension):
        '''
        Get CodeSearch filenames in a directory ending in <extension>.
        '''

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
        '''
        Returns a list of all scripts that match the instance's code extension. For example:

        if <self.code_extension>=".py" ---> this method returns all python files in <script_directory>.
        '''

        script_path_pattern = os.path.join(script_directory, f'*{self.code_extension}')
        script_paths = []

        if self.programming_language != 'all':
            script_paths = glob(script_path_pattern)
        else:
            for extension in self.CODE_EXTENSIONS:
                extension_paths = glob(os.path.join(script_directory, f"*{extension}"))
                script_paths += extension_paths

        return script_paths

    def unpack_gzip_paths(self, gzip_paths):
        for gzip_path in gzip_paths:
            self.unzip_payload_gzip(gzip_path)

    def get_subdirs(self, directory):
        subdirs = [x for x in glob(os.path.join(directory, '*')) if os.path.isdir(x)]
        return subdirs

    def join_df_with_subdirectory_dfs(self, script_directory, script_df):
        '''
        Parse through subdirectories, build their dfs, and join with the current script df.
        '''

        next_script_directories = self.get_subdirs(script_directory)

        for next_script_directory in next_script_directories:
            new_script_df = self.code_directory_to_df(next_script_directory)
            script_df = pd.concat([script_df, new_script_df])

        return script_df

    def join_df_with_codesearch(self, script_directory, script_df):
        '''
        Search for CodeSearch payloads and ingest their script data into the current script_df.
        '''

        code_search_gzip_paths = self.get_code_search_paths(script_directory, ".gz")
        self.unpack_gzip_paths(code_search_gzip_paths)
        code_search_paths = self.get_code_search_paths(script_directory, ".jsonl")

        if code_search_paths:
            code_search_dfs = [self.parse_codesearch_payload(code_search_path) for code_search_path in code_search_paths]
            code_search_df = pd.concat(code_search_dfs)
            script_df = pd.concat([script_df, code_search_df])

        return script_df

    def drop_all_nans(self, script_df):
        rows_before = len(script_df)

        if not rows_before:
            return script_df

        script_df.dropna(subset=self.DROP_NAN_COLS, inplace=True)
        rows_after = len(script_df)
        dropped_rows = rows_before - rows_after

        if dropped_rows:
            print(f"UPDATE: CodeParser dropped {dropped_rows} rows with nan values in {self.DROP_NAN_COLS}.")

        return script_df

    def code_directory_to_df(self, script_directory, shuffle=True):
        '''
        Finds scripts of the parser's code type in input <script_directory>. Then, read these scripts
        into a document-based dataframe for use by a DeClutr model. This will ingest scripts that fall into these categories:

         1. scripts of the specified code extension found in <script_directory>.
         2. scripts found in (recursive) subdirectories of <script_directory>.
         3. scripts available in the CodeSearch project's JSON data.

         Inputs
         script_directory  (str): Path of directory to recursively parse through and build script dataframe from.
         shuffle          (bool): Whether to shuffle the dataframe before returning it. Not doing so can introduce bias,
                                  especially when considering local data whose purpose is likely correlated with directory
                                  distance.
        '''

        script_paths = self.get_all_script_paths(script_directory)
        script_df = []

        for i, script_path in enumerate(script_paths):
            code = open(script_path, 'r').read()
            extension = "." + script_path.split('.')[-1]
            language = self.EXTENTSION_TO_PROGRAMING_LANGUAGE[extension]
            #TODO: Add method(s) for finding docstring when available
            script_row = dict(code=code, path=script_path, url=script_directory, language=language, docstring=None)
            script_df.append(script_row)

        script_df = pd.DataFrame(script_df)
        script_df = self.join_df_with_subdirectory_dfs(script_directory, script_df)
        script_df = self.join_df_with_codesearch(script_directory, script_df)

        # Shuffle the dataframe so that chunks taken from it are unbiased.
        script_df = script_df.sample(frac=1) if shuffle else script_df
        script_df = self.drop_all_nans(script_df)
        return script_df

    def parse_codesearch_payload(self, codesearch_payload):
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
        codesearch_rows = [{column: payload[column] for column in self.codesearch_columns} for payload in payloads]
        code_search_df = pd.DataFrame(codesearch_rows)
        return code_search_df

    def delete_codesearch_payloads(self, directory):
        codesearch_paths = self.get_code_search_paths(directory, ".jsonl")
        print(f'UPDATE: Deleting CodeSearch paths = {codesearch_paths}.')

        for path in codesearch_paths:
            os.remove(path)

        subdirs = self.get_subdirs(directory)

        for subdir in subdirs:
            self.delete_codesearch_payloads(subdir)






