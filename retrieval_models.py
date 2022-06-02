from encoder_deployer import EncoderDeployer

from scipy.spatial import distance_matrix
import numpy as np

from tensorflow.keras.utils import Progbar

from common_funcs import reduce_matrix

from pandas import DataFrame, read_csv, concat
import re

from os.path import join, exists
from pathlib import Path
from ast import literal_eval

class QueryEncoderRetriever():
    '''
    A class for retrieving documents for queries using encoders. Both the query set and document set are encoded using
    potentially different encoders.
    '''
    
    QUERY_COL = "query_index"
    ANSWER_COL = "retrieved_script_index"
    RETRIEVAL_DIR = "retrieval_results"
    CACHED_QUERIES_NAME = "codesearch_results.csv"

    def __init__(self, query_encoder_id, script_encoder_id, retrieval_count=10):
        self.query_encoder_id = query_encoder_id
        self.query_deployer = EncoderDeployer(self.query_encoder_id, text_column="docstring", encoder_model="transformer")
        self.script_encoder_id = script_encoder_id
        self.script_deployer = EncoderDeployer(self.script_encoder_id, text_column="code", encoder_model="transformer")

        # Number of retrieved scripts for each query.
        self.retrieval_count = retrieval_count
        self.build_dir()
        self.cached_queries_path = join(self.dir, self.CACHED_QUERIES_NAME)
        self.load_cached_queries()
    
    def build_dir(self):
        self.dir = join(self.RETRIEVAL_DIR, f"query_id={self.query_encoder_id}_script_id={self.script_encoder_id}")
        Path(self.dir).mkdir(exist_ok=True, parents=True)

    @staticmethod
    def convert_indices_column(indices):
        indices_str = re.sub(r"(?<=[\d])\s+(?=[\d]+)", r", ", indices)
        indices = literal_eval(indices_str)
        return indices
    
    def load_cached_queries(self):
        self.cached_queries_df = DataFrame()
        self.cached_queries = set()
        
        if exists(self.cached_queries_path):
            print(f"UPDATE: Retrieval loading cached results in {self.cached_queries_path}.")
            converters = {self.ANSWER_COL: self.convert_indices_column}
            self.cached_queries_df = read_csv(self.cached_queries_path, converters=converters)
            self.cached_queries = set(self.cached_queries_df["query"].values.tolist())

    def update_cached_queries(self, results_df):
        new_queries = results_df[~results_df["query"].isin(self.cached_queries)]
        self.cached_queries_df = concat([self.cached_queries_df, new_queries])
        print(f"UPDATE: Saving updated cached queries tp {self.cached_queries_path}.")
        self.cached_queries_df.to_csv(self.cached_queries_path)

    def calc_min_document_index(self, query_encoding, document_encodings):
        '''
        Return indices of the <self.retrieval_count> closest documents to the query encoding.
        '''

        query_encoding = np.expand_dims(query_encoding, axis=0)
        query_distances = distance_matrix(document_encodings, query_encoding)
        min_document_index = np.argsort(query_distances, axis=0)
        min_document_index = np.squeeze(min_document_index)[:self.retrieval_count]
        min_distance = query_distances[min_document_index]
        return min_document_index, min_distance

    def minimize_query_distance(self, query, query_encoding, scripts, script_encodings):
        if query in self.cached_queries:
            cached_query = self.cached_queries_df[self.cached_queries_df["query"] == query]
            min_script_index = cached_query[self.ANSWER_COL].values[0]
            min_distance = cached_query["retrieved_distance"].values[0]
        else:
            min_script_index, min_distance = self.calc_min_document_index(query_encoding, script_encodings)

        return min_script_index, min_distance

    @staticmethod
    def make_encodings_compatibile(query_encodings, script_encodings):
        ''''''
        query_dims = query_encodings.shape[1]
        script_dims = script_encodings.shape[1]

        if query_dims > script_dims:
            print(f"WARNING: (Query dimensions = {query_dims}) > (script dims = {script_dims})! "
                  f"Reducing query dimensions: {query_dims} --> {script_dims}.")
            target_dims = script_dims
            query_encodings = reduce_matrix(query_encodings, target_dims)
        elif script_dims < query_dims:
            print(f"WARNING: (Script dimensions = {script_dims}) > (query dims = {query_dims})! "
                  f"Reducing script dimensions: {script_dims} --> {query_dims}.")
            target_dims = query_dims
            script_encodings = reduce_matrix(script_encodings, target_dims)

        return query_encodings, script_encodings

    def transform(self, code_df):
        '''
        Inputs
        code_df (DataFrame): Code-based dataframe with code and docstring text columns.

        Outputs
        results_df (DataFrame): A results-based dataframe, where each row has closest script for the query/docstring and
                                other attributes.
        '''

        code_df.dropna(subset=["code"], inplace=True)
        code_df.dropna(subset=["docstring"], inplace=True)

        scripts = code_df["code"].values
        queries = code_df["docstring"].values

        # Use encoder deployers to encode NL queries and PL scripts.
        query_encodings = self.query_deployer.make_feature_matrix(code_df)
        script_encodings = self.script_deployer.make_feature_matrix(code_df)
        query_encodings, script_encodings = self.make_encodings_compatibile(query_encodings, script_encodings)
        results_df = []
        query_count = len(queries)
        print(f"PROGRESS: Retrieving scripts for {query_count} document strings.")
        prog_bar = Progbar(target=query_count, unit_name="retrieved_document")

        for i, query in enumerate(queries):
            query_encoding = query_encodings[i, :]
            optimal_index, optimal_distance = self.minimize_query_distance(query, query_encoding, scripts, script_encodings)
            optimal_script = scripts[optimal_index]
            results_row = {self.QUERY_COL: i, "query": query, self.ANSWER_COL: optimal_index,
                               "retrieved_script": optimal_script, "retrieved_distance": optimal_distance}
            results_df.append(results_row)
            prog_bar.update(i + 1)

        results_df = DataFrame(results_df)
        self.update_cached_queries(results_df)
        return results_df

    def build_results(self, code_df):
        '''
        Retrieve scripts for queries and update the cache with new results.
        '''

        results_df = self.transform(code_df)


#%%
