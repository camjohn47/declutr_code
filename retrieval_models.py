from encoder_deployer import EncoderDeployer

from scipy.spatial import distance_matrix
import numpy as np

from common_funcs import reduce_matrix

import pandas as pd

class QueryEncoderRetriever():
    '''
    A class for retrieving documents for queries using encoders. Both the query set and document set are encoded using
    potentially different encoders.
    '''

    def __init__(self, query_model_id, script_model_id, retrieval_count=10):
        self.query_model_id = query_model_id
        self.query_deployer = EncoderDeployer(self.query_model_id, text_column="docstring", encoder_model="transformer")
        self.script_model_id = script_model_id
        self.script_deployer = EncoderDeployer(self.script_model_id, text_column="code", encoder_model="transformer")

        # Number of retrieved scripts for each query.
        self.retrieval_count = retrieval_count

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

    def minimize_query_distance(self, query_encoding, scripts, script_encodings):
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

        scripts = code_df["code"].values
        queries = code_df["docstring"].values

        # Use encoder deployers to encode NL queries and PL scripts.
        query_encodings = self.query_deployer.make_feature_matrix(code_df)
        script_encodings = self.script_deployer.make_feature_matrix(code_df)
        query_encodings, script_encodings = self.make_encodings_compatibile(query_encodings, script_encodings)
        results_df = []

        for i, query in enumerate(queries):
            if i >= len(query_encodings):
                break

            query_encoding = query_encodings[i, :]
            optimal_index, optimal_distance = self.minimize_query_distance(query_encoding, scripts, script_encodings)
            optimal_script = scripts[optimal_index]
            results_row = dict(query=query, matched_script_index=optimal_index, matched_script=optimal_script, matched_distance=optimal_distance)
            results_df.append(results_row)

        results_df = pd.DataFrame(results_df)
        return results_df