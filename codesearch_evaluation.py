from encoder_deployer import EncoderDeployer
from code_parser import CodeParser

from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import os
import sys

from ast import literal_eval

from common_funcs import get_code_df

def calc_min_document_index(query_encoding, document_encodings, metric="cosine"):
    '''
    Return index of document with min distance to the query encoding.
    '''

    #print(f"UPDATE: Document encodings = {document_encodings}")
    query_encoding = np.expand_dims(query_encoding, axis=0)
    query_distances = distance_matrix(document_encodings, query_encoding)
    #print(f"UPDATE: Query distance shape ={query_distances.shape}")
    min_document_index = np.argmin(query_distances, axis=0)
    min_distance = query_distances[min_document_index]
    return min_document_index, min_distance

def find_code_for_query(query_encoding, scripts, script_encodings):
    min_script_index, min_distance = calc_min_document_index(query_encoding, script_encodings)
    #print(f"UPDATE: Min index={min_script_index}, min distance = {min_distance}")
    min_script = scripts[min_script_index]
    return min_script

def find_code_for_queries(queries, scripts, code_df, query_deployer, script_deployer):
    query_encodings = query_deployer.make_feature_matrix(code_df)
    script_encodings = script_deployer.make_feature_matrix(code_df)

    for i, query in enumerate(queries):
        query_encoding = query_encodings[i, :]
        optimal_script = find_code_for_query(query_encoding, scripts, script_encodings)
        #print(f"UPDATE: Query ={query}, retrieved script = {optimal_script}. \n")

def prepare_query_search(code_df, query_deployer, script_deployer):
    scripts = code_df["code"].values
    queries = code_df["docstring"].values
    find_code_for_queries(queries, scripts, code_df, query_deployer, script_deployer)

#TODO: Make notebook from this when finalized.
_, sampling = sys.argv
sampling = literal_eval(sampling)

# Initialize natural language query encoder and programming language script encoder.
query_model_id = "docstring_MAL=64"
query_deployer = EncoderDeployer(query_model_id, text_column="docstring", encoder_model="transformer")
script_model_id = query_model_id
script_deployer = EncoderDeployer(script_model_id, text_column="code", encoder_model="transformer")

# Get code dataframe for taking queries and finding code for them.
code_df = get_code_df(sampling=sampling)
code_df.info()

# Run query search on code using the deployed encoders.
prepare_query_search(code_df, query_deployer, script_deployer)


