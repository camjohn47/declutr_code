import os

from common_funcs import get_code_df, make_path_directories
from retrieval_models import QueryEncoderRetriever
from custom_arg_parser import CustomArgParser

# ARGUMENTS
sampling_args = ["-s", "--sampling", dict(required=False, default=1, help="Fraction of scripts to use.")]
query_model_args = ["-qmi", "--query_model_id", dict(required=True, help="ID of the requested query encoder model.")]
script_model_args = ["-smi", "--script_model_id", dict(required=True, help="ID of the requested script encoder model.")]
experiment_args = ["-ei", "--experiment_id", dict(required=False, default=None, help="Experiment ID if this is run as part of an experiment.")]
arg_parser = CustomArgParser()
arg_parser.add_arguments([sampling_args, query_model_args, script_model_args, experiment_args])

#TODO: Make notebook from this when finalized.

# Initialize natural language query encoder and programming language script encoder.
argument_subsets = [["sampling"], ["query_model_id"], ["script_model_id"], ["experiment_id"]]
sampling, query_model_id, script_model_id, experiment_id = arg_parser.get_argument_subsets(argument_subsets)
print(f"UPDATE: query model id = {query_model_id}")
retriever = QueryEncoderRetriever(query_model_id=query_model_id, script_model_id=script_model_id)

# Get code dataframe for taking queries and finding code for them.
code_df = get_code_df(sampling=sampling)
code_df.info()
search_results = retriever.transform(code_df)
search_results.info()

def get_results_path():
    if experiment_id:
        results_path = os.path.join("experiments", experiment_id, "codesearch_results.csv")
    else:
        results_path = os.path.join("results", f"codesearch_query={query_model_id}_script={script_model_id}.csv")
    return results_path

#TODO: Organize results better. If models are from same experiment, should go in experiment directory.
results_path = get_results_path()
make_path_directories(results_path)
print(f"UPDATE: Saving CodeSearch query results to {results_path}.")
search_results.to_csv(results_path)





