import os

from modules.common_funcs import get_code_df, process_df
from classes.retrieval_models import QueryEncoderRetriever
from classes.custom_arg_parser import CustomArgParser

# ARGUMENTS
sampling_args = ["-s", "--sampling", dict(required=False, default=1, help="Fraction of scripts to use.")]
query_model_args = ["-qmi", "--query_encoder_id", dict(required=True, help="ID of the requested query encoder model.")]
script_model_args = ["-smi", "--script_encoder_id", dict(required=True, help="ID of the requested script encoder model.")]
experiment_args = ["-ei", "--experiment_id", dict(required=False, default=None, help="Experiment ID if this is run as part of an experiment.")]
arg_parser = CustomArgParser()
arg_parser.add_arguments([sampling_args, query_model_args, script_model_args, experiment_args])

#TODO: Make notebook from this when finalized.

# Initialize natural language query encoder and programming language script encoder.
argument_subsets = [["sampling"], ["query_encoder_id"], ["script_encoder_id"], ["experiment_id"]]
sampling, query_encoder_id, script_encoder_id, experiment_id = arg_parser.get_argument_subsets(argument_subsets)
print(f"UPDATE: query model id = {query_encoder_id}")
retriever = QueryEncoderRetriever(query_encoder_id=query_encoder_id, script_encoder_id=script_encoder_id)

# Get code dataframe for taking queries and finding code for them.
code_df = get_code_df(sampling=sampling)
search_results = retriever.transform(code_df)

def get_results_path():
    if experiment_id:
        results_path = os.path.join("../experiments", experiment_id, "codesearch_results.csv")
    else:
        results_path = os.path.join("results", f"codesearch_query={query_encoder_id}_script={script_encoder_id}.csv")
    return results_path

#TODO: Organize results better. If models are from same experiment, should go in experiment directory.
results_path = get_results_path()
process_df(df=search_results, df_name="CodeSearch results", csv_path=results_path)





