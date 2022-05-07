import sys
from ast import literal_eval

from common_funcs import get_code_df
from retrieval_models import QueryEncoderRetriever

#TODO: Make notebook from this when finalized.
_, sampling = sys.argv
sampling = literal_eval(sampling)

# TODO: Add text_column argument to sequence processor call for flexibility.
# Initialize natural language query encoder and programming language script encoder.
query_model_id = "transformer_ed=25"
script_model_id = "transformer_ed=25"
retriever = QueryEncoderRetriever(query_model_id=query_model_id, script_model_id=script_model_id)

# Get code dataframe for taking queries and finding code for them.
code_df = get_code_df(sampling=sampling)
code_df.info()
search_results = retriever.transform(code_df)
search_results.info()

#TODO: Organize results better. If models are from same experiment, should go in experiment directory.
search_results.to_csv("codesearch_results.csv")





