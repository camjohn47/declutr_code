from experiment import Experiment
from common_funcs import get_code_df, process_fig
from retrieval_models import QueryEncoderRetriever

from pandas import DataFrame, concat

from plotly.express import bar
from plotly.graph_objects import Scatter, Figure

from numpy import squeeze

from os.path import join
from pathlib import Path
from re import sub

class CodeSearchExperiment(Experiment):
    VARIABLE_CHOICES = ["query_encoder_id", "script_encoder_id"]
    QUERY_COL = QueryEncoderRetriever.QUERY_COL
    ANSWER_COL = QueryEncoderRetriever.ANSWER_COL
    EXPERIMENTS_DIRECTORY = join(Experiment.EXPERIMENTS_DIRECTORY, "CodeSearch")

    def __init__(self, variable_arg, variable_val, variable_domain, constant_arg_vals, sampling=1, add_constants_to_id=None,
                 retrieved_docs=50):
        self.variable_arg = variable_arg

        if self.variable_arg not in self.VARIABLE_CHOICES:
            raise ValueError(f"ERROR: Requested variable arg {self.variable_arg} not in choices = {self.VARIABLE_CHOICES}")

        required_variable_arg = [x for x in self.VARIABLE_CHOICES if x != self.variable_arg][0]
        self.constant_arg_vals = constant_arg_vals
        self.constant_args = list(self.constant_arg_vals.keys())
        self.constant_arg = self.constant_args[0]

        # Make sure that specified constant is either query or answer encoder, depending on which is variable above.
        if self.constant_arg != required_variable_arg:
            raise ValueError(f"ERROR: Constant argument doesn't match required {required_variable_arg}")

        self.constant_arg_val = self.constant_arg_vals[self.constant_arg]
        self.variable_val = variable_val
        self.variable_domain = variable_domain
        self.sampling = sampling
        self.accuracy_fig_title = f"Top K Accuracy for {self.variable_arg}"
        self.add_constants_to_id = add_constants_to_id
        self.experiment_id = self.build_experiment_id()
        self.experiment_directory = join(self.EXPERIMENTS_DIRECTORY, self.experiment_id)
        self.results_dir = join(self.experiment_directory, "results")
        Path(self.results_dir).mkdir(exist_ok=True, parents=True)

        # Build paths in results_dir for saving specific visuals and CSV's.
        self.accuracy_fig_path = join(self.results_dir, "top_<K>_accuracy_fig.html")
        self.top_k_line_path = join(self.results_dir, "top_k_accuracy_line.html")
        self.results_csv_path = join(self.results_dir, "codesearch")
        self.retrieved_docs = retrieved_docs
        self.retrieved_doc_range = list(range(1, self.retrieved_docs + 1))
        self.build_config()
        self.save_config()

    def build_config(self):
        self.config = {"constant_arg": self.constant_arg, "variable_arg": self.variable_arg,
                       "variable_domain": self.variable_domain, "retrieved_docs": self.retrieved_docs}

    def save_config(self):
        self.config_path = join(self.experiment_directory, "config.txt")

        with open(self.config_path, "w") as file:
            file.write(str(self.config))

    def run(self):
        '''
        AFTER RUNNING ABOVE EXPERIMENT: Use different DeClutr models for code retrieval task.
        '''

        results_df = DataFrame()

        for variable_value in self.variable_domain:
            query_encoder_args = {self.variable_arg: variable_value, self.constant_arg: self.constant_arg_val,
                                  "retrieved_docs": self.retrieved_docs}
            query_encoder_retriever = QueryEncoderRetriever(**query_encoder_args)
            code_df = get_code_df(sampling=self.sampling)
            code_df = query_encoder_retriever.transform(code_df)
            code_df[self.variable_arg] = [variable_value] * len(code_df)
            results_df = concat([results_df, code_df])

        return results_df

    #TODO: Clean top K logic compatible with cached result length.
    def calc_top_k_accuracy(self, query_df, top_k):
        '''
        Outputs
        % of docstrings whose script was in top k retrieved scripts.
        '''

        answer_in_top_k = lambda row: row[self.QUERY_COL] in row[self.ANSWER_COL][:top_k]
        query_df["answer_in_top_k"] = query_df.apply(answer_in_top_k, axis=1)
        num_correct = query_df["answer_in_top_k"].sum()
        num_queries = len(query_df)
        top_k_accuracy = num_correct / num_queries
        return top_k_accuracy

    def build_accuracy_fig(self, analysis_df, k):
        acc_fig = bar(data_frame=analysis_df, x=self.variable_arg, y=f"top_{k}_accuracy")
        acc_fig.show()
        acc_fig.update_layout(title_text=self.accuracy_fig_title, title_x=0.5)
        acc_fig.update_xaxes(title_text=self.variable_arg)
        return acc_fig

    def build_accuracy_figs(self, analysis_df):
        for k in self.retrieved_doc_range:
            accuracy_fig = self.build_accuracy_fig(analysis_df, k=k)
            accuracy_fig_path = sub(r'<K>', str(k), self.accuracy_fig_path)
            process_fig(accuracy_fig, accuracy_fig_path)

    def build_top_k_accuracy_line(self, analysis_df):
        top_k_fig = Figure()
        top_k_cols = [f"top_{k}_accuracy" for k in self.retrieved_doc_range]
        colors = ["red", "blue"]
        var_ind = 0

        for var, var_results_df in analysis_df.groupby(self.variable_arg):
            var_accuracies = var_results_df[top_k_cols].values
            var_accuracies = squeeze(var_accuracies)
            color = colors[var_ind]
            line = Scatter(x=self.retrieved_doc_range, y=var_accuracies, name=var, marker=dict(color=color), mode="lines+markers")
            top_k_fig.add_trace(line)
            var_ind += 1

        process_fig(top_k_fig, self.top_k_line_path)

    def run_and_analyze(self):
        results_df = self.run()
        analysis_rows = []

        for variable_val, variable_df in results_df.groupby(self.variable_arg):
            k_accuracies = {}

            for k in self.retrieved_doc_range:
                top_k_accuracy = self.calc_top_k_accuracy(variable_df, top_k=k)
                k_accuracies[f"top_{k}_accuracy"] = top_k_accuracy

            df_row = {self.variable_arg: variable_val, **k_accuracies}
            analysis_rows.append(df_row)

        analysis_df = DataFrame(analysis_rows)
        self.build_top_k_accuracy_line(analysis_df)
        return analysis_df










#%%
