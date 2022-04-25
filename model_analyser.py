import os
from glob import glob

import re

import pandas as pd

class ModelAnalyser():
    def __init__(self, fit_log_base="fit_log.csv"):
        self.fit_log_base = fit_log_base

    def find_fit_log(self, model_directory):
        fit_log_path = os.path.join(model_directory, self.fit_log_base)

        if not os.path.exists(fit_log_path):
            print(f'WARNING: Fit log not found in {model_directory}!')
            return pd.DataFrame([]), None

        fit_log = pd.read_csv(fit_log_path)
        return fit_log, fit_log_path

    def get_fit_log_chunk(self, fit_log_path):
        chunk_str = re.findall(r"(?<=chunk_)\d+", fit_log_path)[0]
        chunk = int(chunk_str)
        return chunk

    def sort_by_chunk_epoch(self, df):
        df.sort_values(by=["chunk", "epoch"], inplace=True)
        columns = df.columns.tolist()
        columns.pop(-1)
        columns.insert(0, "chunk")
        df = df[columns]
        return df

    def join_fit_logs(self, model_dir):
        model_subdir_pattern = os.path.join(model_dir, "*")
        model_subdirs = glob(model_subdir_pattern)
        joined_fit_log = pd.DataFrame([])

        for model_subdir in model_subdirs:
            fit_log, fit_log_path = self.find_fit_log(model_subdir)

            # No fit log found, so continue through subdirs.
            if fit_log.empty:
                continue

            fit_log = self.increment_epochs(fit_log)
            chunk_index = self.get_fit_log_chunk(fit_log_path)
            fit_log["chunk"] = [chunk_index] * len(fit_log)
            joined_fit_log = pd.concat([joined_fit_log, fit_log])

        joined_fit_log = self.sort_by_chunk_epoch(joined_fit_log)
        return joined_fit_log

    def increment_epochs(self, fit_df):
        '''
        Each epoch appears as zero in original fit log because we reset negative samples for each cycle.
        This increments them to be unique and accurate.
        '''

        self.epoch = 0
        print(f"UPDATE: epoch before increment = {fit_df['epoch'].values}")
        fit_df["epoch"] = fit_df.apply(lambda row: row.name, axis=1)
        print(f"UPDATE: epoch after increment = {fit_df['epoch'].values}")
        return fit_df