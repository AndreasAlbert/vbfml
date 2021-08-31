from dataclasses import dataclass
import pandas as pd


@dataclass
class MultiBatchBuffer:
    df: pd.DataFrame = None
    batch_size: int = 1
    min_batch: int = -1
    max_batch: int = -1

    def set_multibatch(self, df: pd.DataFrame, min_batch: int):
        self.df = df
        self.min_batch = min_batch
        self.max_batch = min_batch + len(df) // self.batch_size

    def __contains__(self, batch_index):
        if self.df is None:
            return False
        if not len(self.df):
            return False
        if batch_index < 0:
            return False
        return self.min_batch <= batch_index <= self.max_batch

    def clear(self):
        self.df = None
        self.min_batch = -1
        self.max_batch = -1

    def get_batch_df(self, batch_index):
        if not batch_index in self:
            raise IndexError(f"Batch index '{batch_index}' not in current buffer.")

        row_start = batch_index - self.min_batch
        row_stop = min(row_start + self.batch_size, len(self.df))
        return self.df.loc[row_start:row_stop]
