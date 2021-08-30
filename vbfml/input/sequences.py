from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence, to_categorical

from vbfml.input.uproot import UprootReaderMultiFile
from vbfml.util import LRIDictBuffer


@dataclass
class DatasetInfo:
    name: str
    files: list
    n_events: int
    treename: str
    label: str = ""
    weight: float = 1.0

    def __post_init__(self):
        if not self.label:
            self.label = self.name


def row_vector(branch):
    return np.array(branch).reshape((len(branch), 1))


class MultiDatasetSequence(Sequence):
    def __init__(
        self,
        batch_size: int,
        branches: "list[str]",
        shuffle=True,
        batch_buffer_size=1,
        read_range=(0.0, 1.0),
        weight_expression=None,
    ) -> None:
        self.datasets = {}
        self.readers = {}
        self.branches = branches
        self.batch_size = batch_size
        self._shuffle = shuffle
        self.encoder = LabelEncoder()

        self.batch_buffer_size = batch_buffer_size
        self.batch_buffer = LRIDictBuffer(buffer_size=self.batch_buffer_size)

        self._read_range = read_range
        self._weight_expression = weight_expression

    def __len__(self) -> int:
        read_fraction = self.read_range[1] - self.read_range[0]
        return int(self.total_events() * read_fraction // self.batch_size)

    @property
    def weight_expression(self) -> str:
        return self._weight_expression

    @weight_expression.setter
    def weight_expression(self, weight_expression: str) -> None:
        self._weight_expression = weight_expression

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool) -> None:
        self._shuffle = shuffle

    @property
    def read_range(self) -> tuple:
        return self._read_range

    @read_range.setter
    def read_range(self, read_range: tuple) -> None:
        assert len(read_range) == 2, "Read range must be a tuple of length two."
        assert all(
            0 <= x <= 1 for x in read_range
        ), "Read range bounds must be between 0 and 1."
        assert (
            read_range[0] < read_range[1]
        ), "Read range bounds must be in increasing order."

        # Buffered batches must be thrown out
        # if read range changes
        if read_range != self.read_range:
            self.batch_buffer.clear()

        self._read_range = read_range

    def _format_weights(self, df: pd.DataFrame, dataset_name: str) -> None:
        dataset_weight = self.datasets[dataset_name].weight
        if self.weight_expression:
            df.rename(columns={self.weight_expression: "weight"}, inplace=True)
            df["weight"] = df["weight"] * dataset_weight
        else:
            df["weight"] = dataset_weight

    def _read_dataframes_for_batch_range(
        self, batch_start: int, batch_stop: int
    ) -> list:
        """Reads and returns data for a given batch and all datasets"""
        dataframes = []
        for name in self.datasets.keys():
            df = self._read_single_dataframe_for_batch_range(
                batch_start, batch_stop, name
            )
            if self.is_weighted():
                self._format_weights(df, name)
            dataframes.append(df)
        return dataframes

    def _read_single_dataframe_for_batch_range(
        self, batch_start: int, batch_stop: int, dataset_name: str
    ):
        """Reads and returns data for a given batch and single data"""
        start, stop = self._get_start_stop_for_single_read(
            batch_start, batch_stop, dataset_name
        )
        if not dataset_name in self.readers:
            self._init_reader(dataset_name)
        df = self.readers[dataset_name].read_events(start, stop)
        df["label"] = self.encode_label(self.datasets[dataset_name].label)
        return df

    def _get_start_stop_for_single_read(
        self, batch_start: int, batch_stop: int, dataset_name: str
    ) -> tuple:
        """Returns the start and stop coordinates for reading a given batch of data from one dataset"""

        dataset_events = self.datasets[dataset_name].n_events

        offset = np.ceil(dataset_events * self.read_range[0])
        start = offset + np.floor(
            batch_start * self.batch_size * self.fractions[dataset_name]
        )
        stop = np.floor(batch_stop * self.batch_size * self.fractions[dataset_name]) - 1
        return start, stop

    def dataset_labels(self) -> "list[str]":
        """Return list of labels of all data sets in this Sequence."""
        return [dataset.label for dataset in self.datasets.values()]

    def dataset_names(self) -> "list[str]":
        """Return list of names of all data sets in this Sequence."""
        return [dataset.name for dataset in self.datasets.values()]

    def total_events(self) -> int:
        """Total number of events of all data sets"""
        return sum(dataset.n_events for dataset in self.datasets.values())

    def encode_label(self, label: str) -> np.ndarray:
        """
        Encode string-values labels into one-hot vectors.
        """
        return self.label_encoding[label]

    def _split_multibatch(self, dfs: "list[pd.DataFrame]", index_offset=0) -> dict:
        """
        Convert a dataframe containing multiple batches into per-batch frames
        """
        split_dfs = defaultdict(list)
        for df in dfs:
            for ibatch, df_batch in enumerate(
                np.array_split(df, self.batch_buffer_size)
            ):
                split_dfs[index_offset + ibatch].append(df_batch)
        return dict(split_dfs)

    def _fill_batch_buffer(self, batch_start: int, batch_stop: int) -> None:
        """
        Read batches from file and save them into the buffer for future use.
        """
        multibatch_dfs = self._read_dataframes_for_batch_range(batch_start, batch_stop)
        batched_dfs = self._split_multibatch(multibatch_dfs, index_offset=batch_start)

        for ibatch, dfs in batched_dfs.items():
            if ibatch in self.batch_buffer:
                continue
            df = pd.concat(dfs)

            if self.shuffle:
                df = df.sample(frac=1)

            non_feature_columns = ["label"]
            if self.is_weighted:
                non_feature_columns.append("weight")
            features = df.drop(columns=non_feature_columns).to_numpy()

            labels = to_categorical(
                row_vector(df["label"]),
                num_classes=len(self.dataset_labels()),
            )

            if self.is_weighted:
                weights = row_vector(df["weight"])
                batch = (features, labels, weights)
            else:
                batch = (features, labels)

            self.batch_buffer[ibatch] = batch

    def __getitem__(self, batch: int) -> tuple:
        """Returns a single batch of data"""
        if batch not in self.batch_buffer:
            max_batch_to_buffer = min(batch + self.batch_buffer_size, len(self))
            self._fill_batch_buffer(batch, max_batch_to_buffer)
        return self.batch_buffer[batch]

    def add_dataset(self, dataset: DatasetInfo) -> None:
        """Add a new data set to the Sequence."""
        if dataset.name in self.datasets:
            raise IndexError(f"Dataset already exists: '{dataset.name}'.")
        self.datasets[dataset.name] = dataset
        self._datasets_changed()

    def remove_dataset(self, dataset_name: str) -> DatasetInfo:
        """Remove dataset from this Sequence and return its DatasetInfo object"""
        info = self.datasets.pop(dataset_name)
        self._datasets_changed()
        return info

    def get_dataset(self, dataset_name: str) -> DatasetInfo:
        return self.datasets[dataset_name]

    def _init_reader(self, dataset_name) -> None:
        """
        Initializes file readers for a given data set.

        Note that this operation may be slow as the reader
        will open all files associated to it to determine
        event counts.
        """
        info = self.datasets[dataset_name]

        branches_to_read = self.branches.copy()
        if self.weight_expression:
            branches_to_read.append(self.weight_expression)

        reader = UprootReaderMultiFile(
            files=info.files,
            branches=branches_to_read,
            treename=info.treename,
        )
        self.readers[dataset_name] = reader

    def _init_readers(self) -> None:
        """Initializes file readers for all data sets"""
        for dataset_name in self.datasets.keys():
            self._init_reader(dataset_name)

    def _init_label_encoding(self) -> None:
        """Create encoding of string labels <-> integer class indices"""
        labels = sorted([dataset.label for dataset in self.datasets.values()])
        label_encoding = dict(enumerate(labels))
        label_encoding.update({v: k for k, v in label_encoding.items()})
        self.label_encoding = label_encoding

    def _datasets_changed(self):
        """Perform all updates needed after a change in data sets"""
        self._init_dataset_fractions()
        self._init_label_encoding()

    def _init_dataset_fractions(self) -> None:
        """Determine what fraction of the total events is from a given data set"""
        total = self.total_events()
        self.fractions = {
            name: info.n_events / total for name, info in self.datasets.items()
        }

    def is_weighted(self):
        """
        Weights are needed if a per-event expression is used OR any data set has a weight.
        """
        if self.weight_expression:
            return True
        if any([dataset.weight != 1 for dataset in self.datasets.values()]):
            return True
        return False
