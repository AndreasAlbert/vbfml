from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence, to_categorical

from vbfml.input.uproot import UprootReaderMultiFile
from vbfml.util import MultiBatchBuffer


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


class Normalizer:
    def __init__(self) -> None:
        pass

    def fit(self, features: np.ndarray):
        return self

    def transform(self, features: np.ndarray, ceiling: float = 255.0):
        """Simple normalizer for image data. Divides all values with 255
        to get all values within [0,1] range.

        Args:
            features (np.ndarray): Image pixels.
            ceiling (float, optional): The highest pixel value. Defaults to 255.
        """
        return features / ceiling


class MultiDatasetSequence(Sequence):
    def __init__(
        self,
        batch_size: int,
        branches: "list[str]",
        shuffle=True,
        batch_buffer_size=1,
        read_range=(0.0, 1.0),
        weight_expression=None,
        scale_features="none",
    ) -> None:
        self.datasets = {}
        self.readers = {}
        self.branches = branches
        self._batch_size = batch_size
        self._shuffle = shuffle

        self.batch_buffer_size = batch_buffer_size
        self.buffer = MultiBatchBuffer(batch_size=self.batch_size)
        self._read_range = read_range
        self._weight_expression = weight_expression

        self._scale_features = scale_features
        self._feature_scaler = None
        self._float_dtype = np.float32

    def __len__(self) -> int:
        read_fraction = self.read_range[1] - self.read_range[0]
        return int(np.ceil(self.total_events() * read_fraction / self.batch_size))

    @property
    def weight_expression(self) -> str:
        return self._weight_expression

    @weight_expression.setter
    def weight_expression(self, weight_expression: str) -> None:
        self._weight_expression = weight_expression

    @property
    def batch_size(self) -> str:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self.buffer.clear()
        self.buffer.batch_size = batch_size
        self._batch_size = batch_size

    @property
    def scale_features(self) -> str:
        return self._scale_features

    @scale_features.setter
    def scale_features(self, scale_features: str) -> None:
        if scale_features != self.scale_features:
            self.buffer.clear()
        self._scale_features = scale_features

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
            self.buffer.clear()

        self._read_range = read_range

    def _init_feature_scaler_from_features(self, features: np.ndarray) -> None:
        """
        Initialize the feature scaler object based on a feature tensor.
        """
        scalers = {"standard": StandardScaler, "norm": Normalizer}
        self._feature_scaler = scalers[self._scale_features]().fit(features)

    def _init_feature_scaler_from_multibatch(self, df: "pd.DataFrame") -> None:
        """
        Initialize the feature scaler object based on a list of DataFrames

        Used to initialize the scaler based on data corresponding to a multi-batch
        (i.e. many batches) rather than just a single batch. Higher
        statistiscal accuracy will result in better scaling performance.
        """
        features = df.drop(columns=self._non_feature_columns()).to_numpy()
        self._init_feature_scaler_from_features(features)

    def apply_feature_scaling(self, features: np.ndarray) -> np.ndarray:
        assert self._feature_scaler, "Feature scaler has not been initalized."
        return self._feature_scaler.transform(features)

    def _create_weight_column(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Creates a column called 'weight' in the data frame

        The weight column includes the contributions from both the
        per-event weight (if weight_expression is specified),
        and the per-dataset normalization weight.
        """
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
            # Do not add empty dataframes to the dataframe list
            if len(df) == 0:
                continue
            if self.is_weighted():
                self._create_weight_column(df, name)
            dataframes.append(df)
        return dataframes

    def _read_multibatch(self, batch_start: int, batch_stop: int) -> pd.DataFrame:
        dfs = self._read_dataframes_for_batch_range(batch_start, batch_stop)
        return pd.concat(dfs, ignore_index=True)

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

        n_batches = batch_stop - batch_start
        stop = np.floor(
            start + n_batches * self.batch_size * self.fractions[dataset_name]
        )
        stop = min(stop, int(self.read_range[1] * dataset_events))

        # If start < stop, that typically means we don't have events to read
        # In that case, set start = stop so that we'll generate an empty df
        stop = max(start, stop)

        return start, stop

    def dataset_labels(self) -> "list[str]":
        """Return list of labels of all data sets in this Sequence."""
        return sorted(list(set(dataset.label for dataset in self.datasets.values())))

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

    def _non_feature_columns(self) -> "list[str]":
        columns = ["label"]
        if self.is_weighted():
            columns.append("weight")
        return columns

    def _fill_batch_buffer(self, batch_start: int, batch_stop: int) -> None:
        """
        Read batches from file and save them into the buffer for future use.
        """
        multibatch_df = self._read_multibatch(batch_start, batch_stop)
        if self.scale_features != "none" and not self._feature_scaler:
            self._init_feature_scaler_from_multibatch(multibatch_df)
        if self.shuffle:
            multibatch_df = multibatch_df.sample(frac=1, ignore_index=True)
        self.buffer.set_multibatch(multibatch_df, batch_start)

    def _batch_df_formatting(self, df: pd.DataFrame) -> "tuple[np.ndarray]":
        """Convert from a batch from pd.DataFrame to a tuple of np.ndarray for keras"""

        features = df.drop(columns=self._non_feature_columns()).to_numpy()
        features = features.astype(self._float_dtype)
        if self.scale_features != "none":
            features = self.apply_feature_scaling(features)

        # Double checking the feature range here
        if self.scale_features == "norm":
            valid = np.all((features >= 0) & (features <= 1))
            assert (
                valid
            ), "Features are not scaled correctly to [0,1] range, please check!"

        labels = to_categorical(
            row_vector(df["label"]),
            num_classes=len(self.dataset_labels()),
        )

        if self.is_weighted():
            weights = np.abs(row_vector(df["weight"]).astype(np.float16))
            weights = weights.astype(self._float_dtype)
            batch = (features, labels, weights)
        else:
            batch = (features, labels)

        return batch

    def __getitem__(self, batch: int) -> tuple:
        """Returns a single batch of data"""
        if batch not in self.buffer:
            max_batch_to_buffer = min(batch + self.batch_buffer_size, len(self))
            self._fill_batch_buffer(batch, max_batch_to_buffer)

        batch_df = self.buffer.get_batch_df(batch)
        batch = self._batch_df_formatting(batch_df)
        return batch

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

    def _init_label_encoding(self) -> None:
        """Create encoding of string labels <-> integer class indices"""
        labels = self.dataset_labels()
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
