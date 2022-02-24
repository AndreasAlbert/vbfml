import pickle
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import boost_histogram as bh
import tensorflow as tf
import uproot
from tabulate import tabulate
from tqdm import tqdm

from vbfml.input.sequences import DatasetInfo, MultiDatasetSequence


def load(fpath: str) -> object:
    with open(fpath, "rb") as f:
        return pickle.load(f)


def save(data: object, fpath: str) -> None:
    with open(fpath, "wb") as f:
        return pickle.dump(data, f)


def get_n_events(filename: str, treename: str) -> int:
    """Return number of events in a TTree in a given file"""
    try:
        return int(uproot.open(filename)[treename].num_entries)
    except uproot.exceptions.KeyInFileError:
        return 0


def get_weight_integral_by_label(sequence: MultiDatasetSequence) -> Dict[str, float]:
    """
    Integrate the weights of all samples, accumulate by their label.
    """
    # Save settings so we can restore later
    backup = {}
    for key in ["batch_size", "batch_buffer_size"]:
        backup[key] = getattr(sequence, key)

    # For integration, batches can be large
    sequence.batch_size = int(1e6)
    sequence.batch_buffer_size = 10

    # Histogram to handle storage
    string_labels = sequence.dataset_labels()
    total_weight = bh.Histogram(
        bh.axis.IntCategory(list(range(len(string_labels))), growth=False),
        storage=bh.storage.Weight(),
    )

    # Batch loop
    for i in tqdm(range(len(sequence)), desc="Determining weight integrals"):
        _, labels, weight = sequence[i]
        total_weight.fill(labels.argmax(axis=1), weight=weight)

    # Integrals are separated by truth label
    integrals = {}
    for i, label in enumerate(string_labels):
        h = total_weight[{0: i}]
        integrals[label] = h.value

    # Restore original settings
    for key, value in backup.items():
        setattr(sequence, str(key), value)

    return integrals


def normalize_classes(
    sequence: MultiDatasetSequence, target_integral: float = 1
) -> None:
    """Changes data set weights in-place so that all classes have same integral."""
    label_to_weight_dict = get_weight_integral_by_label(sequence)
    for dataset_obj in sequence.datasets.values():
        weight = label_to_weight_dict[dataset_obj.label]
        dataset_obj.weight = target_integral * dataset_obj.weight / weight


def generate_callbacks_for_profiling() -> None:
    """
    Callbacks for profiling of keras training.

    See https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
    """

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs, histogram_freq=1, profile_batch="1,1000"
    )
    callbacks = [tboard_callback]

    return callbacks


def summarize_datasets(datasets: List[DatasetInfo]) -> None:
    """
    Prints a neat summary of a group of datasets to the terminal.
    """
    table = []
    total_by_label = defaultdict(int)
    for dataset in datasets:
        table.append((dataset.label, dataset.name, dataset.n_events))
        total_by_label[dataset.label] += dataset.n_events

    for key, value in total_by_label.items():
        table.append((key, "--- TOTAL ---", value))

    print(
        tabulate(
            sorted(table),
            headers=["Class label", "Physics data set name", "Number of events"],
        )
    )


def select_and_label_datasets(
    all_datasets: List[DatasetInfo], labels: Dict[str, str]
) -> List[DatasetInfo]:
    """
    Slim down a list of datasets and apply labeling.

    Labels is a dictionary that maps a new target label name to a regular
    expression matching data set names. Only data sets matching one of the
    regular expressions are returned. The labels of these data sets are
    set to the corresponding key from the labels dict.
    """
    selected_datasets = []
    for label, regex in labels.items():
        matching_datasets = [
            deepcopy(dataset)
            for dataset in all_datasets
            if re.match(regex, dataset.name)
        ]
        for dataset in matching_datasets:
            dataset.label = label
        selected_datasets.extend(matching_datasets)
    return selected_datasets


def append_history(
    history1: Dict[str, List[float]],
    history2: Dict[str, List[float]],
    validation_frequence: int = 1,
) -> Dict[str, List[float]]:
    """
    Append keras training histories.

    The second history is appended to the first one, and the combined history is returned.
    """
    new_history = {}
    for key, value_list in history2.items():
        n_entries = len(value_list)

        x_freq = 1
        x_offset = 0

        if "val" in key:
            x_freq = validation_frequence
            x_offset = 1

        if key in history1:
            original_x = history1[f"x_{key}"]
            new_x = original_x + [
                original_x[-1] + (ix + x_offset) * x_freq for ix in range(n_entries)
            ]
            new_y = history1[key] + value_list
        else:
            new_x = [(ix + x_offset) * x_freq for ix in range(n_entries)]
            new_y = value_list

        new_history[f"x_{key}"] = new_x
        new_history[f"y_{key}"] = new_y
    return new_history
