import glob
import os
import re

from vbfml.input.sequences import DatasetInfo, MultiDatasetSequence
from vbfml.training.util import get_n_events


def build_sequence(
    datasets: "list[DatasetInfo]",
    features: "list[str]",
    weight_expression: str = "weight_total*xs/sumw",
    absolute_weight: bool = False,
    shuffle: bool = True,
    scale_features: str = "standard",
) -> MultiDatasetSequence:
    """Shortcut to set up a MultiDatasetSequence"""

    if absolute_weight:
        weight_expression = f"abs({weight_expression})"

    sequence = MultiDatasetSequence(
        batch_size=50,
        branches=features,
        shuffle=shuffle,
        batch_buffer_size=int(1e5),
        weight_expression=weight_expression,
        scale_features=scale_features,
    )

    for dataset in datasets:
        sequence.add_dataset(dataset)

    return sequence


def dataset_from_file_name_bucoffea(filepath: str) -> str:
    """
    Decodes data set name from file name, assuming bucoffea origin.
    """
    base = os.path.basename(filepath)
    m = re.match("tree_(.*_\d{4}).root", base)
    if not m:
        raise ValueError(f"Could not parse input file name: {filepath}")
    dataset_name = m.groups()[0]
    return dataset_name


def load_datasets_bucoffea(directory: str) -> "list[DatasetInfo]":
    """
    Load data sets based on a directory of ROOT files from bucoffea.

    Data set names are decoded from the file names, assuming one file per data set.
    By default, the labels and names of the data sets are set equal.
    """
    files = glob.glob(f"{directory}/*root")
    datasets = []
    for file in files:
        dataset_name = dataset_from_file_name_bucoffea(file)
        if "2018" in dataset_name:
            continue

        n_events = get_n_events(file, "sr_vbf")
        if not n_events:
            continue

        dataset = DatasetInfo(
            name=dataset_name,
            label=dataset_name,
            files=[file],
            treename="sr_vbf",
            n_events=int(n_events),
            weight=1,
        )
        datasets.append(dataset)
    return datasets
