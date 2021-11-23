#!/usr/bin/env python3
import copy
import os
import re
import warnings
import pandas as pd
from datetime import datetime

import click
import tensorflow as tf
from keras import backend as K
from tabulate import tabulate
from tqdm import tqdm

from vbfml.models import sequential_dense_model, sequential_convolutional_model
from vbfml.training.data import TrainingLoader
from vbfml.training.input import build_sequence, load_datasets_bucoffea
from vbfml.training.util import (
    append_history,
    normalize_classes,
    save,
    select_and_label_datasets,
)
from vbfml.util import ModelConfiguration, ModelFactory, vbfml_path

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def get_training_directory(tag: str) -> str:
    return os.path.join("./output", f"model_{tag}")


@click.group()
@click.option(
    "--tag",
    default=datetime.now().strftime("%Y-%m-%d_%H-%M"),
    required=False,
    help="A string-valued tag used to identify the run. If a run with this tag exists, will use existing run.",
)
@click.pass_context
def cli(ctx, tag):
    ctx.ensure_object(dict)
    ctx.obj["TAG"] = tag


@cli.command()
@click.pass_context
@click.option(
    "--learning-rate", default=1e-3, required=False, help="Learning rate for training."
)
@click.option("--dropout", default=0.5, required=False, help="Dropout rate.")
@click.option(
    "--input-dir",
    default=vbfml_path("root/2021-11-13_vbfhinv_treesForML"),
    required=False,
    help="Input directory containing the ROOT files for training and validation.",
)
@click.option(
    "--model-config",
    required=True,
    help="Path to the .yml file that has the model configuration parameters.",
)
def setup(ctx, learning_rate: float, dropout: float, input_dir: str, model_config: str):
    """
    Creates a new working area. Prerequisite for later training.
    """

    all_datasets = load_datasets_bucoffea(input_dir)

    dataset_labels = {
        "ewk_17": "(EWK.*2017|VBF_HToInvisible_M125_withDipoleRecoil_pow_pythia8_2017)",
        "v_qcd_nlo_17": "(WJetsToLNu_Pt-\d+To.*|Z\dJetsToNuNu_M-50_LHEFilterPtZ-\d+To\d+)_MatchEWPDG20-amcatnloFXFX_2017",
    }
    datasets = select_and_label_datasets(all_datasets, dataset_labels)
    for dataset_info in datasets:
        if re.match(dataset_labels["v_qcd_nlo_17"], dataset_info.name):
            dataset_info.n_events = 0.01 * dataset_info.n_events

    # Object containing data for different models
    # (set of features, dropout rate etc.)
    # Loaded from the YML configuration file
    mconfig = ModelConfiguration(model_config)

    features = mconfig.get("features")

    training_params = mconfig.get("training_parameters")
    validation_params = mconfig.get("validation_parameters")

    training_sequence = build_sequence(
        datasets=copy.deepcopy(datasets),
        features=features,
        weight_expression=mconfig.get("weight_expression"),
        shuffle=training_params["shuffle"],
        scale_features=training_params["scale_features"],
    )
    validation_sequence = build_sequence(
        datasets=copy.deepcopy(datasets),
        features=features,
        weight_expression=mconfig.get("weight_expression"),
        shuffle=validation_params["shuffle"],
        scale_features=validation_params["scale_features"],
    )
    normalize_classes(training_sequence)
    normalize_classes(validation_sequence)

    # Training sequence
    train_size = training_params["train_size"]

    training_sequence.read_range = (0.0, train_size)
    training_sequence.batch_size = training_params["batch_size"]
    training_sequence.batch_buffer_size = training_params["batch_buffer_size"]
    training_sequence[0]

    # Validation sequence
    validation_sequence.read_range = (train_size, 1.0)
    validation_sequence._feature_scaler = copy.deepcopy(
        training_sequence._feature_scaler
    )
    validation_sequence.batch_size = validation_params["batch_size"]
    validation_sequence.batch_buffer_size = validation_params["batch_buffer_size"]

    # Build model
    model = ModelFactory.build(mconfig)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        weighted_metrics=["categorical_accuracy"],
    )
    model.summary()

    training_directory = get_training_directory(ctx.obj["TAG"])

    # The trained model
    model.save(
        os.path.join(training_directory, "models/initial"), include_optimizer=True
    )
    model.save(
        os.path.join(training_directory, "models/latest"), include_optimizer=True
    )

    def prepend_path(fname):
        return os.path.join(training_directory, fname)

    # Feature scaling object for future evaluation
    save(training_sequence._feature_scaler, prepend_path("feature_scaler.pkl"))

    # List of features
    save(
        features,
        prepend_path(
            "features.pkl",
        ),
    )

    # Training and validation sequences
    # Clear buffer before saving to save space
    for seq in training_sequence, validation_sequence:
        seq.buffer.clear()
    save(training_sequence, prepend_path("training_sequence.pkl"))
    save(validation_sequence, prepend_path("validation_sequence.pkl"))


@cli.command()
@click.pass_context
@click.option(
    "--steps-per-epoch",
    type=int,
    default=int(1e3),
    help="Number of batches in an epoch.",
)
@click.option(
    "--training-passes",
    type=int,
    default=1,
    help="Number of iterations through the whole training set.",
)
@click.option(
    "--learning-rate", type=float, default=None, help="Set new learning rate."
)
def train(ctx, steps_per_epoch: int, training_passes: int, learning_rate: float):
    """
    Train in a previously created working area.
    """
    training_directory = get_training_directory(ctx.obj["TAG"])

    loader = TrainingLoader(training_directory)

    model = loader.get_model("latest")

    if learning_rate:
        assert learning_rate > 0, "Learning rate should be positive."
        K.set_value(model.optimizer.learning_rate, learning_rate)

    training_sequence = loader.get_sequence("training")
    validation_sequence = loader.get_sequence("validation")
    assert training_sequence._feature_scaler
    assert validation_sequence._feature_scaler
    steps_total = len(training_sequence)
    epochs = training_passes * steps_total // steps_per_epoch

    checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            training_directory, "models", "checkpoint_epoch{epoch:02d}_loss{loss:.2e}"
        ),
        save_weights_only=False,
        mode="auto",
        save_freq=steps_total,
    )
    checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(training_directory, "models", "checkpoint_latest"),
        save_weights_only=False,
        mode="auto",
        save_freq=steps_total,
    )

    validation_freq = 1  # epochs // training_passes

    model.fit(
        x=training_sequence,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        max_queue_size=0,
        shuffle=False,
        validation_data=validation_sequence,
        validation_freq=validation_freq,
        callbacks=[checkpoint1, checkpoint2],
    )

    model.save(
        os.path.join(training_directory, "models/latest"), include_optimizer=True
    )

    def prepend_path(fname):
        return os.path.join(training_directory, fname)

    try:
        history = loader.get_history()
    except:
        history = {}
    history = append_history(
        history, model.history.history, validation_frequence=validation_freq
    )

    save(history, prepend_path("history.pkl"))


if __name__ == "__main__":
    cli()
