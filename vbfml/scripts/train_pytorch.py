#!/usr/bin/env python3
import copy
import os
import re
import warnings
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate
from torch.optim import SGD
from tqdm import tqdm
from vbfml.models import Net
from vbfml.training.input import build_sequence, load_datasets_bucoffea
from vbfml.training.util import normalize_classes, save, select_and_label_datasets


def get_training_directory(tag: str) -> str:
    return os.path.join("./output", f"model_{tag}")


def weighted_crossentropy_loss_from_labels(
    true_labels: torch.Tensor, predicted_labels: torch.Tensor, weights: torch.Tensor
):
    """
    Manual implementation of the cross entropy loss for debugging.
    """
    log1 = torch.log(predicted_labels)
    log2 = torch.log(1 - predicted_labels)

    raw_loss = -true_labels * log1 - (1 - true_labels) * log2
    raw_loss[true_labels == predicted_labels] = 0
    loss = torch.sum(weights * raw_loss) / torch.sum(weights)
    return loss


def weighted_loss(model, features, true_labels, weights, loss_criterion):
    """
    Compute the weighted loss for a batch of data.
    """
    predicted_labels = model(features)

    raw_loss = loss_criterion(predicted_labels, true_labels)
    loss = torch.sum(weights * raw_loss) / torch.sum(weights)
    return loss, predicted_labels


def train(
    model,
    features: torch.Tensor,
    true_labels: torch.Tensor,
    weights: torch.Tensor,
    optimizer,
    loss_criterion,
):
    """
    One iteration of training on a batch of data.
    """
    model.zero_grad()
    loss, predicted_labels = weighted_loss(
        model, features, true_labels, weights, loss_criterion
    )
    loss.backward()
    optimizer.step()
    return loss, predicted_labels


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

all_datasets = load_datasets_bucoffea("/data/cms/vbfml/2021-08-25_treesForML_v2/")

dataset_labels = {
    "ewk_17": "(EWK.*2017|VBF_HToInvisible_M125_withDipoleRecoil_pow_pythia8_2017)",
    "v_qcd_nlo_17": "(WJetsToLNu_Pt-\d+To.*|Z\dJetsToNuNu_M-50_LHEFilterPtZ-\d+To\d+)_MatchEWPDG20-amcatnloFXFX_2017",
}
datasets = select_and_label_datasets(all_datasets, dataset_labels)
for dataset_info in datasets:
    if re.match(dataset_labels["v_qcd_nlo_17"], dataset_info.name):
        dataset_info.n_events = 0.02 * dataset_info.n_events
features = [
    "mjj",
    "dphijj",
    "detajj",
    # "mjj_maxmjj",
    # "dphijj_maxmjj",
    # "detajj_maxmjj",
    # "recoil_pt",
    # "dphi_ak40_met",
    # "dphi_ak41_met",
    # "ht",
    # "leadak4_pt",
    # "leadak4_eta",
    # "trailak4_pt",
    # "trailak4_eta",
    # "leadak4_mjjmax_pt",
    # "leadak4_mjjmax_eta",
    # "trailak4_mjjmax_pt",
    # "trailak4_mjjmax_eta",
]

training_sequence = build_sequence(
    datasets=copy.deepcopy(datasets),
    features=features,
    weight_expression="weight_total*xs/sumw",
)
validation_sequence = build_sequence(
    datasets=copy.deepcopy(datasets),
    features=features,
    weight_expression="weight_total*xs/sumw",
)
normalize_classes(training_sequence, target_integral=1e6)
normalize_classes(validation_sequence, target_integral=1e6)

# Training sequence
training_sequence.read_range = (0.0, 0.8)
training_sequence.scale_features = True
training_sequence.batch_size = 25
training_sequence.batch_buffer_size = 5e5
training_sequence[0]

# Validation sequence
validation_sequence.read_range = (0.8, 1.0)
validation_sequence.scale_features = True
validation_sequence._feature_scaler = copy.deepcopy(training_sequence._feature_scaler)
validation_sequence.batch_size = 1e4
validation_sequence.batch_buffer_size = 100


model = Net(n_features=len(features), n_classes=2, dropout=0.2)
print(model)


# Binary cross entropy loss
# Do not apply reduction, so that we can implement
# manual weighted reduction later on
criterion = nn.BCELoss(reduction="none")

learning_rate = 0.1


optm = SGD(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optm, factor=0.1, patience=5, threshold=1e-2, verbose=True, cooldown=2, min_lr=1e-5
)

# Use GPU
device = torch.device("cuda:0")
model.to(device)

history = defaultdict(list)

EPOCHS = 50

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0

    # Put model into training mode
    model.train(True)

    # Loop over training batches and train
    for bidx, batch in tqdm(enumerate(training_sequence)):
        x_train, y_train, w_train = batch
        loss, predictions = train(
            model,
            torch.Tensor(x_train).to(device),
            torch.Tensor(y_train).to(device),
            torch.Tensor(w_train).to(device),
            optm,
            criterion,
        )
        epoch_loss += loss.item()

    # Put model into inference mode
    model.train(False)

    # Calculate validation loss
    validation_loss = 0
    for bidx, batch in enumerate(validation_sequence):
        x, y, w = batch
        loss, _ = weighted_loss(
            model,
            torch.Tensor(x).to(device),
            torch.Tensor(y).to(device),
            torch.Tensor(w).to(device),
            criterion,
        )
        validation_loss += loss.item()

    # Pass validation loss to learning rate scheduler
    scheduler.step(validation_loss)

    print(f"Epoch {epoch+1} Loss : {epoch_loss} Validation loss : {validation_loss}")

    history["x_loss"].append(epoch)
    history["x_val_loss"].append(epoch)
    history["y_loss"].append(loss)
    history["y_val_loss"].append(validation_loss)


# Save everything
training_directory = get_training_directory("test_pytorch")


def prepend_path(fname):
    return os.path.join(training_directory, fname)


try:
    os.makedirs(training_directory)
except FileExistsError:
    pass

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
save(history, prepend_path("history.pkl"))
torch.save(model, prepend_path("model.pt"))
