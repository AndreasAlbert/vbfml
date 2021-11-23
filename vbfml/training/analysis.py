import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

import hist
import numpy as np
import sklearn
from tqdm import tqdm
from vbfml.input.sequences import MultiDatasetSequence
from vbfml.training.data import TrainingLoader

# Hard coded sane default binnings
# If it extends more, might warrant
# separate configuration file.
dphi_bins = np.linspace(0, np.pi, 20)
phi_bins = np.linspace(-np.pi, np.pi, 20)
pt_bins = np.logspace(2, 3, 20)
eta_bins = np.linspace(-5, 5, 20)
axis_bins = {
    "mjj": np.logspace(2, 4, 20),
    "dphijj": dphi_bins,
    "detajj": np.linspace(0, 10, 20),
    "mjj_maxmjj": np.logspace(2, 4, 20),
    "dphijj_maxmjj": dphi_bins,
    "detajj_maxmjj": np.linspace(0, 10, 20),
    "recoil_pt": pt_bins,
    "dphi_ak40_met": dphi_bins,
    "dphi_ak41_met": dphi_bins,
    "ht": pt_bins,
    "leadak4_pt": pt_bins,
    "leadak4_phi": phi_bins,
    "leadak4_eta": eta_bins,
    "trailak4_pt": pt_bins,
    "trailak4_phi": phi_bins,
    "trailak4_eta": eta_bins,
    "leadak4_mjjmax_pt": pt_bins,
    "leadak4_mjjmax_phi": phi_bins,
    "leadak4_mjjmax_eta": eta_bins,
    "trailak4_mjjmax_pt": pt_bins,
    "trailak4_mjjmax_phi": phi_bins,
    "trailak4_mjjmax_eta": eta_bins,
    "score": np.linspace(0, 1, 100),
    "composition": np.linspace(-0.5, 4.5, 5),
    "transformed": np.linspace(-5, 5, 20),
}


@dataclass
class TrainingAnalyzer:
    """
    Analyzer to make histograms based on
    training / validation / test data sets.

    """

    directory: str
    cache: str = "analyzer_cache.pkl"

    data: "dict" = field(default_factory=dict)

    def __post_init__(self):
        self.loader = TrainingLoader(self.directory)
        self.cache = os.path.join(self.directory, os.path.basename(self.cache))

    def _make_histogram(self, quantity_name: str) -> hist.Hist:
        """Creates an empty histogram for a given quantity (feature or score)"""
        if "score" in quantity_name:
            bins = axis_bins["score"]
        elif "transform" in quantity_name:
            bins = axis_bins["transformed"]
        else:
            bins = axis_bins[quantity_name]
        histogram = hist.Hist(
            hist.axis.Variable(bins, name=quantity_name, label=quantity_name),
            hist.axis.IntCategory(range(10), name="label", label="label"),
            storage=hist.storage.Weight(),
        )

        return histogram

    def load_from_cache(self) -> bool:
        """Histograms loaded from disk cache."""
        success = False
        try:
            with open(self.cache, "rb") as f:
                data = pickle.load(f)
            success = True
        except:
            data = {}
        self.data = data
        return success

    def write_to_cache(self) -> bool:
        """Cached histograms written to disk."""
        with open(self.cache, "wb") as f:
            return pickle.dump(self.data, f)

    def analyze(self):
        """
        Loads all relevant data sets and analyze them.
        """
        histograms = {}
        for sequence_type in ["training", "validation"]:
            sequence = self.loader.get_sequence(sequence_type)
            sequence.scale_features = "none"
            sequence.batch_size = int(1e6)
            sequence.batch_buffer_size = 10
            (
                histogram_out,
                predicted_scores,
                validation_scores,
                weights,
            ) = self._analyze_sequence(sequence, sequence_type)
            histograms[sequence_type] = histogram_out
            if sequence_type == "validation":
                self.data["validation_scores"] = validation_scores
                self.data["predicted_scores"] = predicted_scores
                self.data["weights"] = weights
        self.data["histograms"] = histograms

    def _fill_feature_histograms(
        self,
        histograms: Dict[str, hist.Hist],
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        feature_scaler: Optional[sklearn.preprocessing.StandardScaler],
    ):
        feature_names = self.loader.get_features()

        if feature_scaler:
            features = feature_scaler.transform(features)

        for ifeat, feature_name in enumerate(feature_names):
            if feature_scaler:
                feature_name = f"{feature_name}_transform"
            if not feature_name in histograms:
                histograms[feature_name] = self._make_histogram(feature_name)
            histograms[feature_name].fill(
                **{
                    feature_name: features[:, ifeat].flatten(),
                    "label": labels,
                    "weight": weights.flatten(),
                }
            )

    def _fill_score_histograms(
        self,
        histograms: Dict[str, hist.Hist],
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        n_classes = scores.shape[1]
        for scored_class in range(n_classes):
            name = f"score_{scored_class}"
            if not name in histograms:
                histograms[name] = self._make_histogram(name)
            histograms[name].fill(
                **{
                    name: scores[:, scored_class],
                    "label": labels,
                    "weight": weights,
                }
            )

    def _fill_feature_covariance(
        self, features: np.ndarray, labels: np.ndarray, feature_scaler
    ):
        """
        Calculate covariance coefficients between different features (e.g. mjj vs met).

        Covariance coefficients are stored in a dictionary member of the analyzer.
        For each batch of feature values, the previously calculated covariance is updated
        by calculating the weighted mean between the old and new covariance values,
        using the underlying number of events as the weight for the mean.

        To make this procedure more accurate, the feature covariance is calculated
        after feature scaling, which ensures that the individual feature averages
        are close to zero, which simplifies the formula for the covariance.
        """
        feature_names = self.loader.get_features()
        features = feature_scaler.transform(features)

        if not "covariance" in self.data:
            self.data["covariance"] = defaultdict(dict)
            self.data["covariance_event_count"] = defaultdict(int)

        for iclass in range(labels.shape[1]):
            n_event_previous = self.data["covariance_event_count"][iclass]
            mask = labels[:, iclass] == 1
            n_events = np.sum(mask)
            for ifeat1, feat1 in enumerate(feature_names):
                for ifeat2, feat2 in enumerate(feature_names):
                    if ifeat2 < ifeat1:
                        continue

                    covariance = np.cov(
                        features[mask][:, ifeat1], features[mask][:, ifeat2]
                    )

                    key = tuple(sorted((feat1, feat2)))

                    previous_covariance = self.data["covariance"][iclass].get(key, 0)
                    updated_covariance = (
                        previous_covariance * n_event_previous + covariance * n_events
                    )
                    updated_covariance /= n_event_previous + n_events
                    self.data["covariance"][iclass][key] = updated_covariance

            self.data["covariance_event_count"][iclass] = n_event_previous + n_events

    def _analyze_sequence(
        self, sequence: MultiDatasetSequence, sequence_type: str
    ) -> Dict[str, hist.Hist]:
        """
        Analyzes a specific sequence.

        Loop over batches, make and return histograms.
        """
        histograms = {}
        model = self.loader.get_model()

        feature_scaler = self.loader.get_feature_scaler()
        predicted_scores = []
        validation_scores = []
        sample_weights = []
        for ibatch in tqdm(
            range(len(sequence)), desc=f"Analyze batches of {sequence_type} sequence."
        ):
            features, labels_onehot, weights = sequence[ibatch]
            labels = labels_onehot.argmax(axis=1)

            scores = model.predict(feature_scaler.transform(features))
            if sequence_type == "validation":
                predicted_scores.append(scores)
                validation_scores.append(labels_onehot)
                sample_weights.append(weights)

            for scaler in feature_scaler, None:
                self._fill_feature_histograms(
                    histograms, features, labels, weights, feature_scaler=scaler
                )

            self._fill_score_histograms(histograms, scores, labels, weights)
            self._fill_feature_covariance(features, labels_onehot, feature_scaler)
            # self._fill_composition_histograms(histograms, scores, labels, weights)

        return histograms, predicted_scores, validation_scores, weights
