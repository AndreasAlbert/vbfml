import os
import warnings
import matplotlib.colors
import matplotlib.cbook
import mplhep as hep
import numpy as np

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from hist import Hist
from matplotlib import pyplot as plt
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from tqdm import tqdm
from sklearn import metrics

# plt.style.use(hep.style.CMS)

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

pjoin = os.path.join


def keys_to_list(keys):
    return list(sorted(map(str, keys)))


def merge_flow_bins(values):
    new_values = values[1:-1]
    new_values[0] = new_values[0] + values[0]
    new_values[-1] = new_values[-1] + values[-1]
    return new_values


colors = ["b", "r", "k", "g"]


@dataclass
class PlotSaver:
    outdir: str

    def __post_init__(self):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def save(self, fig: plt.figure, outfilename: str):
        """Save a given figure object.

        Args:
            fig (plt.figure): The figure object.
            outfilename (str): The path to save the file.
        """
        outpath = pjoin(self.outdir, outfilename)
        fig.savefig(outpath)
        plt.close(fig)


@dataclass
class PlotterBase:
    """
    Base class for plotters.
    """

    weights: "list"
    predicted_scores: "list"
    validation_scores: "list"
    histograms: "list[Hist]"
    output_directory: str = "./output"
    features: "Optional[np.array]" = None

    def __post_init__(self):
        self.sequence_types = keys_to_list(self.histograms.keys())
        histogram_names = []
        for sequence_type in self.sequence_types:
            histogram_names.extend(keys_to_list(self.histograms[sequence_type].keys()))
        self.histogram_names = sorted(list(set(histogram_names)))

        self.output_directory = os.path.abspath(self.output_directory)

        self.saver = PlotSaver(self.output_directory)

    def create_output_directory(self, subdirectory: str) -> None:
        directory = os.path.join(self.output_directory, subdirectory)
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

    def get_histogram(self, sequence_type: str, name: str) -> Hist:
        return self.histograms[sequence_type][name]

    def get_histograms_across_sequences(self, histogram_name: str) -> "list[Hist]":
        histograms = {}
        for sequence_type in self.sequence_types:
            histograms[sequence_type] = self.get_histogram(
                sequence_type, histogram_name
            )
        return histograms


@dataclass
class ImageTrainingPlotter(PlotterBase):
    """
    Plotter for image training results, created with ImageTrainingAnalyzer
    """

    def plot_confusion_matrix(self, normalize="true"):
        """Plots the confusion matrix based on truth and predicted labels.

        Args:
            normalize (str, optional): Whether to normalize the rows in the matrix. Defaults to 'true'.
        """
        truth_labels = self.validation_scores.argmax(axis=1)
        predicted_labels = self.predicted_scores.argmax(axis=1)

        cm = metrics.confusion_matrix(
            truth_labels,
            predicted_labels,
            normalize=normalize,
            sample_weight=self.weights.flatten(),
        )

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        ax.text(
            1,
            1,
            f"# of Events: {len(truth_labels)}",
            fontsize=14,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

        self.saver.save(fig, "confusion_matrix.pdf")

    def _add_ax_labels(self, ax: plt.axis):
        """Include axis labels for 2D eta/phi plots.

        Args:
            ax (plt.axis): The ax object to plot on.
        """
        ax.set_xlabel(r"PF Candidate $\eta$")
        ax.set_ylabel(r"PF Candidate $\phi$")

    def _plot_label(self, ax: plt.axis, truth_label: int, prediction_scores: list):
        """Add label info on the plot.

        Args:
            ax (plt.axis): The ax object to plot on.
            truth_label (int): The class label.
            prediction_scores (int): The predicted class label.
        """
        ax.text(
            0,
            1,
            f"Label: {truth_label}",
            fontsize=14,
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

        score_text = f"""Score for 0: {prediction_scores[0]:.3f}
        Score for 1: {prediction_scores[1]:.3f}"""

        ax.text(
            1,
            1,
            score_text,
            fontsize=10,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

    def plot_features(self, nevents: int = 5, image_shape: tuple = (40, 20)):
        """Plot several event images together with truth labels and predicted scores.

        Args:
            nevents (int, optional): Number of events to plot. Defaults to 5.
            image_shape (tuple, optional): Shape of each event image. Defaults to (40,20).
        """
        truth_labels = self.validation_scores.argmax(axis=1)
        predicted_labels = self.predicted_scores.argmax(axis=1)
        features = self.features[:nevents, :]

        new_shape = (features.shape[0], *image_shape)
        features = np.reshape(features, new_shape)

        etabins = np.linspace(-5, 5, image_shape[0])
        phibins = np.linspace(-np.pi, np.pi, image_shape[1])
        for idx in tqdm(range(len(features)), desc="Plotting images"):
            fig, ax = plt.subplots()
            cmap = ax.pcolormesh(
                etabins,
                phibins,
                features[idx].T,
                norm=matplotlib.colors.LogNorm(1e-3, 1e0),
            )

            self._plot_label(ax, truth_labels[idx], self.predicted_scores[idx])
            self._add_ax_labels(ax)

            cb = fig.colorbar(cmap)
            cb.set_label("Transformed Pixels")
            self.saver.save(fig, f"image_ievent_{idx}.pdf")

    def plot_scores(self):
        output_subdir = "scores"
        self.create_output_directory(output_subdir)

        for name in tqdm(self.histogram_names, desc="Plotting histograms"):
            histograms = self.get_histograms_across_sequences(name)

            for sequence, histogram in histograms.items():
                label_axis = [axis for axis in histogram.axes if axis.name == "label"][
                    0
                ]

                for label in label_axis.edges[:-1]:
                    label = int(label)
                    color = colors[label % len(colors)]

                    histogram_for_label = histogram[{"label": int(label)}]

                    edges = histogram_for_label.axes[0].edges
                    values = merge_flow_bins(histogram_for_label.values(flow=True))

                    if np.all(values == 0):
                        continue
                    variances = merge_flow_bins(
                        histogram_for_label.variances(flow=True)
                    )
                    norm = np.sum(values)

                    fig, ax = plt.subplots()

                    if sequence == "validation":
                        hep.histplot(
                            values / norm,
                            edges,
                            yerr=np.sqrt(variances) / norm,
                            histtype="errorbar",
                            label=f"class={label}, {sequence}",
                            marker="o",
                            color=color,
                            ax=ax,
                            markersize=5,
                        )

                        ax.set_xlabel("Score Value")
                        ax.set_ylabel("Normalized Counts")

                        ax.text(
                            0,
                            1,
                            f"score_{label}",
                            fontsize=14,
                            ha="left",
                            va="bottom",
                            transform=ax.transAxes,
                        )

                        self.saver.save(fig, f"score_dist_{label}.pdf")

    def plot(self) -> None:
        self.plot_features()
        self.plot_scores()
        self.plot_confusion_matrix()


@dataclass
class TrainingHistogramPlotter(PlotterBase):
    """
    Tool to make standard plots based on histograms created with
    the TrainingAnalyzer
    """

    def plot_by_sequence_types(self) -> None:
        """
        Comparison plots between the training and validation sequences.
        Generates plots for features + tagger scores.
        """
        output_subdir = "by_sequence"
        self.create_output_directory(output_subdir)

        for name in tqdm(self.histogram_names, desc="Plotting histograms"):
            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=(10, 10),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True,
            )
            histograms = self.get_histograms_across_sequences(name)

            for sequence, histogram in histograms.items():
                label_axis = [axis for axis in histogram.axes if axis.name == "label"][
                    0
                ]

                for label in label_axis.edges[:-1]:
                    label = int(label)
                    color = colors[label % len(colors)]

                    histogram_for_label = histogram[{"label": int(label)}]

                    edges = histogram_for_label.axes[0].edges
                    values = merge_flow_bins(histogram_for_label.values(flow=True))

                    if np.all(values == 0):
                        continue
                    variances = merge_flow_bins(
                        histogram_for_label.variances(flow=True)
                    )

                    norm = np.sum(values)

                    if sequence == "validation":
                        hep.histplot(
                            values / norm,
                            edges,
                            yerr=np.sqrt(variances) / norm,
                            histtype="errorbar",
                            label=f"class={label}, {sequence}",
                            marker="o",
                            color=color,
                            ax=ax,
                            markersize=5,
                        )
                    else:
                        values_normalized = values / norm
                        histogram1 = hep.histplot(
                            values / norm,
                            edges,
                            color=color,
                            label=f"class={label}, {sequence}",
                            ls="-",
                            ax=ax,
                        )
                        steppatch = (histogram1[0][0]).get_data()
                        values_hist = np.reshape(
                            np.asarray(steppatch[0]), (len(steppatch[0]), 1)
                        )
                        edges_hist = np.reshape(
                            (np.asarray(steppatch[1])), (len(steppatch[1]), 1)
                        )
                        integral = np.zeros((len(edges_hist), 1))
                        # compute efficency
                        for bin in range(1, len(values_hist) + 1):
                            integral[bin - 1] = sum(
                                values_hist[0:bin]
                                * np.diff(edges_hist[0 : bin + 1], axis=0)
                            )
                        rax.plot(
                            edges_hist[1:],
                            integral[:-1],
                            color=color,
                            label=f"class={label}, {sequence}",
                            axes=rax,
                        )
                    ax.legend()
                    rax.legend(prop={"size": 15})
                    rax.set_xlabel(name)
                    rax.set_ylabel("Efficency (a.u.)")
                    ax.set_ylabel("Events (a.u.)")
                    ax.set_yscale("log")
                    for ext in "pdf", "png":
                        fig.savefig(
                            os.path.join(
                                self.output_directory, output_subdir, f"{name}.{ext}"
                            )
                        )
                    plt.close(fig)

    def _covariance_dict_to_matrices(
        self, covariance: Dict[Tuple[str, str], float]
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        feature_names = sorted(set([x[0] for x in covariance.keys()]))
        n_feat = len(feature_names)

        matrices = {}
        matrices["covariance"] = np.zeros((n_feat, n_feat))
        matrices["correlation"] = np.zeros((n_feat, n_feat))

        for ifeat1, feat1 in enumerate(feature_names):
            for ifeat2, feat2 in enumerate(feature_names):
                key = (feat1, feat2)
                if key not in covariance:
                    key = (feat2, feat1)
                cov = covariance[key][0, 1]
                v1 = covariance[(feat1, feat1)][0, 1]
                v2 = covariance[(feat2, feat2)][0, 1]

                matrices["covariance"][ifeat1, ifeat2] = cov
                matrices["correlation"][ifeat1, ifeat2] = cov / np.sqrt(v1 * v2)

        return feature_names, matrices

    def plot_covariance(
        self, covariance: Dict[int, Dict[Tuple[str, str], float]]
    ) -> None:

        ncolors = 10
        cmap = plt.get_cmap("RdBu", ncolors)
        for class_index, class_covariance in covariance.items():
            feature_names, matrices = self._covariance_dict_to_matrices(
                class_covariance
            )

            output_subdirectory = os.path.join(self.output_directory, "covariance")
            try:
                os.makedirs(output_subdirectory)
            except FileExistsError:
                pass

            for tag, matrix in matrices.items():
                fig = plt.figure()
                ax = plt.gca()

                if "correlation" in tag:
                    vmin, vmax = -1, 1
                else:
                    vmax = 1.2 * np.max(np.abs(matrix))
                    vmin = -vmax

                cax = ax.matshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
                cbar = fig.colorbar(cax, shrink=0.6)
                cbar.set_label(tag)
                cbar.set_ticks(
                    [vmin + i * (vmax - vmin) / ncolors for i in range(ncolors + 1)]
                )
                plt.xticks(
                    list(range(len(feature_names))), feature_names, rotation="vertical"
                )
                plt.yticks(
                    list(range(len(feature_names))),
                    feature_names,
                    rotation="horizontal",
                )
                plt.subplots_adjust(left=0.3, bottom=0.0, right=0.9, top=0.8)

                for i in range(matrix.shape[0]):
                    ax.axhline(i + 0.5, color="k")
                    ax.axvline(i + 0.5, color="k")
                for ext in "pdf", "png":
                    fig.savefig(
                        os.path.join(
                            output_subdirectory,
                            f"feature_{tag}_class{class_index}.{ext}",
                        )
                    )

                plt.close(fig)

    def plot(self) -> None:
        self.plot_by_sequence_types()
        self.plot_ROC()

    # plot ROC curve for each class
    def plot_ROC(self) -> None:
        fpr = dict()
        tpr = dict()
        score_classes = np.array(self.predicted_scores).shape[2]
        number_batches = np.array(self.predicted_scores).shape[0]
        for sclass in tqdm(range(score_classes), desc="Plotting ROC curves"):
            # combine batch scores for each class
            p_score = np.array(self.predicted_scores[0][:, sclass])
            v_score = np.array(self.validation_scores[0][:, sclass])
            sample_weights = np.array(self.weights)
            predicted_scores_combined = p_score
            validation_scores_combined = v_score
            for batch in range(1, number_batches):
                sample_weights = np.append(sample_weights, sample_weights)
                p_score = np.asarray(self.predicted_scores[batch][:, sclass])
                v_score = np.asarray(self.validation_scores[batch][:, sclass])
                predicted_scores_combined = np.append(
                    predicted_scores_combined, p_score
                )
                validation_scores_combined = np.append(
                    validation_scores_combined, v_score
                )
            np.reshape(predicted_scores_combined, (len(predicted_scores_combined), 1))
            np.reshape(validation_scores_combined, (len(validation_scores_combined), 1))
            # make ROC curve
            fpr[sclass], tpr[sclass], thresholds = metrics.roc_curve(
                validation_scores_combined,
                predicted_scores_combined,
                sample_weight=sample_weights,
            )
            auc = metrics.roc_auc_score(
                validation_scores_combined,
                predicted_scores_combined,
                sample_weight=sample_weights,
            )
            fig, ax = plt.subplots()
            label_1 = "Current Classifier- AUC =" + str(round(auc, 3))
            ax.plot(fpr[sclass], tpr[sclass], label=label_1)
            ax.set_title("Label " + str(sclass))
            ax.set_ylabel("True Positive Rate")
            ax.set_xlabel("False Positive Rate")
            x = np.linspace(0, 1, 20)
            ax.plot(x, x, linestyle="--", label="Random Classifier")
            ax.legend()
            outdir = os.path.join(self.output_directory, "ROC")
            try:
                os.makedirs(outdir)
            except FileExistsError:
                pass

            for ext in "png", "pdf":
                fig.savefig(os.path.join(outdir, f"ROC_score_{sclass}.{ext}"))


def plot_history(history, outdir):

    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()

    accuracy_types = [
        key for key in history.keys() if "acc" in key and not "val" in key
    ]
    accuracy_types = [x.replace("x_", "").replace("y_", "") for x in accuracy_types]

    for accuracy_type in tqdm(accuracy_types, desc="Plotting History"):
        ax.plot(
            history["x_loss"],
            history["y_loss"],
            color="b",
            ls="--",
            marker="s",
            label="Training",
        )
        ax.plot(
            history["x_val_loss"],
            history["y_val_loss"],
            color="b",
            label="Validation",
            marker="o",
        )

        ax2 = ax.twinx()
        ax2.plot(
            history[f"x_{accuracy_type}"],
            history[f"y_{accuracy_type}"],
            color="r",
            ls="--",
            marker="s",
            label="Training",
        )
        ax2.plot(
            history[f"x_val_{accuracy_type}"],
            history[f"y_val_{accuracy_type}"],
            color="r",
            label="Validation",
            marker="o",
        )
        ax.set_xlabel("Training time (a.u.)")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax2.set_ylabel(f"Accuracy ({accuracy_type})")
        ax2.set_ylim(0, 1)
        ax.legend(title="Loss")
        ax2.legend(title="Accuracy")
        outdir = os.path.join(outdir, "history")
        try:
            os.makedirs(outdir)
        except FileExistsError:
            pass

        for ext in "png", "pdf":
            fig.savefig(os.path.join(outdir, f"history_{accuracy_type}.{ext}"))
