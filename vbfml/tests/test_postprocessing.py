import copy
import os
import shutil
from unittest import TestCase

from vbfml.models import sequential_dense_model
from vbfml.tests.util import create_test_tree, make_tmp_dir
from vbfml.training.analysis import TrainingAnalyzer
from vbfml.training.data import TrainingLoader
from vbfml.training.input import build_sequence, load_datasets_bucoffea
from vbfml.training.plot import TrainingHistogramPlotter, plot_history
from vbfml.training.util import append_history, save


def keras_model_compare(model1, model2):
    for l1, l2 in zip(model1.layers, model2.layers):
        if not l1.get_config() == l2.get_config():
            return False
    return True


class TestTrainingLoader(TestCase):
    def setUp(self):
        self.wdir = make_tmp_dir()

        self.training_sequence_file = os.path.join(self.wdir, "training_sequence.pkl")
        self.validation_sequence_file = os.path.join(
            self.wdir, "validation_sequence.pkl"
        )
        self.feature_file = os.path.join(self.wdir, "features.pkl")
        self.history_file = os.path.join(self.wdir, "history.pkl")
        self.feature_scaler_file = os.path.join(self.wdir, "feature_scaler.pkl")

        self.model = sequential_dense_model(
            n_features=1,
            n_layers=1,
            n_nodes=[1],
            n_classes=1,
        )

        self.model.save(os.path.join(self.wdir, "models/latest"))
        save("training_sequence", self.training_sequence_file)
        save("validation_sequence", self.validation_sequence_file)
        save("features", self.feature_file)
        save("history", self.history_file)
        save("feature_scaler", self.feature_scaler_file)

    def test_loader(self):
        loader = TrainingLoader(self.wdir)
        self.assertTrue(keras_model_compare(loader.get_model(), self.model))
        self.assertEqual(loader.get_sequence("training"), "training_sequence")
        self.assertEqual(loader.get_sequence("validation"), "validation_sequence")
        self.assertEqual(loader.get_features(), "features")
        self.assertEqual(loader.get_history(), "history")
        self.assertEqual(loader.get_feature_scaler(), "feature_scaler")


class TestTrainingAnalysisAndPlot(TestCase):
    def setUp(self):
        self.wdir = make_tmp_dir()
        self.addCleanup(shutil.rmtree, self.wdir)

        files = [
            os.path.join(self.wdir, "tree_firstdataset_2017.root"),
            os.path.join(self.wdir, "tree_seconddataset_2017.root"),
        ]

        self.features = ["mjj"]
        for file in files:
            create_test_tree(
                filename=file,
                treename="sr_vbf",
                branches=self.features + ["weight_total", "xs", "sumw"],
                n_events=10,
                value=2,
            )
        self.addCleanup(os.remove, file)

        datasets = load_datasets_bucoffea(self.wdir)
        sequence = build_sequence(datasets, features=self.features)
        sequence.scale_features = "standard"
        self.training_sequence = sequence
        self.validation_sequence = copy.deepcopy(sequence)

        self.training_sequence_file = os.path.join(self.wdir, "training_sequence.pkl")
        self.validation_sequence_file = os.path.join(
            self.wdir, "validation_sequence.pkl"
        )
        self.feature_file = os.path.join(self.wdir, "features.pkl")
        self.history_file = os.path.join(self.wdir, "history.pkl")
        self.feature_scaler_file = os.path.join(self.wdir, "feature_scaler.pkl")

        self.model = sequential_dense_model(
            n_features=len(self.features),
            n_layers=1,
            n_nodes=[1],
            n_classes=2,
        )
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["categorical_accuracy"],
        )

        self.model.fit(
            self.training_sequence,
            shuffle=False,
            validation_data=self.validation_sequence,
        )

        self.model.save(os.path.join(self.wdir, "models", "latest"))
        save(self.training_sequence, self.training_sequence_file)
        save(self.validation_sequence, self.validation_sequence_file)
        save(self.features, self.feature_file)

        save(append_history({}, self.model.history.history), self.history_file)
        save(self.training_sequence._feature_scaler, self.feature_scaler_file)

        self.analyzer = TrainingAnalyzer(self.wdir)
        self.loader = TrainingLoader(self.wdir)

    def test_training_analyzer_cache(self):
        """Test cache mechanism"""

        # Cache does not exist at the start
        self.assertFalse(os.path.exists(self.analyzer.cache))
        self.assertFalse(self.analyzer.load_from_cache())

        # Write cache and confirm existence
        self.analyzer.write_to_cache()
        self.assertTrue(os.path.exists(self.analyzer.cache))
        self.assertTrue(self.analyzer.load_from_cache())
        self.assertEqual(self.analyzer.data, {})

    def test_training_analyzer_analysis(self):
        """Test event loop and histogram creation"""
        self.analyzer.analyze()
        for feature in self.features:
            for sequence in "training", "validation":
                self.assertTrue(feature in self.analyzer.data["histograms"][sequence])
                self.assertTrue("score_0" in self.analyzer.data["histograms"][sequence])
                self.assertTrue("score_1" in self.analyzer.data["histograms"][sequence])

    def test_plot(self):
        """Test that plotting runs -- no verification of output"""
        self.analyzer.analyze()
        plotter = TrainingHistogramPlotter(
            self.analyzer.data["weights"],
            self.analyzer.data["predicted_scores"],
            self.analyzer.data["validation_scores"],
            self.analyzer.data["histograms"],
        )
        plotter.plot_by_sequence_types()
        plot_history(self.loader.get_history(), outdir=self.wdir)
