import os
from copy import deepcopy
from unittest import TestCase

import numpy as np
from tensorflow.keras.utils import to_categorical
from vbfml.input.sequences import DatasetInfo, MultiDatasetSequence
from vbfml.tests.util import create_test_tree
from vbfml.models import sequential_dense_model


class TestMultiDatasetSequence(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.branches = ["a", "b"]
        self.nevents_per_file = 500
        self.nevents_per_dataset = [100, 500]
        self.n_datasets = 2
        self.n_features = len(self.branches)
        self.values = list(range(self.n_datasets))
        self.total_events = self.nevents_per_file * self.n_datasets
        self.dataset = "dataset"
        self.files = []
        self.batch_size = 50

        self.mds = MultiDatasetSequence(
            batch_size=self.batch_size, branches=self.branches, shuffle=False
        )

        for i in range(self.n_datasets):
            name = f"dataset_{i}"
            fname = os.path.abspath(f"test_{name}.root")
            create_test_tree(
                filename=fname,
                treename=self.treename,
                branches=self.branches,
                n_events=self.nevents_per_file,
                value=self.values[i],
            )
            self.files.append(fname)
            self.addCleanup(os.remove, fname)

            dataset = DatasetInfo(
                name=name,
                files=[fname],
                n_events=self.nevents_per_dataset[i],
                treename=self.treename,
            )
            self.mds.add_dataset(dataset)

    def test_n_dataset(self):
        """Test that the number of datasets is as expected"""
        self.assertEqual(len(self.mds.datasets), self.n_datasets)

    def test_total_events(self):
        """Test that the number of total events is as expected"""
        self.assertEqual(self.mds.total_events(), sum(self.nevents_per_dataset))

    def test_fractions(self):
        """Test that the fraction of each dataset relative to the total events is correct."""
        total_dataset_events = sum(self.nevents_per_dataset)
        for i in range(len(self.nevents_per_dataset)):
            self.assertAlmostEqual(
                self.mds.fractions[f"dataset_{i}"],
                self.nevents_per_dataset[i] / total_dataset_events,
            )

    def test_add_dataset_guard(self):
        """Test that add_dataset is guarded against incorrect usage"""
        # Try to add existing data set again
        dataset = self.mds.get_dataset("dataset_0")
        with self.assertRaises(IndexError):
            self.mds.add_dataset(dataset)

    def test_remove_dataset(self):
        """Test removing a data set from the Sequence"""
        info = self.mds.remove_dataset("dataset_0")
        new_n_datasets = self.n_datasets - 1
        self.assertEqual(len(self.mds.datasets), new_n_datasets)
        self.assertEqual(len(self.mds.fractions), new_n_datasets)

        # The label_encoding dict stores two entries per data set
        # 1. data set name -> integer identifier
        # 2. integer identifier -> data set name
        self.assertEqual(len(self.mds.label_encoding), 2 * new_n_datasets)

        self.assertEqual(info.name, "dataset_0")

    def test_batch_shapes(self):
        """
        Test that the individual batches are shaped correctly.

        The expected shape is
        (N_batch, N_feature) for features
        (N_batch, 1) for labels
        """

        for idx in range(len(self.mds)):
            batch = self.mds[idx]

            # In unweighted readout, batch should be a tuple of two
            self.assertEqual(len(batch), 2)
            features, labels = batch

            # First index must agree between labels, features
            self.assertEqual(labels.shape[0], features.shape[0])

            # Batch size might vary slightly
            self.assertTrue(features.shape[0] < self.batch_size * 1.5)
            self.assertTrue(features.shape[0] > self.batch_size / 1.5)

            # Second index differs
            self.assertEqual(labels.shape[1], len(self.mds.datasets))
            self.assertEqual(features.shape[1], len(self.branches))

    def test_batch_content_noshuffle(self):
        self.mds.shuffle = False
        features, labels = self.mds[0]

        first_dataset = True

        for i_batch in range(features.shape[0]):

            x = features[i_batch, :]
            y = labels[i_batch, :]

            label_0 = to_categorical(
                self.mds.encode_label("dataset_0"), num_classes=self.n_datasets
            )
            label_1 = to_categorical(
                self.mds.encode_label("dataset_1"), num_classes=self.n_datasets
            )

            # Check that values and labels agree
            valid = False
            valid |= (x[0] == 0) and np.all(y == label_0)
            valid |= (x[0] == 1) and np.all(y == label_1)
            self.assertTrue(valid)

            # Since we did not shuffle, can check
            # that we never see dataset 0 again
            # after once seeing dataset 1
            if x[0] == 1:
                first_dataset = False
            if not first_dataset:
                self.assertEqual(x[0], 1)

    def test_batch_content_custom_dataset_label(self):
        self.mds.shuffle = False

        # Add two new datasets with different names
        # but same label
        info = deepcopy(self.mds.get_dataset("dataset_0"))
        info.name = "dataset_0_copy_1"
        info.label = "some_label"
        self.mds.add_dataset(info)

        info = deepcopy(self.mds.get_dataset("dataset_0"))
        info.name = "dataset_0_copy_2"
        info.label = "some_label"
        self.mds.add_dataset(info)

        # Remove original data sets for simplicity
        self.mds.remove_dataset("dataset_0")
        self.mds.remove_dataset("dataset_1")

        _, y = self.mds[0]
        # Check that we have exactly two data sets
        self.assertEqual(len(self.mds.datasets), 2)

        # But only one label
        # the construction with argmax is needed to undo
        # the one-hot encoding
        self.assertEqual(len(np.unique(np.argmax(y, axis=1))), 1)

    def test_keras(self):
        """
        Ensure that our output does not make keras crash. No validation of result!
        """
        model = sequential_dense_model(
            n_features=len(self.branches),
            n_layers=1,
            n_nodes=[2],
            n_classes=len(self.mds.dataset_labels()),
        )

        model.summary()
        model.fit(self.mds, epochs=1)


class TestMultiDatasetSequenceSplit(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.branches = ["a"]
        self.nevents_per_file = 103
        self.values = list(range(self.nevents_per_file))
        self.total_events = self.nevents_per_file
        self.files = []

        self.mds = MultiDatasetSequence(
            batch_size=37, branches=self.branches, shuffle=False
        )

        fname = os.path.abspath(f"test.root")
        create_test_tree(
            filename=fname,
            treename=self.treename,
            branches=self.branches,
            n_events=self.nevents_per_file,
            value=self.values,
        )
        self.files.append(fname)
        self.addCleanup(os.remove, fname)

        dataset = DatasetInfo(
            name="dataset",
            files=[fname],
            n_events=self.nevents_per_file,
            treename=self.treename,
        )
        self.mds.add_dataset(dataset)

    def _test_read_range(self, read_range):
        self.mds.read_range = read_range

        def in_range(x):
            n_values = len(self.values)
            fraction = x / n_values
            return read_range[0] <= fraction <= read_range[1]

        valid_values = set(filter(in_range, self.values))

        for features, _ in self.mds:
            all_valid = all([x in valid_values for x in features.flatten()])
            self.assertTrue(all_valid)

    def test_read_range(self):
        ranges = [(0.1, 0.9), (0.25, 0.75), (0.3, 1.0), (0.0, 0.7)]
        for irange in ranges:
            self._test_read_range(irange)


class TestMultiDatasetSequenceWeight(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.feature_branches = ["a"]
        self.weight_branch = "b"
        self.weight_expression = "2*b"
        self.nevents_per_file = 103
        self.all_branches = self.feature_branches + [self.weight_branch]
        self.values = list(range(self.nevents_per_file))
        self.total_events = self.nevents_per_file
        self.files = []

        self.mds = MultiDatasetSequence(
            batch_size=37,
            branches=self.feature_branches,
            shuffle=False,
            weight_expression=self.weight_expression,
        )

        fname = os.path.abspath(f"test.root")
        create_test_tree(
            filename=fname,
            treename=self.treename,
            branches=self.all_branches,
            n_events=self.nevents_per_file,
            value=self.values,
        )
        self.files.append(fname)
        self.addCleanup(os.remove, fname)

        dataset = DatasetInfo(
            name="dataset",
            files=[fname],
            n_events=self.nevents_per_file,
            treename=self.treename,
        )
        self.mds.add_dataset(dataset)

    def test_weighted_batch_shapes(self):
        """
        Test that the individual batches are shaped correctly.

        The expected shape is
        (N_batch, N_feature) for features
        (N_batch, 1) for labels
        (N_batch, 1) for weights
        """

        for idx in range(len(self.mds)):
            batch = self.mds[idx]

            # For weighted readout, batch should be a tuple of three
            self.assertEqual(len(batch), 3)
            features, labels, weights = batch

            # First index must agree between labels, features
            self.assertEqual(labels.shape[0], features.shape[0])
            self.assertEqual(labels.shape[0], weights.shape[0])
            self.assertEqual(labels.shape[1], weights.shape[1])

            # Second index
            self.assertEqual(labels.shape[1], len(self.mds.datasets))
            self.assertEqual(features.shape[1], len(self.feature_branches))
            self.assertEqual(weights.shape[1], 1)

    def test_batch_content_weighted(self):
        """Test that weights are loaded correctly"""
        self.mds.shuffle = False
        features, _, weights = self.mds[0]

        if self.weight_expression == "2*b":
            expected_weights = list(2 * features.flatten())
        else:
            raise NotImplementedError(
                "Test for weight expressions other than '2*b' not implemented!"
            )

        self.assertListEqual(expected_weights, list(weights.flatten()))

    def test_keras(self):
        """
        Ensure that our output does not make keras crash. No validation of result!
        """
        model = sequential_dense_model(
            n_features=len(self.feature_branches),
            n_layers=1,
            n_nodes=[2],
            n_classes=len(self.mds.dataset_labels()),
        )

        model.summary()
        model.fit(self.mds, epochs=1)


class TestMultiDatasetSequenceFeatureScaling(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.feature_branches = ["a"]
        self.weight_branch = "b"
        self.weight_expression = "b"
        self.nevents_per_file = int(1e4)
        self.all_branches = self.feature_branches + [self.weight_branch]

        self.feature_mean = 2
        self.feature_std = 5
        self.values = (
            self.feature_std * np.random.randn(self.nevents_per_file)
            + self.feature_mean
        )
        self.total_events = self.nevents_per_file
        self.files = []

        self.mds = MultiDatasetSequence(
            batch_size=int(1e3),
            branches=self.feature_branches,
            shuffle=False,
            weight_expression=self.weight_expression,
        )

        fname = os.path.abspath(f"test.root")
        create_test_tree(
            filename=fname,
            treename=self.treename,
            branches=self.all_branches,
            n_events=self.nevents_per_file,
            value=self.values,
        )
        self.files.append(fname)
        self.addCleanup(os.remove, fname)

        dataset = DatasetInfo(
            name="dataset",
            files=[fname],
            n_events=self.nevents_per_file,
            treename=self.treename,
        )
        self.mds.add_dataset(dataset)

    def test_feature_scaling(self):
        def deviation_from_target(features):
            """
            Mean is supposed to be ~=0, std dev ~=1
            -> Calculate and return the absolute differences
            """
            deviation_mean = np.max(np.abs(np.mean(features, axis=0)))
            deviation_std = np.max(np.abs(np.std(features, axis=0) - 1))
            return deviation_mean, deviation_std

        # Read without feature scaling
        self.mds.scale_features = False
        features, _, weights = self.mds[0]
        dev_mean, dev_std = deviation_from_target(features)
        self.assertNotAlmostEqual(dev_mean, self.feature_mean)
        self.assertNotAlmostEqual(dev_std, self.feature_std - 1)
        self.assertTrue(np.all(features == weights))

        # Read with feature scaling
        self.mds.scale_features = True
        features, _, weights = self.mds[0]
        dev_mean, dev_std = deviation_from_target(features)
        self.assertAlmostEqual(dev_mean, 0)
        self.assertAlmostEqual(dev_std, 0)
        self.assertTrue(np.all(features != weights))
