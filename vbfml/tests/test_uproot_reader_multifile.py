import os
from unittest import TestCase

import numpy as np
from vbfml.input.uproot import UprootReaderMultiFile
from vbfml.tests.util import create_test_tree, make_tmp_dir


class TestUprootReaderMultiFile(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.branches = ["a", "b"]
        self.nevents_per_file = 10
        self.n_file = 5
        self.n_features = len(self.branches)
        self.values = list(range(self.n_file))
        self.total_events = self.nevents_per_file * self.n_file
        self.files = []
        self.wdir = make_tmp_dir()
        self.addCleanup(os.rmdir, self.wdir)
        
        for i in range(self.n_file):
            fname = os.path.join(self.wdir, f"test_single_{i}.root")

            create_test_tree(
                filename=fname,
                treename=self.treename,
                branches=self.branches,
                n_events=self.nevents_per_file,
                value=self.values[i],
            )
            self.files.append(fname)
            self.addCleanup(os.remove, fname)

        self.reader = UprootReaderMultiFile(
            files=self.files,
            branches=self.branches,
            treename=self.treename,
        )

    def test_file_list(self):
        """Test that the file list is propagated correctly"""
        expected_keys = list(sorted(self.files))
        observed_keys = list(sorted(self.reader.files))
        self.assertTrue(expected_keys == observed_keys)

    def test_nevents(self):
        """Test that the number of events is calculated correctly"""
        self.assertTrue(len(self.reader.nevents_per_file))
        for file_index, nevents in self.reader.nevents_per_file.items():
            filepath = self.reader._file_path(file_index)
            self.assertTrue(filepath in self.files)
            self.assertTrue(nevents == self.nevents_per_file)

    def test_index_into_file(self):
        """Test translation of global event index to file index + local event index"""
        for global_event_index in range(self.total_events):
            file_index, local_event_index = self.reader._index_into_file(
                global_event_index
            )
            expected_file_index = global_event_index // self.nevents_per_file
            expected_local_event_index = global_event_index % self.nevents_per_file
            self.assertEqual(file_index, expected_file_index)
            self.assertEqual(local_event_index, expected_local_event_index)

    def test_read_event_features_single_file(self):
        """Read data without crossing file boundaries, test output shape & content"""
        start = 0
        stop = 5

        for file_index in range(len(self.files)):
            df = self.reader.read_events_single_file(
                file_index=file_index, local_start=start, local_stop=stop
            )

            expected_nevents = stop - start
            self.assertEqual(len(df), expected_nevents)
            self.assertEqual(len(df.columns), self.n_features)

            for column in df.columns:
                self.assertTrue(np.all(df[column] == file_index))

    def test_readevents(self):
        """Read data across file boundaries, test output shape & content"""
        for start in range(0, self.total_events, 13):
            for stop in range(start, self.total_events, 19):
                df = self.reader.read_events(start=start, stop=stop)

                # Test shape
                expected_nevents = stop - start
                self.assertEqual(len(df), expected_nevents)
                self.assertEqual(len(df.columns), self.n_features)

                # Test content
                for branch in self.branches:
                    branch_data = list(df[branch])
                    expected = [
                        (start + i) // self.nevents_per_file
                        for i in range(expected_nevents)
                    ]
                    self.assertListEqual(branch_data, expected)

    def test_read_continuous(self):
        def value_list(value_index, length):
            return [self.values[value_index]] * length

        def compare(df, expected):
            for branch in self.branches:
                self.assertListEqual(list(df[branch]), expected)

        # Part of first file
        n_read = self.nevents_per_file // 2
        df = self.reader.read_events_continuous(n_read)
        expected = value_list(0, n_read)
        compare(df, expected)

        # Rest of first file
        n_read = self.nevents_per_file - n_read
        df = self.reader.read_events_continuous(n_read)
        expected = value_list(0, n_read)
        compare(df, expected)

        # Entire second file
        n_read = self.nevents_per_file
        df = self.reader.read_events_continuous(n_read)
        expected = value_list(1, n_read)
        compare(df, expected)

        # Third file + part of fourth
        n_read = self.nevents_per_file + 3
        df = self.reader.read_events_continuous(n_read)
        expected = value_list(2, self.nevents_per_file) + value_list(3, 3)
        compare(df, expected)

        # Rest of fourth + part of fifth
        n_read = self.nevents_per_file
        df = self.reader.read_events_continuous(n_read)
        expected = value_list(3, self.nevents_per_file - 3) + value_list(4, 3)
        compare(df, expected)

        # Reset and make sure we come out at the start again
        self.reader.reset_continuous_read()
        n_read = self.nevents_per_file // 2
        df = self.reader.read_events_continuous(n_read)
        expected = value_list(0, n_read)
        compare(df, expected)
