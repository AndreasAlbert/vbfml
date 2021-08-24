import os
from unittest import TestCase

from vbfml.input.datagen import SingleDatasetGeneratorUproot
from vbfml.tests.util import create_test_tree


class TestSingleDatasetGen(TestCase):

    def setUp(self):
        self.treename = "tree"
        self.branches = ["a","b"]
        self.n_events = 10
        self.n_file = 2
        self.total_events = self.n_events * self.n_file

        files = []
        for i in range(self.n_file):
            fname = os.path.abspath(f"test_single_{i}.root")

            create_test_tree(
                filename=fname,
                treename=self.treename,
                branches=self.branches,
                n_events=self.n_events,
                max_instances=1
            )
            files.append(fname)
            self.addCleanup(os.remove, fname)

        self.sdg = SingleDatasetGeneratorUproot(
            files=files,
            branches=self.branches,
            treename=self.treename,
            dataset="dataset"
        )

    def test_full_read_no_overflow(self):
        '''Read all events in one go.'''
        try:
            x, y = self.sdg.read_events(self.total_events)
        except EOFError:
            self.fail("SingleDatasetGenerator raised unexpected EOFError.")
        self.assertTrue(x.shape == (self.total_events, len(self.branches)))
        self.assertTrue(y.shape == (self.total_events, 1))

    def test_partial_read_no_overflow_two_files(self):
        '''Read some events in one go, accessing two files.'''
        try:
            x, y = self.sdg.read_events(self.total_events - 1)
        except EOFError:
            self.fail("SingleDatasetGenerator raised unexpected EOFError.")
        self.assertTrue(x.shape == (self.total_events-1, len(self.branches)))
        self.assertTrue(y.shape == (self.total_events-1, 1))

    def test_partial_read_no_overflow_single_file(self):
        '''Read some events in one go, only accessing one file.'''
        try:
            x, y = self.sdg.read_events(3)
        except EOFError:
            self.fail("SingleDatasetGenerator raised unexpected EOFError.")
        self.assertTrue(x.shape == (3, len(self.branches)))
        self.assertTrue(y.shape == (3, 1))

    def test_full_read_with_overflow(self):
        '''Try to read more events than exist'''
        with self.assertRaises(EOFError):
            self.sdg.read_events(self.total_events+1)

    def test_partial_reads_with_overflow(self):
        '''Try to read more events than exist in multiple steps'''
        ### First read: All but one event
        try:
            x, y = self.sdg.read_events(self.total_events - 1)
        except EOFError:
            self.fail("SingleDatasetGenerator raised unexpected EOFError.")
        self.assertTrue(x.shape == (self.total_events - 1, len(self.branches)))
        self.assertTrue(y.shape == (self.total_events - 1, 1))

        ### Second read: Last event
        try:
            x, y = self.sdg.read_events(1)
        except EOFError:
            self.fail("SingleDatasetGenerator raised unexpected EOFError.")
        self.assertTrue(x.shape == (1, len(self.branches)))
        self.assertTrue(y.shape == (1, 1))

        ### Second read: no events left, should fail
        with self.assertRaises(EOFError):
            self.sdg.read_events(2)