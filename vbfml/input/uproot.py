import pandas as pd
import uproot


class UprootReaderMultiFile:
    """
    Wrapper class for reading data from multiple consecutive ROOT files.

    The implementation hides the multi-file nature of the input data from
    the user. Data is treated as a continuous stream of events without
    file boundaries, and arbitrary continuous sequences of events can be
    read.
    """

    def __init__(
        self, files: "list[str]", branches: "list[str]", treename: str
    ) -> None:
        self.files = files
        self.branches = branches
        self.n_files = len(self.files)
        self.treename = treename
        self.nevents_per_file = {}
        self._update_nevents_dict_all_files()
        self.reset_continuous_read()

    def _open_file(self, file_index):
        """Open and return file object for given index."""
        return uproot.open(self._file_path(file_index))

    def _get_tree(self, file_index):
        """Open file and return tree object for given index."""
        return self._open_file(file_index)[self.treename]

    def _file_path(self, file_index):
        """Translate file index to file path"""
        return self.files[file_index]

    def _update_nevents_dict_single_file(self, file_index):
        """Save number of events in this file into the cache"""
        self.nevents_per_file[file_index] = self._get_tree(file_index).num_entries

    def _update_nevents_dict_all_files(self):
        """Save number of events in all files into the cache"""
        for file_index in range(self.n_files):
            self._update_nevents_dict_single_file(file_index)

    def _index_into_file(self, global_event_index):
        """
        Translate global file index to tuple of (file index, local event index).

        A global event index is just the index of the desired events in the list of all events in all files (between 0 and total event number)

        The file index is the index of the file that contains this event, and the local event index is the index of the given event counting from the beginning of the file.
        """
        events_before = 0

        target_file_index = None
        local_event_index = None

        for file_index in range(self.n_files):
            nevents = self.nevents_per_file[file_index]

            right_file = global_event_index < events_before + nevents
            if right_file:
                target_file_index = file_index
                local_event_index = global_event_index - events_before
                break
            events_before += nevents

        return (target_file_index, local_event_index)

    def read_events_single_file(self, file_index, local_start, local_stop):
        """
        Read and return event data, within in a given file.
        """
        tree = self._get_tree(file_index)
        df = tree.arrays(
            expressions=self.branches,
            entry_start=local_start,
            entry_stop=local_stop,
            library="pandas",
        )
        return df

    def read_events(self, start, stop):
        """
        Read and return event data, possibly across file boundaries.
        """
        file_index_start, local_event_index_start = self._index_into_file(start)
        assert file_index_start is not None
        assert local_event_index_start is not None

        # If more events are requested than exist, read until the end
        file_index_stop, local_event_index_stop = self._index_into_file(stop)
        if file_index_stop is None:
            file_index_stop = self.n_files - 1
            local_event_index_stop = self.nevents_per_file[file_index_stop]

        dataframes = []
        for file_index in range(file_index_start, file_index_stop + 1):
            # Read from start except in first file
            local_start = 0
            if file_index == file_index_start:
                local_start = local_event_index_start

            # Read until the end except in last file
            local_stop = None
            if file_index == file_index_stop:
                local_stop = local_event_index_stop

            df = self.read_events_single_file(file_index, local_start, local_stop)

            dataframes.append(df)

        df = pd.concat(dataframes)

        return df

    def read_events_continuous(self, n_events_to_read: int) -> pd.DataFrame:
        """Read and return the next N events"""
        start = int(self.continuous_read_position)
        stop = start + n_events_to_read
        df = self.read_events(start, stop)
        self.continuous_read_position += n_events_to_read
        return df

    def reset_continuous_read(self):
        """Move read position back to start"""
        self.continuous_read_position = 0
