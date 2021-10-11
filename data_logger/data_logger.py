import os
from typing import List

import numpy as np
import pandas as pd


class DataLogger:
    """Class for logging experimental data.

    Data can be stored in a csv.
    """

    def __init__(self, checkpoint_path: str, logfile_path: str, columns: List[str]):
        self._checkpoint_path = checkpoint_path
        self._logfile_path = logfile_path
        self._df_columns = columns
        self._logger_data = {}

    def write_scalar(self, tag: str, step: int, scalar: float) -> None:
        """Write (scalar) data to dictionary.

        Args:
            tag: tag for data to be logged.
            step: current step count.
            scalar: data to be written.
        """
        if tag not in self._logger_data:
            self._logger_data[tag] = {}

        self._logger_data[tag][step] = scalar

    def checkpoint(self) -> None:
        """Construct dataframe from data and merge with previously saved checkpoint.

        Raises:
            AssertionError: if columns of dataframe to be appended do
            not match previous checkpoints.
        """
        assert set(self._logger_data.keys()) == set(
            self._df_columns
        ), "Incorrect dataframe columns for merging"

        series_data = {k: pd.Series(self._logger_data[k]) for k in self._df_columns}

        # only append header on first checkpoint/save.
        header = not os.path.exists(self._logfile_path)
        pd.DataFrame(series_data).to_csv(
            self._logfile_path, mode="a", header=header, index=False
        )

        # reset logger in memory to empty.
        self._logger_data = {}

    def write_array_data(self, file_name: str, data: np.ndarray) -> None:
        """Write array data to np save file.

        Args:
            file_name: filename for save.
            data: data to save.
        """
        full_path = os.path.join(self._checkpoint_path, file_name)
        np.save(file=full_path, arr=data)
