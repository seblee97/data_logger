import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class DataLogger:
    """Class for logging experimental data.

    Data can be stored in a csv.
    """

    def __init__(
        self,
        checkpoint_path: str,
        logfile_path: str,
        columns: List[str],
        index: Optional[str] = None,
    ):
        self._checkpoint_path = checkpoint_path
        self._logfile_path = logfile_path
        self._df_columns = columns
        self._index = index

        self._logger_data = {}

    @property
    def logger_data(self) -> Dict[str, np.ndarray]:
        return self._logger_data

    @logger_data.setter
    def logger_data(self, logger_data: Dict[str, np.ndarray]):
        self._logger_data = logger_data

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
        existing_keys = set(self._logger_data.keys())
        new_keys = set(self._df_columns)

        unspecified_keys = existing_keys - new_keys
        misspecified_keys = new_keys - existing_keys

        error_message_1 = (
            "Incorrect dataframe columns for merging.\n"
            f"{existing_keys} must match {new_keys}"
        )
        error_message_2 = (
            f"ADD (REMOVE) {unspecified_keys} TO ORIGINAL KEYS (FROM NEW KEYS)."
        )
        error_message_3 = (
            f"ADD (REMOVE) {misspecified_keys} TO NEW KEYS (FROM ORIGINAL KEYS)."
        )

        error_message = ("\n").join([error_message_1, error_message_2, error_message_3])

        assert set(self._logger_data.keys()) == set(self._df_columns), error_message

        series_data = {k: pd.Series(self._logger_data[k]) for k in self._df_columns}

        # only append header on first checkpoint/save.
        header = not os.path.exists(self._logfile_path)

        if self._index is not None:
            pd.DataFrame(series_data).set_index(self._index).to_csv(
                self._logfile_path, mode="a", header=header, index=True
            )
        else:
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
