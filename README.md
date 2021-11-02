# Data Logger

This package is a lightweight data logging module built on Pandas, intended for use in machine learning experiments. 
It is also designed to complement [this plotter package](https://github.com/seblee97/plotter) and [this runner package](https://github.com/seblee97/run_modes).

## Installation

This package is written in Python 3 and requires only pandas and numpy. Installation can either be performed by cloning the repository and running ```pip install -e .``` from the package root, or via ```pip install -e git://github.com/seblee97/data_logger.git#egg=data_logger-seblee97```.

## Usage

The format of data logger used by this package is csv. 
The DataLogger class in ```data_logger.py``` takes three arguments for its constructor: the directory for the location of the csv file, the full path to the csv file itself, and a list of column names for each quantity that is being logged (e.g. loss).

There are three methods: write_scalar, which is used to log a particular scalar value for a given column (tag) and a given step (timestamp); checkpoint, which checkpoints the csv (reducing memory footprint and protecting data); and write_array_data, which separately saves array-like data via np.save.