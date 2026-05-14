The Python scripts and their respective functions are explained as follows:

config.py: This file serves as a centralized location for hyperparameters that require tuning, designed for ease of use in future applications.

datadeal.py: This module provides utility functions for constructing catchment-specific datasets and PyTorch DataLoaders, ensuring consistent data normalization across training, validation, and test sets.

dataloader.py: This module defines the core CamelsTXTdataset class for loading, preprocessing, and normalizing CAMELS text data, handling unit conversion, sequence generation.

train.py: This script serves as the training and validation pipeline for optimizing the deep learning model.

eval.py: This script serves as the evaluation framework for assessing the model's final predictive performance.

main.py: This script implements a regional modeling workflow that trains a single LSTM model per hydrological region, evaluates its performance across multiple basins using NSE/KGE/TPE metrics.






