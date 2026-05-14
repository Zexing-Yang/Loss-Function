Explanation of each Python subfile

datadeal.py Used to generate rainfall events

dataset.py Generate training/validation and test sets

config.py Place the parameters that need to be adjusted here as much as possible for future use

plotresults.py Plot charts, the function to analyze results is placed here

train.py Training and validation are placed here

test.py Test are placed here

data_divided A set of test data is provided as a reference for the data format. It should be noted that the test data consists of randomly generated values as the original data of the study cannot be shared.

run.py This project implements a batch automation script that systematically executes training and testing pipelines across multiple basins and time parameters, managing isolated environments and logging for hydrological experiments.