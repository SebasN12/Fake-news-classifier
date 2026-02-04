import os

# Dataset selection flags for local experiments.
isOtherDataset = False
datasetName = "all_news.csv"

# Project root and dataset paths resolved relative to this file.
ROOT_DIR = os.path.abspath(__file__ + "/../../../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DATA_PATH = os.path.join(DATASET_DIR, datasetName)
