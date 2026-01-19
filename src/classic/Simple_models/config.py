import os

# Chaange these variables if using other dataset
isOtherDataset = False
datasetName = "all_news.csv"

ROOT_DIR = os.path.abspath(__file__ + "/../../../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DATA_PATH = os.path.join(DATASET_DIR, datasetName)