import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DATA_PATH = os.path.join(DATASET_DIR, "all_news.csv")
