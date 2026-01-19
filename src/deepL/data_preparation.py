import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# Change these variables if using other dataset
isOtherDataset = False
datasetPath = 'dataset/all_news.csv'


def run_data_preparation():
    df = pd.read_csv(datasetPath)

    df["text"] = (
        df["title"].fillna("") + " " +
        df["text"].fillna("") + " [SEP] " +
        df["subject"].fillna("unknown")
    )

    y = df["is_fake"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    train_df = pd.DataFrame({
        "text": X_train,
        "label": y_train
    })

    test_df = pd.DataFrame({
        "text": X_test,
        "label": y_test
    })

    train_df.to_csv("dataset/train_DL.csv", index=False)
    test_df.to_csv("dataset/test_DL.csv", index=False)
    print("Data preparation completed. Train and test CSV files created.")

if __name__ == "__main__":
    run_data_preparation()