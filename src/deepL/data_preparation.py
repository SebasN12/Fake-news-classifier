import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# Change these variables if using other dataset
isOtherDataset = False
datasetPath = 'dataset/all_news.csv'
datasetPath_test = 'dataset/all_news.csv'   # only if other dataset is used for testing


def run_data_preparation():
    df = pd.read_csv(datasetPath)

    df["text"] = (
        df["title"].fillna("") + " " +
        df["text"].fillna("")
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

def run_data_preparation_more_df():
    
    df = pd.read_csv(datasetPath)
    df_test = pd.read_csv(datasetPath_test)

    df["text"] = (
        df["title"].fillna("") + " " +
        df["text"].fillna("")
    )

    df_test["text"] = (
        df_test["title"].fillna("") + " " +
        df_test["text"].fillna("")
    )

    df_test = df_test.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    x_test = df_test["text"]
    y_test = df_test["is_fake"]

    x = df["text"]
    y = df["is_fake"]

    train_df = pd.DataFrame({
        "text": x,
        "label": y
    })

    test_df = pd.DataFrame({
        "text": x_test,
        "label": y_test
    })

    train_df.to_csv("dataset/train_DL.csv", index=False)
    test_df.to_csv("dataset/test_DL.csv", index=False)
    print("Data preparation completed. Train and test CSV files created.")


if __name__ == "__main__":
    # change this if using other dataset for test. 
    use_other_dataset_test = False

    if use_other_dataset_test:
        print("Preparing train/test with two separate datasets...")
        run_data_preparation_more_df()
    else:
        print("Preparing train/test split from the same dataset...")
        run_data_preparation()