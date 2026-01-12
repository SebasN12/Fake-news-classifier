import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ROOT_DIR
from preprocessing import load_dataset, get_features_and_labels, load_or_create_word_counts
from classic.classic_model import (
    is_fake1_from_counts,
    is_fake2_from_counts,
    classify_linear_regression,
    get_linear_regression_model
)
from classic.LogReg import get_pipeline, get_param_grid


RANDOM_SEED = 42


def compute_metrics(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{name} Metrics:")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='fake'):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, pos_label='fake'):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred, pos_label='fake'):.4f}")
    print(f"MCC      : {matthews_corrcoef(y_true, y_pred):.4f}")

    return {
        "Classifier": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="fake"),
        "Recall": recall_score(y_true, y_pred, pos_label="fake"),
        "F1": f1_score(y_true, y_pred, pos_label="fake"),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }


def main():
    df = load_dataset()
    X, y = get_features_and_labels(df)
    body_counts = load_or_create_word_counts(df)

    print("\nSamples:", len(df))

    fake_all = Counter()
    real_all = Counter()
    for c, label in zip(body_counts, y):
        (fake_all if label == "fake" else real_all).update(c)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = []

    # ==========================================================================
    # COUNTING MODELS
    # ==========================================================================
    print("\n=== COUNTING MODELS ===")

    print("\n-> Evaluating is_fake1")
    y_true_1, y_pred_1 = [], []

    for fold, (_, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}")
        top_fake = fake_all.most_common(1)[0][0]
        top_real = real_all.most_common(1)[0][0]

        for i in val_idx:
            y_true_1.append(y[i])
            y_pred_1.append(is_fake1_from_counts(body_counts[i], top_fake, top_real))

    results.append(compute_metrics("is_fake1", np.array(y_true_1), np.array(y_pred_1)))

    print("\n-> Evaluating is_fake2")
    y_true_2, y_pred_2 = [], []

    for fold, (_, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}")
        
        fake_train_counts = fake_all
        real_train_counts = real_all

        for i in val_idx:
            y_true_2.append(y[i])
            y_pred_2.append(is_fake2_from_counts(
                body_counts[i], fake_train_counts, real_train_counts
            ))

    results.append(compute_metrics("is_fake2", np.array(y_true_2), np.array(y_pred_2)))


    # ==========================================================================
    # LOGISTIC REGRESSION TF-IDF GRIDSEARCH
    # ==========================================================================
    print("\n=== LOGISTIC REGRESSION ===")

    pipeline = get_pipeline()
    param_grid = get_param_grid()

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=skf,
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )
    grid.fit(X, y)

    print("Best params:", grid.best_params_)
    y_pred_lr = grid.best_estimator_.predict(X)
    results.append(compute_metrics("Logistic Regression", y, y_pred_lr))

    # ==========================================================================
    # LINEAR REGRESSION BASELINE
    # ==========================================================================
    print("\n=== LINEAR REGRESSION BASELINE ===")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        lowercase=True
    )
    y_bool = (y == "fake").astype(int)

    y_true_lr, y_pred_lr = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bool), start=1):
        print(f"Fold {fold}")

        X_train = vectorizer.fit_transform(X[train_idx])
        X_val = vectorizer.transform(X[val_idx])

        model = get_linear_regression_model()
        model.fit(X_train, y_bool[train_idx])

        scores = model.predict(X_val)
        preds = classify_linear_regression(scores)

        y_true_lr.extend(y_bool[val_idx])
        y_pred_lr.extend(preds)

    y_true_s = np.where(np.array(y_true_lr) == 1, "fake", "real")
    y_pred_s = np.where(np.array(y_pred_lr) == 1, "fake", "real")
    results.append(compute_metrics("Linear Regression", y_true_s, y_pred_s))

    # ==========================================================================
    # Final Summary as CSV
    # ==========================================================================
    results_df = pd.DataFrame(results)
    out = os.path.join(ROOT_DIR, "classic_model_metrics.csv")
    results_df.to_csv(out, index=False)

    print("\nResults saved in:", out)


if __name__ == "__main__":
    main()
