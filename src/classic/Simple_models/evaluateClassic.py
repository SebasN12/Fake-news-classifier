import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ROOT_DIR, isOtherDataset
from preprocessing import load_dataset, get_features_and_labels, load_or_create_word_counts
from baseline_models import (
    is_fake1_from_counts,
    is_fake2_from_counts,
    classify_linear_regression,
    get_linear_regression_model
)
from LogReg import get_pipeline, get_param_grid


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


def compute_metrics_dict(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="fake"),
        "Recall": recall_score(y_true, y_pred, pos_label="fake"),
        "F1": f1_score(y_true, y_pred, pos_label="fake"),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def print_fold_metrics(name, fold, y_true, y_pred):
    metrics = compute_metrics_dict(y_true, y_pred)
    print(f"\n{name} Fold {fold} Metrics:")
    print(
        f"TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}  TP={metrics['TP']}"
    )
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-Score : {metrics['F1']:.4f}")
    print(f"MCC      : {metrics['MCC']:.4f}")


def print_top_words(counter, label, top_n=10):
    print(f"Top {top_n} {label} words:")
    for word, count in counter.most_common(top_n):
        print(f"  {word}: {count}")


def print_top_logreg_features(vectorizer, clf, top_n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_.ravel()
    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_real_idx = np.argsort(coefs)[:top_n]

    print(f"Top {top_n} features for fake:")
    for idx in top_fake_idx:
        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
    print(f"Top {top_n} features for real:")
    for idx in top_real_idx:
        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")


def print_overlap_counts(fake_counts, real_counts, top_n=10):
    fake_top = [w for w, _ in fake_counts.most_common(top_n)]
    real_top = [w for w, _ in real_counts.most_common(top_n)]

    print("Top fake words in real counts:")
    for word in fake_top:
        print(f"  {word}: {real_counts[word]}")

    print("Top real words in fake counts:")
    for word in real_top:
        print(f"  {word}: {fake_counts[word]}")


def print_keyword_label_distribution(texts, labels, keywords):
    text_series = pd.Series(texts).fillna("").astype(str)
    labels = np.asarray(labels)

    print("\nKeyword label distribution:")
    for keyword in keywords:
        kw = str(keyword).strip().lower()
        if not kw:
            continue

        if kw.isalpha():
            pattern = rf"\\b{re.escape(kw)}\\b"
        else:
            pattern = re.escape(kw)

        mask = text_series.str.contains(pattern, case=False, regex=True, na=False).to_numpy()

        if not mask.any():
            print(f"  {keyword}: 0 matches")
            continue

        matched_labels = labels[mask]
        fake_count = np.sum(matched_labels == "fake")
        real_count = np.sum(matched_labels == "real")
        total = fake_count + real_count
        fake_ratio = fake_count / total if total else 0.0
        print(
            f"  {keyword}: {total} matches | fake={fake_count} real={real_count} fake_ratio={fake_ratio:.3f}"
        )


def print_subject_distribution(df, label_col="is_fake", subject_col="subject", top_n=10, min_count=50):
    if subject_col not in df.columns:
        print("\nNo subject column found.")
        return

    subject_series = df[subject_col].fillna("Unknown").astype(str)
    labels = df[label_col].astype(bool)

    counts = subject_series.value_counts()
    print(f"\nSubject distribution (top {top_n} by count):")
    for subject, count in counts.head(top_n).items():
        print(f"  {subject}: {count}")

    stats = (
        pd.DataFrame({"subject": subject_series, "is_fake": labels})
        .groupby("subject")["is_fake"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "fake_ratio"})
        .sort_values("fake_ratio", ascending=False)
    )

    stats_filtered = stats[stats["count"] >= min_count]
    print(f"\nSubjects with highest fake ratio (min_count={min_count}):")
    for subject, row in stats_filtered.head(top_n).iterrows():
        print(f"  {subject}: fake_ratio={row['fake_ratio']:.3f} (n={int(row['count'])})")

    print(f"\nSubjects with lowest fake ratio (min_count={min_count}):")
    for subject, row in stats_filtered.tail(top_n).iterrows():
        print(f"  {subject}: fake_ratio={row['fake_ratio']:.3f} (n={int(row['count'])})")

    label_counts = pd.crosstab(subject_series, labels)
    label_counts.columns = ["real_count" if c is False else "fake_count" for c in label_counts.columns]
    label_counts = label_counts.sort_values(by=label_counts.columns.tolist(), ascending=False)
    print("\nSubject label counts (top 10 by total):")
    label_counts["total"] = label_counts.sum(axis=1)
    for subject, row in label_counts.sort_values("total", ascending=False).head(top_n).iterrows():
        real_count = int(row.get("real_count", 0))
        fake_count = int(row.get("fake_count", 0))
        total = int(row["total"])
        print(f"  {subject}: real={real_count} fake={fake_count} total={total}")


def main():
    print("Using other dataset?: ", isOtherDataset)
    print("Starting evaluation of classic models...")
    df = load_dataset()
    X, y = get_features_and_labels(df)
    body_counts = load_or_create_word_counts(df)

    print("\nSamples:", len(df))
    print_subject_distribution(df, label_col="is_fake")
    print_keyword_label_distribution(
        df["body"].astype(str).values,
        y,
        keywords=["reuters", "washington", "featured", "image", "watch", "com"],
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = []

    # ==========================================================================
    # COUNTING MODELS
    # ==========================================================================
    print("\n=== COUNTING MODELS ===")

    print("\n-> Evaluating is_fake1")
    y_true_1, y_pred_1 = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}")
        print(f"Train size: {len(train_idx)} | Test size: {len(val_idx)}")
        fake_counts = Counter()
        real_counts = Counter()
        for i in train_idx:
            (fake_counts if y[i] == "fake" else real_counts).update(body_counts[i])

        top_fake = fake_counts.most_common(1)[0][0]
        top_real = real_counts.most_common(1)[0][0]
        print_top_words(fake_counts, "fake", top_n=10)
        print_top_words(real_counts, "real", top_n=10)
        print_overlap_counts(fake_counts, real_counts, top_n=10)

        for i in val_idx:
            y_true_1.append(y[i])
            y_pred_1.append(is_fake1_from_counts(body_counts[i], top_fake, top_real))

        print_fold_metrics("is_fake1", fold, y[val_idx], y_pred_1[-len(val_idx):])

    results.append(compute_metrics("is_fake1", np.array(y_true_1), np.array(y_pred_1)))

    print("\n-> Evaluating is_fake2")
    y_true_2, y_pred_2 = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}")
        print(f"Train size: {len(train_idx)} | Test size: {len(val_idx)}")

        fake_train_counts = Counter()
        real_train_counts = Counter()
        for i in train_idx:
            (fake_train_counts if y[i] == "fake" else real_train_counts).update(body_counts[i])
        print_top_words(fake_train_counts, "fake", top_n=10)
        print_top_words(real_train_counts, "real", top_n=10)
        print_overlap_counts(fake_train_counts, real_train_counts, top_n=10)

        for i in val_idx:
            y_true_2.append(y[i])
            y_pred_2.append(is_fake2_from_counts(
                body_counts[i], fake_train_counts, real_train_counts
            ))

        print_fold_metrics("is_fake2", fold, y[val_idx], y_pred_2[-len(val_idx):])

    results.append(compute_metrics("is_fake2", np.array(y_true_2), np.array(y_pred_2)))


    # ==========================================================================
    # LOGISTIC REGRESSION TF-IDF GRIDSEARCH
    # ==========================================================================
    print("\n=== LOGISTIC REGRESSION ===")

    pipeline = get_pipeline()
    param_grid = get_param_grid()

    y_true_lr, y_pred_lr = [], []
    best_params = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}")
        print(f"Train size: {len(train_idx)} | Test size: {len(val_idx)}")

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring="accuracy",
            n_jobs=1,
            verbose=0,
        )
        grid.fit(X[train_idx], y[train_idx])
        best_params.append(grid.best_params_)

        best_estimator = grid.best_estimator_
        y_pred_fold = best_estimator.predict(X[val_idx])
        y_true_lr.extend(y[val_idx])
        y_pred_lr.extend(y_pred_fold)

        tfidf = best_estimator.named_steps["tfidf"]
        clf = best_estimator.named_steps["clf"]
        print_top_logreg_features(tfidf, clf, top_n=10)
        print_fold_metrics("Logistic Regression", fold, y[val_idx], y_pred_fold)

    print("Best params (last fold):", best_params[-1])
    results.append(compute_metrics("Logistic Regression", np.array(y_true_lr), np.array(y_pred_lr)))

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
    metrics_dir = os.path.join(ROOT_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_name = "classic_model_metrics_other.csv" if isOtherDataset else "classic_model_metrics.csv"
    out = os.path.join(metrics_dir, metrics_name)
    results_df.to_csv(out, index=False)

    print("\nResults saved in:", out)


if __name__ == "__main__":
    main()
