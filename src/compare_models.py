import argparse
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from classic.classic_model import LinearRegressionClassifier
from config import ROOT_DIR
from preprocessing import load_dataset, get_features_and_labels, sample_dataset

RANDOM_SEED = 42


@dataclass
class ModelConfig:
    name: str
    estimator: object
    param_grid: Optional[Dict[str, List[object]]] = None


def compute_metrics(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"])
    tn, fp, fn, tp = cm.ravel()

    return {
        "Classifier": name,
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


def summarize_params(params_list):
    if not params_list:
        return ""
    items = [tuple(sorted(p.items())) for p in params_list]
    best = Counter(items).most_common(1)[0][0]
    return "; ".join([f"{k}={v}" for k, v in best])


def evaluate_model_cv(config, X, y, cv, inner_folds, scorer):
    y_true_all = []
    y_pred_all = []
    best_params = []

    print(f"\n=== {config.name} ===")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"Fold {fold}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        estimator = clone(config.estimator)
        if config.param_grid:
            search = GridSearchCV(
                estimator,
                config.param_grid,
                cv=inner_folds,
                scoring=scorer,
                n_jobs=1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best_params.append(search.best_params_)
            model = search.best_estimator_
        else:
            model = estimator.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

    metrics = compute_metrics(config.name, np.array(y_true_all), np.array(y_pred_all))
    metrics["BestParams"] = summarize_params(best_params)
    return metrics


def build_model_configs(random_seed):
    def tfidf(ngram_range=(1, 2), max_features=30000, min_df=2, max_df=0.95):
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
        )

    def count_vec(ngram_range=(1, 1), max_features=30000, min_df=2, max_df=0.95, binary=False):
        return CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            binary=binary,
        )

    configs = []

    configs.append(ModelConfig(
        name="Linear Regression (threshold)",
        estimator=Pipeline([
            ("tfidf", tfidf(ngram_range=(1, 1))),
            ("clf", LinearRegressionClassifier(threshold=0.5)),
        ]),
        param_grid={"clf__threshold": [0.4, 0.5, 0.6]},
    ))

    configs.append(ModelConfig(
        name="Logistic Regression",
        estimator=Pipeline([
            ("tfidf", tfidf(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_seed)),
        ]),
        param_grid={
            "clf__C": [0.1, 1.0, 10.0],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        },
    ))

    configs.append(ModelConfig(
        name="Softmax Regression",
        estimator=Pipeline([
            ("tfidf", tfidf(ngram_range=(1, 2))),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=random_seed,
                multi_class="multinomial",
                solver="lbfgs",
            )),
        ]),
        param_grid={
            "clf__C": [0.1, 1.0],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        },
    ))

    configs.append(ModelConfig(
        name="Multinomial Naive Bayes",
        estimator=Pipeline([
            ("tfidf", tfidf(ngram_range=(1, 2))),
            ("clf", MultinomialNB()),
        ]),
        param_grid={
            "clf__alpha": [0.5, 1.0],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        },
    ))

    configs.append(ModelConfig(
        name="Bernoulli Naive Bayes",
        estimator=Pipeline([
            ("count", count_vec(binary=True)),
            ("clf", BernoulliNB()),
        ]),
        param_grid={
            "clf__alpha": [0.5, 1.0],
        },
    ))

    configs.append(ModelConfig(
        name="Linear SVM",
        estimator=Pipeline([
            ("tfidf", tfidf(ngram_range=(1, 2))),
            ("clf", LinearSVC(random_state=random_seed)),
        ]),
        param_grid={
            "clf__C": [0.1, 1.0, 2.0],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        },
    ))

    configs.append(ModelConfig(
        name="Polynomial SVM",
        estimator=Pipeline([
            ("tfidf", tfidf(ngram_range=(1, 2))),
            ("svd", TruncatedSVD(n_components=300, random_state=random_seed)),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="poly")),
        ]),
        param_grid={
            "svd__n_components": [200, 300],
            "clf__C": [1.0, 10.0],
            "clf__degree": [2, 3],
            "clf__gamma": ["scale"],
        },
    ))

    configs.append(ModelConfig(
        name="RBF SVM",
        estimator=Pipeline([
            ("tfidf", tfidf(ngram_range=(1, 2))),
            ("svd", TruncatedSVD(n_components=300, random_state=random_seed)),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf")),
        ]),
        param_grid={
            "svd__n_components": [200, 300],
            "clf__C": [1.0, 10.0],
            "clf__gamma": ["scale", "auto"],
        },
    ))

    configs.append(ModelConfig(
        name="MLP",
        estimator=Pipeline([
            ("tfidf", tfidf(ngram_range=(1, 2))),
            ("svd", TruncatedSVD(n_components=200, random_state=random_seed)),
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128,),
                max_iter=30,
                early_stopping=True,
                random_state=random_seed,
            )),
        ]),
        param_grid={
            "svd__n_components": [100, 200],
            "clf__hidden_layer_sizes": [(128,), (256,)],
            "clf__alpha": [1e-4, 1e-3],
        },
    ))

    return configs


def run_comparison(
    include_deep=True,
    folds=5,
    inner_folds=3,
    random_seed=RANDOM_SEED,
    max_samples=None,
    balance_classes=False,
    deep_model_name="bert-base-uncased",
    deep_epochs=3,
    deep_batch_size=8,
    deep_max_length=256,
    deep_freeze_layers=0,
    deep_learning_rate=2e-5,
    output_path=None,
):
    df = load_dataset()
    X, y = get_features_and_labels(df)

    X = np.array(X)
    y = np.array(y)

    X, y = sample_dataset(
        X,
        y,
        max_samples=max_samples,
        random_seed=random_seed,
        balance_classes=balance_classes,
    )

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    scorer = make_scorer(f1_score, pos_label="fake")

    results = []
    for config in build_model_configs(random_seed):
        results.append(evaluate_model_cv(config, X, y, cv, inner_folds, scorer))

    if include_deep:
        from deepL.deep_model import run_transformer_kfold

        safe_name = deep_model_name.replace("/", "_")
        deep_out = os.path.join(ROOT_DIR, "metrics", "deep", safe_name)
        deep_metrics = run_transformer_kfold(
            texts=X,
            labels=y,
            model_name=deep_model_name,
            folds=folds,
            max_length=deep_max_length,
            epochs=deep_epochs,
            batch_size=deep_batch_size,
            freeze_layers=deep_freeze_layers,
            learning_rate=deep_learning_rate,
            random_seed=random_seed,
            output_dir=deep_out,
        )
        results.append(deep_metrics)

    results_df = pd.DataFrame(results)
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Compare classic and deep models with 5-fold CV.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-samples", type=float, default=None)
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--deep-model", type=str, default="bert-base-uncased")
    parser.add_argument("--deep-epochs", type=int, default=3)
    parser.add_argument("--deep-batch-size", type=int, default=8)
    parser.add_argument("--deep-max-length", type=int, default=256)
    parser.add_argument("--deep-freeze-layers", type=int, default=0)
    parser.add_argument("--deep-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT_DIR, "model_comparison.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_comparison(
        include_deep=not args.skip_deep,
        folds=args.folds,
        inner_folds=args.inner_folds,
        random_seed=args.random_seed,
        max_samples=args.max_samples,
        balance_classes=args.balance_classes,
        deep_model_name=args.deep_model,
        deep_epochs=args.deep_epochs,
        deep_batch_size=args.deep_batch_size,
        deep_max_length=args.deep_max_length,
        deep_freeze_layers=args.deep_freeze_layers,
        deep_learning_rate=args.deep_learning_rate,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
