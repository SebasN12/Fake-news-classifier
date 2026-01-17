import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


LABEL_MAP = {"real": 0, "fake": 1}


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def labels_to_ids(labels):
    labels = np.asarray(labels)
    if labels.dtype.kind in {"U", "S", "O"}:
        return np.where(labels == "fake", 1, 0).astype(int)
    return labels.astype(int)


def ids_to_labels(label_ids):
    label_ids = np.asarray(label_ids)
    return np.where(label_ids == 1, "fake", "real")


def compute_metrics_from_ids(y_true_ids, y_pred_ids):
    y_true = ids_to_labels(y_true_ids)
    y_pred = ids_to_labels(y_pred_ids)

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


def freeze_encoder_layers(model, num_layers):
    if num_layers <= 0:
        return

    base_model = getattr(model, model.base_model_prefix, None)
    if base_model is None:
        base_model = getattr(model, "base_model", model)

    layers = None
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
        layers = base_model.encoder.layer
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "layer"):
        layers = base_model.transformer.layer
    elif hasattr(base_model, "layers"):
        layers = base_model.layers

    if layers is None:
        return

    for layer in layers[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False


def run_transformer_kfold(
    texts,
    labels,
    model_name="bert-base-uncased",
    folds=5,
    max_length=256,
    epochs=3,
    batch_size=8,
    freeze_layers=0,
    learning_rate=2e-5,
    random_seed=42,
    output_dir=None,
):
    set_seed(random_seed)

    texts = np.asarray(texts)
    labels = np.asarray(labels)
    label_ids = labels_to_ids(labels)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    y_true_all = []
    y_pred_all = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    safe_name = model_name.replace("/", "_")
    output_root = output_dir or os.path.join(os.getcwd(), "metrics", "deep", safe_name)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, label_ids), start=1):
        fold_dir = os.path.join(output_root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        freeze_encoder_layers(model, freeze_layers)

        train_ds = NewsDataset(texts[train_idx], label_ids[train_idx], tokenizer, max_length)
        val_ds = NewsDataset(texts[val_idx], label_ids[val_idx], tokenizer, max_length)

        args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="no",
            save_strategy="no",
            logging_steps=50,
            report_to=[],
            seed=random_seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
        )

        trainer.train()
        preds = trainer.predict(val_ds).predictions
        y_pred = np.argmax(preds, axis=1)

        y_true_all.extend(label_ids[val_idx])
        y_pred_all.extend(y_pred)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = compute_metrics_from_ids(y_true_all, y_pred_all)
    metrics["Classifier"] = f"Transformer ({model_name})"
    return metrics
