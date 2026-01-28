import pandas as pd
import torch
import random
import numpy as np
from data_preparation import isOtherDataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, set_seed
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

#AI was used here for BERT coding. It was used for suggestions, workflow design, and resolving errors and incompatibilities.

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# --------------------
# Config
# --------------------

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 256
LR = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reportPath = "metrics/bert_base_report_other.txt" if isOtherDataset else "metrics/bert_base_report.txt"
reportTitle = "=== BERT-base fine-tuned (other dataset) ===\n\n" if isOtherDataset else "=== BERT-base fine-tuned ===\n\n"
confusionMatrixPath = "metrics/bert_base_confusion_matrix_other.png" if isOtherDataset else "metrics/bert_base_confusion_matrix.png"
confusionMatrixTitle = "BERT-base confusion matrix (other dataset)" if isOtherDataset else "BERT-base confusion matrix"
# --------------------

print("Using device:", DEVICE)
print("Using other dataset?:", isOtherDataset)


train_df = pd.read_csv("dataset/train_DL.csv")
test_df = pd.read_csv("dataset/test_DL.csv")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = NewsDataset(
    train_df["text"].tolist(),
    train_df["label"].tolist()
)

test_dataset = NewsDataset(
    test_df["text"].tolist(),
    test_df["label"].tolist()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(RANDOM_SEED + worker_id))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --------------------
# BERT Model
# --------------------

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# --------------------
# Training
# --------------------

for epoch in range(EPOCHS):
    model.train()
    losses = []

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} loss: {sum(losses)/len(losses):.4f}")

# --------------------
# Evaluation
# --------------------

print("Starting evaluation on the test set... This may take a few minutes.")

model.eval()
preds = []
true = []

labels = ["real", "fake"]

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true.extend(labels_batch.cpu().numpy())

y_test = true
y_pred_final = preds

os.makedirs("metrics", exist_ok=True)

with open(reportPath, "w") as file:
    file.write(reportTitle)
    file.write(f"Test Accuracy: {accuracy_score(y_test, y_pred_final):.4f}\n")
    file.write(f"Test MCC: {matthews_corrcoef(y_test, y_pred_final):.4f}\n\n")
    file.write(
        classification_report(
            y_test,
            y_pred_final,
            target_names=labels,
            zero_division=0
        )
    )

# Confusion matrix
plt.figure()
cm = confusion_matrix(y_test, y_pred_final)

ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="rocket",
    xticklabels=labels,
    yticklabels=labels
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(confusionMatrixTitle)

plt.tight_layout()
plt.savefig(confusionMatrixPath, dpi=300)
plt.close()