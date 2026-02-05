from pathlib import Path
import pickle
from preprocessing_SVM import isOtherDataset
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 42
CV_FOLDS = 5
SUBSET_TUNING_RATIO = 0.3

print('Using other dataset?: ', isOtherDataset)

# --------------------------------
# Evaluation output file paths
# --------------------------------

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

# Paths

basicReportPath = METRICS_DIR / ("svm_basic_report_other.txt" if isOtherDataset else "svm_basic_report.txt")
basicConfusionMatrixPath = METRICS_DIR / ("svm_basic_confusion_matrix_other.png" if isOtherDataset else "svm_basic_confusion_matrix.png")
finalReportPath = METRICS_DIR / ("svm_final_report_other.txt" if isOtherDataset else "svm_final_report.txt")
finalConfusionMatrixPath = METRICS_DIR / ("svm_final_confusion_matrix_other.png" if isOtherDataset else "svm_final_confusion_matrix.png")

# output titles

basicReportTitle = "=== Basic SVM (other dataset) ===\n" if isOtherDataset else "=== Basic SVM ===\n"
basicConfusionMatrixTitle = 'Basic SVM Confusion matrix (other dataset)' if isOtherDataset else 'Basic SVM Confusion matrix'
finalReportTitle = '=== SVM with Hyperparameter tuning (other dataset) ===\n' if isOtherDataset else '=== SVM with Hyperparameter tuning ===\n'
finalConfusionMatrixTitle = 'SVM with Hyperparameter tuning Confusion matrix (other dataset)' if isOtherDataset else 'SVM with Hyperparameter tuning Confusion matrix'


# -------------------------------
# Load preprocessed matrices and classes
# -------------------------------

print("Loading preprocessed data...")

DATASET_DIR = Path("dataset")

X_train_final = pickle.load(open(DATASET_DIR / "X_train_final.pkl", 'rb'))
X_test_final = pickle.load(open(DATASET_DIR / "X_test_final.pkl", 'rb'))
y_train = pickle.load(open(DATASET_DIR / "y_train.pkl", 'rb'))
y_test = pickle.load(open(DATASET_DIR / "y_test.pkl", 'rb'))

# -------------------------------
# Version 1: Basic SVM training
# -------------------------------

print("Training basic SVM...")

svm_basic = SVC(random_state=RANDOM_SEED)
svm_basic.fit(X_train_final, y_train)
y_pred_basic = svm_basic.predict(X_test_final)

# Evaluation outputs

labels = ['Real', 'Fake']
with open(basicReportPath, 'w') as file:
    file.write(basicReportTitle)
    file.write(f"Accuracy: {accuracy_score(y_test, y_pred_basic)}\n")
    file.write(f"MCC: {matthews_corrcoef(y_test, y_pred_basic)}\n\n")
    file.write(classification_report(y_test, y_pred_basic, target_names=labels, zero_division=0))

plt.figure()
cm = confusion_matrix(y_test, y_pred_basic)
cm_ax = sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', xticklabels=labels, yticklabels=labels)
cm_ax.set_xlabel('Predicted')
cm_ax.set_ylabel('Actual')
cm_ax.set_title(basicConfusionMatrixTitle)
plt.tight_layout()
plt.savefig(basicConfusionMatrixPath, dpi=300)
plt.close()

print("Basic SVM evaluation metrics saved.")

# -------------------------------
# Version 2: SVM with RandomizedSearchCV + 5-fold CV on a subset (for optimization reasons)
# -------------------------------

# Create subset for tuning
X_train_sub, _, y_train_sub, _ = train_test_split(
    X_train_final, y_train, test_size=1-SUBSET_TUNING_RATIO,
    random_state=RANDOM_SEED, stratify=y_train
)

param_dist = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(
    estimator=SVC(random_state=RANDOM_SEED),
    param_distributions=param_dist,
    n_iter=5,
    cv=CV_FOLDS,
    scoring='accuracy',
    random_state=RANDOM_SEED,
    n_jobs=-1
)

print("Performing hyperparameter tuning with RandomizedSearchCV...")

random_search.fit(X_train_sub, y_train_sub)

best_params = random_search.best_params_
best_score_cv = random_search.best_score_

print(f"Best hyperparameters found: {best_params}")
print(f"Best CV accuracy on subset: {best_score_cv:.4f}")

print("Training final SVM with best hyperparameters...")
svm_final = SVC(**best_params, random_state=RANDOM_SEED)
svm_final.fit(X_train_final, y_train)
y_pred_final = svm_final.predict(X_test_final)

# Evaluation outputs

with open(finalReportPath, 'w') as file:
    file.write(finalReportTitle)
    file.write(f"Best CV Accuracy (subset): {best_score_cv:.4f}\n")
    file.write(f"Best Hyperparameters: {best_params}\n\n")
    file.write(f"Test Accuracy: {accuracy_score(y_test, y_pred_final):.4f}\n")
    file.write(f"Test MCC: {matthews_corrcoef(y_test, y_pred_final):.4f}\n\n")
    file.write(classification_report(y_test, y_pred_final, target_names=labels, zero_division=0))

plt.figure()
cm = confusion_matrix(y_test, y_pred_final)
cm_ax = sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', xticklabels=labels, yticklabels=labels)
cm_ax.set_xlabel('Predicted')
cm_ax.set_ylabel('Actual')
cm_ax.set_title(finalConfusionMatrixTitle)
plt.tight_layout()
plt.savefig(finalConfusionMatrixPath, dpi=300)
plt.close()

print("Final SVM evaluation metrics saved.")