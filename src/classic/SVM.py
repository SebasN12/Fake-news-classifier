from pathlib import Path
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 42
CV_FOLDS = 5
SUBSET_TUNING_RATIO = 0.3

# -------------------------------
# Load preprocessed matrices and classes
# -------------------------------

X_train_final = pickle.load(open('dataset/X_train_final.pkl', 'rb'))
X_test_final = pickle.load(open('dataset/X_test_final.pkl', 'rb'))
y_train = pickle.load(open('dataset/y_train.pkl', 'rb'))
y_test = pickle.load(open('dataset/y_test.pkl', 'rb'))

#directory for metrics
Path('metrics').mkdir(exist_ok=True)

# -------------------------------
# Version 1: Basic SVM training
# -------------------------------

svm_basic = SVC(random_state=RANDOM_SEED)
svm_basic.fit(X_train_final, y_train)
y_pred_basic = svm_basic.predict(X_test_final)

# Evaluation outputs

labels = ['Real', 'Fake']
with open('metrics/svm_basic_report.txt', 'w') as file:
    file.write("=== Basic SVM ===\n")
    file.write(f"Accuracy: {accuracy_score(y_test, y_pred_basic)}\n")
    file.write(f"MCC: {matthews_corrcoef(y_test, y_pred_basic)}\n\n")
    file.write(classification_report(y_test, y_pred_basic, target_names=labels, zero_division=0))

plt.figure()
cm = confusion_matrix(y_test, y_pred_basic)
cm_ax = sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', xticklabels=labels, yticklabels=labels)
cm_ax.set_xlabel('Predicted')
cm_ax.set_ylabel('Actual')
cm_ax.set_title('Confusion matrix')
plt.tight_layout()
plt.savefig('metrics\\svm_basic_confusion_matrix.png', dpi=300)
plt.close()

# -------------------------------
# Version 2: SVM with RandomizedSearchCV + 5-fold CV on a subset
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

random_search.fit(X_train_sub, y_train_sub)

best_params = random_search.best_params_
best_score_cv = random_search.best_score_

svm_final = SVC(**best_params, random_state=RANDOM_SEED)
svm_final.fit(X_train_final, y_train)
y_pred_final = svm_final.predict(X_test_final)

# Evaluation outputs

with open('metrics/svm_final_report.txt', 'w') as file:
    file.write("=== SVM with Hyperparameter Tuning ===\n")
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
cm_ax.set_title('Confusion matrix')
plt.tight_layout()
plt.savefig('metrics\\svm_final_confusion_matrix.png', dpi=300)
plt.close()