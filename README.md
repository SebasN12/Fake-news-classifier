# Fake News Classifier

This project builds a **fake news classifier** using both **classical machine learning models** and **deep learning models** (pretrained Transformers).  
The goal is to classify news articles as *fake* or *true*, using the title and the text of the article, aswell as other columns of the dataset if needed.

---

## 📌 Features

- Classical NLP classifier (e.g., Logistic Regression)
- Deep learning classifier (BERT-base-uncased)
- Performance evaluation: accuracy, F1-score, confusion matrix, MCC
- Modular and clean code: separate scripts for preprocessing and training / evaluation
- Dataset handled locally (not included in the repository)
- Metrics and plots saved locally (not included in the repository)
- Combine datasets: script to merge fake and true news datasets into a single CSV
- Dataset adapter: allows easy integration of new datasets with different column names or label formats

---

## 🗂 Project Structure
```markdown
fake-news-classifier/
│
├── dataset/                       ← datasets (train/test CSV, NOT uploaded to GitHub)
│
├── src/                           ← source code
│   ├── classic/                   ← preprocessing and classical ML model implementation, training and evaluation
│   │   |
│   │   ├── Simple_models/         ← simple ML models (e.g. counting, Linear Regression, Logistic Regression, Naive Bayes)
│   │   │
│   │   └── SVM/
│   │
│   ├── deepL/                     ← deep learning model
│   │
│   ├── combine_dataset.py         ← combine fake and true news datasets into a single CSV
│   │
│   └── dataset_adapter.py         ← Script to adapt different datasets to a common format
│
├── metrics/                       ← evaluation OUTPUTS (plots, reports, CSVs, images, NOT uploaded to GitHub)
│
├── README.md
├── requirements.txt
└── .gitignore

```
> **Note:** `dataset/` and `metrics/` folders are excluded from GitHub using `.gitignore` to avoid uploading large files.

---

## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SebasN12/Fake-news-classifier.git
cd fake-news-classifier
```

📦 Requirements
```text
numpy
pandas
scikit-learn
matplotlib
seaborn
torch>=2.1
transformers>=4.56,<5.0
accelerate>=0.26
nltk==3.8.1
tqdm
scipy
```

You can install them all at once with:
```bash
pip install -r requirements.txt
```
⚠️ **Important:** For GPU support with deep learning models, it is strongly recommended to use Python 3.11. Newer Python versions (e.g. Python 3.12 and 3.13) may cause installation or runtime errors due to incomplete compatibility with PyTorch and Hugging Face Transformers. If you want more information, please refer to the official communities dissussions: 
- PyTorch forum discussion: https://discuss.pytorch.org/t/unable-to-install-pytorch-on-python-3-13/212112. 
- Hugging Face Transformers issue: https://github.com/huggingface/transformers/issues/35443.

To install PyTorch with GPU support, use the command corresponding to your CUDA version. For example, for CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
If you don’t have a GPU or don’t need GPU support, the regular installation from requirements.txt will work fine.

## 🚀 Usage

1. Prepare your dataset:

- Place your dataset CSV files inside the dataset/ folder.
- If your dataset is splitted into fake and true news files, you can use the combine_dataset.py script to merge them into a single CSV file. The splitted files should not be labeled before combining.
- If your dataset has different column names or label formats, use the dataset_adapter.py script to convert it to the expected format (columns: 'title', 'text', 'label' with labels as 0 for true and 1 for fake).

2.  Classic models (recommended path): 
Simple Models: run evaluateSimple.py for the Simple_models suite. It handles preprocessing internally.
If you use a different dataset, update the dataset settings in config.py.
SVM: run preprocessing_SVM.py first, then SVM.py. Update datasetPath/isOtherDataset in preprocessing_SVM.py if needed.

3. Train and evaluate the classical models:
Simple_models: evaluateSimple.py (preprocessing is built-in).
SVM: SVM.py (after preprocessing_SVM.py).

4. Train and evaluate the deep learning model: There is no manual preprocessing for the deep learning model. You must first run the data_preparation.py file (which prepares the dataset for use by BERT) and then run BERT_base.py to train and evaluate it. If you are using a different dataset, remember to change the datasetPath and the isOtherDataset variables in data_preparation.py.


📊 Results

All generated evaluation outputs, plots, and metrics are saved in the metrics/ folder.
This allows you to quickly compare classical and deep learning approaches.

⚡ Notes

GPU is recommended for training the deep learning model.

The repository does not include datasets or trained model files. You can train models from scratch using the provided scripts.

This structure is modular, so you can easily extend it with new models or preprocessing techniques.

👥 Collaborators

- @SebasN12
- @Heiligenthal