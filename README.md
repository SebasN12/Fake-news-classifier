# Fake News Classifier

This project builds a **fake news classifier** using both **classical machine learning models** and **deep learning models** (pretrained Transformers).  
The goal is to classify news articles as *fake* or *true*, using the title and the text of the article, aswell as other columns of the dataset if needed.

---

## 📌 Features

- Classical NLP classifier (e.g., Logistic Regression)
- Deep learning classifier (pretrained Transformer, TBD)
- Performance evaluation: accuracy, F1-score, confusion matrix, MCC
- Modular and clean code: separate scripts for preprocessing and training / evaluation
- Dataset handled locally (not included in the repository)
- Metrics and plots saved locally (not included in the repository)

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
│   │   ├── Simple_models/         ← simple ML models (e.g. counting, Linear Regression, Logistic Regression)
│   │   │
│   │   └── SVM/
│   │
│   └── deepL/                     ← deep learning model
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
torch
transformers
nltk
```

You can install them all at once with:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Place your dataset CSV files inside the dataset/ folder.

2. Preprocess the text: For Simple_models run the preprocess.py script. For SVM run the preprocessing_SVM.py script.

3. Train and evaluate the classical model: Go to the respective model folder and run the training or main script. For Simple_models run the evaluateClassic.py script. For SVM run the SVM.py script.

4. Train and evaluate the deep learning model: There is no manual preprocessing for the deep learning model. You must first run the data_preparation.py file (which prepares the dataset for use by BERT) and then run BERT_base.py to train it.


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