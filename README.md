# Fake News Classifier

This project builds a **fake news classifier** using both **classical machine learning models** and **deep learning models** (pretrained Transformers).  
The goal is to classify news articles as *fake* or *true*, using both the title and the text of the article.

---

## 📌 Features

- Classical NLP classifier (e.g., Logistic Regression)
- Deep learning classifier (pretrained Transformer, TBD)
- Performance evaluation: accuracy, F1-score, confusion matrix, MCC
- Modular and clean code: separate scripts for preprocessing, training, and evaluation
- Dataset handled locally (not included in the repository)

---

## 🗂 Project Structure
```markdown
```text
fake-news-classifier/
│
├── dataset/ ← datasets (train/test CSV, NOT uploaded to GitHub)
│
├── src/ ← source code
│ ├── preprocessing.py ← text cleaning and preprocessing
│ ├── classic_model.py ← classical ML model implementation
│ ├── deep_model.py ← deep learning model implementation
│ ├── train_classic.py ← script to train classical model
│ ├── train_deep.py ← script to train deep learning model
│ └── evaluate.py ← evaluation functions (metrics, confusion matrix)
│
├── metrics/ ← evaluation outputs (plots, reports)
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
pip install -r requirements.txt
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
```

You can install them all at once with:
```bash
pip install -r requirements.txt
```

🚀 Usage

Place your dataset CSV files inside the dataset/ folder.

Preprocess the text:

python src/preprocessing.py


Train the classical model:

python src/train_classic.py


Train the deep learning model:

python src/train_deep.py


Evaluate the models:

python src/evaluate.py



📊 Results

All generated evaluation outputs, plots, and metrics are saved in the metrics/ folder.
This allows you to quickly compare classical and deep learning approaches.

⚡ Notes

GPU is recommended for training the deep learning model.

The repository does not include datasets or trained model files. You can train models from scratch using the provided scripts.

This structure is modular, so you can easily extend it with new models or preprocessing techniques.

👥 Collaborators

- @SebasN12
-