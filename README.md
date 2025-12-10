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
fake-news-classifier/
│
├── dataset/                       ← datasets (train/test CSV, NOT uploaded to GitHub)
│
├── src/                           ← source code
│   ├── classic/                   ← classical ML model implementation, training and evaluation
│   ├── deepL/                     ← deep learning model implementation, training and evaluation
│   │
│   └── preprocessing.py           ← text cleaning and preprocessing 
│
├── metrics/                       ← evaluation OUTPUTS (plots, reports, CSVs, images)
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

2. Preprocess the text:

```bash
python src/preprocessing.py
```

3. Train and evaluate the classical model:

```bash
python src/classic/train_classic.py
```

4. Train and evaluate the deep learning model:
```bash
python src/deepL/train_deep.py
```


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