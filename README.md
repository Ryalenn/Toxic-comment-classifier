# Toxic Comment Classifier (SpaCy + scikit-learn MLP)

This repository contains two Python scripts:

- `Train_model.py` – trains a multilabel classifier to detect toxic attributes in text and saves the model to disk.
- `Use_model.py` – loads the saved model and predicts the toxic classes for a new sentence.

The pipeline uses **SpaCy** (`en_core_web_md`) word vectors to build a fixed-size sentence representation (sum of token vectors) and a **scikit-learn `MLPClassifier`** for multilabel classification.

---

## 1) Environment & Dependencies

- Python 3.9+ (3.10/3.11 also fine)
- Packages:
  - `spacy`
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `pickle` (standard library)

Install and download the SpaCy model:

```bash
pip install -U spacy scikit-learn numpy pandas
python -m spacy download en_core_web_md
```

> The code disables SpaCy’s parser/NER/tok2vec to keep tokenization lightweight, but **requires the vectors** from `en_core_web_md` (300-dim).

---

## 2) Data Format

`Train_model.py` expects a CSV file named `train.csv` with the following columns:

- `comment_text` (string): the raw text to classify  
- Six binary label columns (0/1):
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`

A seventh “helper” label **`not toxic`** is created during preprocessing: it is `1` when **all six labels are `0`**, otherwise `0`.  
This makes the problem **7-output multilabel** (not mutually exclusive).

Minimal example row:

| comment_text           | toxic | severe_toxic | obscene | threat | insult | identity_hate |
|------------------------|:-----:|:------------:|:-------:|:------:|:------:|:-------------:|
| "You are awful."       |  1    |      0       |    0    |   0    |   1    |       0       |

Place `train.csv` next to the scripts.

---

## 3) Feature Engineering (what the scripts do)

Both scripts share the same `preprocessing(text)` function:

1. Tokenize with SpaCy (`en_core_web_md`).
2. For each token:
   - keep it if it is **not a stopword**, **alphabetic**, and **length > 2**;
   - use its **lemma** to retrieve the **300-dim vector** from `nlp.vocab`.
3. **Sum** all kept token vectors → one 300-dim sentence vector (`numpy` array).

> This “bag-of-vectors sum” is simple, fast, and robust to variable length.  
> Note: empty texts (or texts with only filtered tokens) become a zero vector.

---

## 4) Training (`Train_model.py`)

Main steps:

1. Load `train.csv` into a DataFrame.
2. Build multilabel targets `y_train` by appending the derived **`not toxic`** column.
3. Preprocess every `comment_text` to a 300-dim vector (`train["text_preprocessed"]`).
4. **Random split** into train/test with a custom function (`threshold=0.8` for ~80/20).
5. Define an `MLPClassifier`.  
   - The script includes a **GridSearchCV** with a hyperparameter grid, **commented out** to save time.
   - By default it uses:
     ```python
     MLPClassifier(
       activation='tanh',
       alpha=0.05,
       hidden_layer_sizes=(100, 100, 100),
       learning_rate='constant',
       solver='adam'
     )
     ```
6. Fit the model, evaluate on the test split (`classification_report`), and **save** it to `NLP_BEST_Model.pkl` with `pickle`.

Run:

```bash
python Train_model.py
```

Outputs:

- Logs during preprocessing (`index 1000`, `index 2000`, …)
- Train/test lengths
- Scikit-learn `classification_report` on the held-out split
- Saved model file: **`NLP_BEST_Model.pkl`**

---

## 5) Inference (`Use_model.py`)

`Use_model.py` loads the pickled model and runs a prediction on a sentence:

1. Load the same SpaCy model and preprocessing function (must match training).
2. Load `NLP_BEST_Model.pkl`.
3. Preprocess the input `sentence` (currently hard-coded).
4. Predict with `loaded_model.predict([preprocessedSentence])`.
5. Print all classes whose predicted value is `1`.

Default class order (7 outputs):

```text
["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "not toxic"]
```

Run:

```bash
python Use_model.py
```

To classify your own text, edit the line:

```python
sentence = "your text here"
```

---

## 6) Reproducibility Notes

- **SpaCy model** must be `en_core_web_md` to get 300-dim vectors consistent with training.
- Token filtering rules (stopwords/alpha/len>2) and **lemmatization** must remain unchanged between training and inference.
- The custom random split uses `random.random()` without a fixed seed; results vary run to run.  
  To make it deterministic, set a seed at the top:
  ```python
  import random
  random.seed(42)
  ```

---

## 7) Extending / Improving

- Replace vector **sum** with average, TF-IDF weighted sum, or SIF (Smooth Inverse Frequency).
- Try alternative classifiers (Logistic Regression per label, Linear SVMs, tree-based, etc.).
- Re-enable the **GridSearchCV** block to tune `hidden_layer_sizes`, `alpha`, etc.
- Add proper **train/validation/test** split and per-label metrics (F1 by class).
- Handle **class imbalance** (e.g., class weights or resampling).
- Package the preprocessing as a transformer and wrap the whole pipeline in `sklearn`’s `Pipeline`.

---

## 8) Troubleshooting

- **`OSError: [E050] Can't find model 'en_core_web_md'`**  
  → Run `python -m spacy download en_core_web_md` and retry.

- **All zeros or weird predictions**  
  → Ensure the same preprocessing is used; check that texts are not empty after filtering.

- **Pickle incompatibility** across Python versions  
  → Load with the same Python/scikit-learn versions used for training, or retrain.

---

## 9) File Structure

```
.
├── train.csv
├── Train_model.py
├── Use_model.py
└── NLP_BEST_Model.pkl     # created after training
```

---

## 10) License & Data Use

- Ensure you have the right to use and distribute `train.csv`.  
- SpaCy models are under their respective licenses; see SpaCy documentation.
