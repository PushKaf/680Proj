import os
import re
import json
import pickle
import logging
from typing import Tuple
import requests

import nltk
from nltk.corpus import stopwords
import contractions
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import dump, load
from sklearnex import patch_sklearn

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import AdamWeightDecay

from datasets import Dataset
import tensorflow as tf

from dataset import load_liar, load_fever, load_scifact, load_pheme, combine_fever_liar, combine_fever_liar_scifact_pheme

tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
patch_sklearn()

# Config
VECTORIZER_DIR = os.getenv("VECTORIZER_DIR", "vectorizers")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
RESULTS_LOCATION = os.getenv("RESULTS_LOCATION", "results.json")
PARAM_DIR = os.getenv("PARAM_DIR", "params")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

STOPWORDS = set(stopwords.words("english"))

# Preprocess
def preprocess_text(text: str) -> str:
    # Remove Contractions
    try:
        text = contractions.fix(text)
    except Exception:
        pass

    # Lowercase
    text = text.lower()

    # Remove Punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove Stopwards
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)
    
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def apply_vectorization(texts, save=False):
    tfidf = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=True, stop_words="english")
    vectors = tfidf.fit_transform(texts)

    if save:
        os.makedirs(VECTORIZER_DIR, exist_ok=True)
        save_pickle(tfidf, os.path.join(VECTORIZER_DIR, "tfidf.pkl"))

    return vectors, tfidf

def get_metrics_from_preds(y_pred, y_true):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return get_metrics_from_preds(y_pred, y_test)

def run_grid_search(name: str, model, params, X_train, y_train, cv=3):
    grid = GridSearchCV(model, params, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PARAM_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    dump(grid.best_estimator_, model_path)
    with open(os.path.join(PARAM_DIR, f"{name}.json"), "w") as f:
        json.dump(grid.best_params_, f, indent=2)

    return grid.best_estimator_

def train_models(X_train, y_train):
    configs = {
        "LogisticRegression": (
            LogisticRegression(),
            {"C": [0.1, 1, 10],
             "max_iter": [200, 500], 
             "solver": ["lbfgs"],
             "class_weight": ["balanced"]}
        ),
        "BernoulliNB": (
            BernoulliNB(),
            {"alpha": [0.01, 0.1, 1, 5, 10],
             "fit_prior": [True, False]}
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {"n_estimators": [50, 100], 
             "max_depth": [10, 20, 50],
             "min_samples_split": [2, 5], 
             "max_features": ["sqrt", "log2"],
             "class_weight": ["balanced"]}
        ),
        "XGBoost": (
            XGBClassifier(eval_metric='logloss'),
            {"max_depth": [3, 5], 
             "n_estimators": [50, 100], 
             "subsample": [0.9, 1.0],
             "colsample_bytree": [0.8, 1.0],
             "learning_rate": [0.01, 0.1],
             "scale_pos_weight": [1, 5, 10]}
        ),
        "PassiveAggressive": (
            PassiveAggressiveClassifier(),
            {"C": [0.01, 0.1, 1, 10],
            "max_iter": [1000],
            "loss": ["hinge", "squared_hinge"],
            "tol": [1e-3, 1e-4],
            "class_weight": ["balanced"]}
        ),
        "SGDClassifier": (
            SGDClassifier(),
            {"loss": ["hinge", "log_loss"],
             "alpha": [1e-4, 1e-3],
             "max_iter": [1000],
             "tol": [1e-3, 1e-4],
             "penalty": ["l2"],
             "class_weight": ["balanced"],
             "early_stopping": [True]}
        ),
        "RidgeClassifier": (
            RidgeClassifier(),
            {"alpha": [0.1, 1.0, 10.0],
             "tol": [1e-3, 1e-4],
            "solver": ["auto", "sparse_cg"]}
        )
    }

    results = {}
    for name, (model, params) in configs.items():
        print(f"Training: {name}")
        best_model = run_grid_search(name, model, params, X_train, y_train)
        results[name] = best_model

    return results

def train_transformer(df, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(texts):
        return tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="tf"
        )

    # Tokenize
    inputs = tokenize_function(df["claim"])
    labels = tf.convert_to_tensor(df["label"].values)

    dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), labels))
    dataset = dataset.shuffle(buffer_size=len(df))
    
    # Manual 80/20 split BEFORE batching
    split = int(0.8 * len(df))
    train_dataset = dataset.take(split)
    val_dataset = dataset.skip(split)
    
    # Batch after splitting
    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    optimizer = AdamWeightDecay(learning_rate=3e-5, weight_decay_rate=1e-2)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
    callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)
    model.fit(train_dataset, validation_data=val_dataset, epochs=5, callbacks=[callback])

    # Evaluate
    preds = model.predict(val_dataset)["logits"]
    pred_labels = tf.argmax(preds, axis=1).numpy()
    true_labels = tf.concat([y for x, y in val_dataset], axis=0).numpy()

    # Save
    save_path = os.path.join(MODEL_DIR, model_name.replace("/", "_"))
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return get_metrics_from_preds(pred_labels, true_labels)

def send_discord_notification(model_name, f1_score):
    if not WEBHOOK_URL: 
        return None
    
    message = {
        "content": f"**Best Model:** `{model_name}`\n**F1 Score:** {f1_score:.4f}"
    }
    try:
        requests.post(WEBHOOK_URL, json=message)
    except Exception as e:
        print(f"Failed to send webhook: {e}")

def main():
    from dataset import load_liar, load_fever, load_scifact, load_pheme, combine_fever_liar, combine_fever_liar_scifact_pheme

    liar_df = load_liar()
    fever_df = load_fever()
    scifact_df = load_scifact()
    pheme_df = load_pheme()

    df = combine_fever_liar(fever_df, liar_df)
    df = combine_fever_liar_scifact_pheme(df, scifact_df, pheme_df, shuffle=True)

    df["claim"] = df["claim"].map(preprocess_text)
    y = df["label"]
    X_vec, _ = apply_vectorization(df["claim"], save=True)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
    models = train_models(X_train, y_train)

    metrics = {}
    for name, model in models.items():
        metrics[name] = get_metrics(model, X_test, y_test)

    # print("Training MiniLM transformer...")
    metrics["MiniLM"] = train_transformer(df, "microsoft/MiniLM-L12-H384-uncased")

    # print("Training DistilBERT transformer...")
    metrics["DistilBERT"] = train_transformer(df, "distilbert-base-uncased")

    with open(RESULTS_LOCATION, "w") as f:
        json.dump(metrics, f, indent=2)

    best_model = max(metrics.items(), key=lambda kv: kv[1]["f1"])
    send_discord_notification(best_model[0], best_model[1]["f1"])

if __name__ == "__main__":
    main()
