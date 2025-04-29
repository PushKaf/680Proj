import re
import json
import nltk
nltk.download('punkt_tab')
import pickle
import contractions
from dataset import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB # Binary data, duh
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from joblib import Parallel, delayed, dump, load
from sklearnex import patch_sklearn
patch_sklearn()


def preprocess_text(text):
    try:
        text = contractions.fix(text)
    except Exception as e:
        print("Probably non-english lang, skipping contractions fix")

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE) # we get some different langs for tweets
    tokens = nltk.word_tokenize(text)

    return " ".join(tokens)

def save_vecotorizer(fn, vec):
    with open(fn, "wb") as fout:
        pickle.dump(vec, fout)

def apply_vectorizations(texts, save=False):
    tfidf_vec = TfidfVectorizer()
    count_vec = CountVectorizer()

    tfidf_texts = tfidf_vec.fit_transform(texts)
    count_texts = count_vec.fit_transform(texts)

    if save:
        os.makedirs("vectorizers", exist_ok=True)
        save_vecotorizer("vectorizers/vectorizer_tfidf.pkl", tfidf_vec)
        save_vecotorizer("vectorizers/vectorizer_count.pkl", count_vec)

    return tfidf_texts, count_texts

def get_model_metrics(model, X_test, y_test, print_metrics=False):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    if print_metrics:
        print(f"Accuracy: {accuracy} | Percision: {precision} | Recall: {recall} | F1: {f1}")
        print("Confusion Matrix:\n", conf_matrix)
        print()

    return accuracy, precision, recall, f1, conf_matrix

def load_model(model_file, folder="models"):
    if folder:
        model_path = os.path.join(folder, model_file)
    else:
        model_path = model_file

    model = load(model_path)

    return model

def grid_search(vec, X_train, y_train, cv):
    model_params = {
        "Logistic Regression": {
            "max_iter": [100, 200, 400, 800],
            "penalty": ["l1", "l2", "elasticnet", None],
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear", "newton-cg", "sag"]
        },

        "Bernoulli NB": {
            "alpha": [.001, .01, .1, 1, 10]
        },

        "Random Forest": {
            "n_estimator": [25, 50, 75, 100],
            "max_depth": [None, 10, 15, 20],
            "max_features": [1, 3, 5, 7]
        },

        "XGBoost": {
            "max_depth": [3, 5, 7, 9],
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.05, 0.01, 0.1]
        }
    }

    _models = {
        "Logistic Regression": LogisticRegression(),
        "Bernoulli NB": BernoulliNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier()
    }

    for k, m in _models.items():
        print("Training:",k)
        grid = GridSearchCV(m, model_params[k], scoring="f1", cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        model_fn = f"models/{vec}_{k}_searched_model.pkl"
        params_fn = f"params/{vec}_{k}_params.json" 

        print(f"Saving Model: {model_fn}")
        dump(grid.best_estimator_, model_fn)
        
        print(f"Saving Params: {params_fn}")
        with open(params_fn, "w") as fout:
            json.dump(grid.best_params_, fout, indent=4)

def get_saved_models(folder="models"):
    model_files = []

    for files in os.listdir(folder):
        model_files.append(files)

    return model_files

def test_all_models(save_report=True):
    print("Loading models")
    loaded_models = {}
    for file in get_saved_models():
        if "NB_searched_model" in file:
            print("BIG BOY YES", file)
            loaded_models[file.split(".pkl")[0]] = load_model(file)

    save_report = {}
    for name, model in loaded_models.items():
        m_type = name.split("_")[0]

        print("Name:", name)
        
        if m_type == "count":
            accuracy, precision, recall, f1, conf_matrix = get_model_metrics(model, count_X_test, count_y_test, print_metrics=True)
        else:
            accuracy, precision, recall, f1, conf_matrix = get_model_metrics(model, tfidf_X_test, tfidf_y_test, print_metrics=True)

        save_report[name] = {
            "accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "confusion_matrix": conf_matrix.tolist()
        }

    if save_report:
        print("Saving Model Report")
        with open("model_res.json", "w") as fout:
            json.dump(save_report, fout, indent=4)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("params", exist_ok=True)

    # load dfs
    print("Loading DFs, small boom")
    liar_df = load_liar()
    fever_df = load_fever()
    scifact_df = load_scifact()
    pheme_df = load_pheme()

    print("Combing DFs, boom")
    # comb to make mega df
    fever_liar_df = combine_fever_liar(fever_df, liar_df)
    df = combine_fever_liar_scifact_pheme(fever_liar_df, scifact_df, pheme_df, shuffle=True)
    print(df)

    # Preprocess
    print("Preprocessing")
    df["claim"] = df["claim"].apply(preprocess_text)
    texts = df["claim"]

    # vector
    print("Vectorizing")
    X_tfidf, X_count = apply_vectorizations(texts, save=True)
    y = df["label"]

    # Split
    print("Splitting")
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    count_X_train, count_X_test, count_y_train, count_y_test = train_test_split(X_count, y, test_size=0.2, random_state=42)

    # Search for best mosels w/ both vecs
    print("Grid Search, BIG BOOM")
    grid_search("tfidf", tfidf_X_train, tfidf_y_train, cv=3)
    grid_search("count", count_X_train, count_y_train, cv=3)

    # print("Testing Models, BIG BIG BOOM BOOM")
    # test_all_models()

