# TruthRadar Training

Training files for models run by the [TruthRadar API]("https://github.com/meyersa/truthradar-api")

## Models

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| LogisticRegression | 0.744    | 0.848     | 0.777  | 0.811    |
| BernoulliNB        | 0.765    | 0.791     | 0.908  | 0.845    |
| RandomForest       | 0.754    | 0.793     | 0.880  | 0.834    |
| XGBoost            | 0.739    | 0.734     | 0.986  | 0.842    |
| PassiveAggressive  | 0.744    | 0.849     | 0.775  | 0.810    |
| SGDClassifier      | 0.733    | 0.774     | 0.877  | 0.822    |
| RidgeClassifier    | 0.790    | 0.801     | 0.934  | 0.862    |
| MiniLM             | 0.587    | 0.701     | 0.720  | 0.710    |
| DistilBERT         | 0.592    | 0.708     | 0.719  | 0.713    |

> All results are based on binary classification with class-balanced data and evaluated using held-out test sets. Confusion matrices are available in `results.json`.

## Output

- Best model per type saved in `/models`
- TF-IDF vectorizer saved in `/vectorizers`
- Training metrics saved in `results.json`
- Hyperparameters saved in `/params`

## Usage

```bash
pip install -r requirements.txt
python ml.py
```

## ENVs

| Variable         | Default Value | Description                                      |
|------------------|----------------|--------------------------------------------------|
| `VECTORIZER_DIR` | `vectorizers`  | Directory to save TF-IDF vectorizers             |
| `MODEL_DIR`      | `models`       | Directory to store trained models                |
| `RESULTS_LOCATION` | `results.json` | File path for saving evaluation metrics          |
| `PARAM_DIR`      | `params`       | Directory to store best hyperparameters          |
| `WEBHOOK_URL`    | *(none)*       | Optional Discord webhook for training summaries  |
