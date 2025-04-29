import matplotlib.pyplot as plt
import numpy as np
import json


data = {}
with open("model_res.json", "r") as fin:
    j_data = json.load(fin)

    for key, val in j_data.items():
        key_type = key.split("_")[0]
        m_name = " ".join(key.split("_")[1:])

        if not data.get(m_name):
            data[m_name] = {}

        acc = val["accuracy"]
        precision = val["precision"]
        recall = val["recall"]
        f1 = val["f1"]

        data[m_name][key_type] = {
            "accuracy": acc, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
        }

metrics = ['accuracy', 'precision', 'recall', 'f1']
vectorizers = ['count', 'tfidf']
bar_width = 0.35
x = np.arange(len(metrics)) 

# Create one plot per model
num_models = len(data)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, (model_name, vec_scores) in enumerate(data.items()):
    ax = axs[i]
    count_scores = [vec_scores['count'][m] for m in metrics]
    tfidf_scores = [vec_scores['tfidf'][m] for m in metrics]

    # Plot bars
    b1 = ax.bar(x - bar_width/2, count_scores, bar_width, label='Count')
    b2 = ax.bar(x + bar_width/2, tfidf_scores, bar_width, label='TFIDF')

    ax.set_title(model_name, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend()

    for bar in b1 + b2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

plt.suptitle("Model Performance: Count vs TFIDF", fontsize=14)
plt.tight_layout()
plt.show()
