import os
import pandas as pd


def load_liar(folder="liar_dataset", shuffle=False):
    test_fp = os.path.join(folder, "test.tsv")
    train_fp = os.path.join(folder, "train.tsv")
    valid_fp = os.path.join(folder, "valid.tsv")

    # Drops 45-46 malformed lines probs without context 
    test_df = pd.read_csv(test_fp, sep="\t", header=None)
    train_df = pd.read_csv(train_fp, sep="\t", header=None)
    valid_df = pd.read_csv(valid_fp, sep="\t", header=None)

    if not shuffle:
        df = pd.concat([test_df, train_df, valid_df], axis=0, ignore_index=True)
        return df
    
    #Return shuffled ver
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_fever(folder="fever", shuffle=False):
    train_fp = os.path.join(folder, "fever_train.jsonl")
    valid_fp = os.path.join(folder, "shared_task_dev.jsonl")

    train_df = pd.read_json(train_fp, lines=True)
    valid_df = pd.read_json(valid_fp, lines=True)

    if not shuffle:
        df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
        return df
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_scifact(folder="scifact", shuffle=False):
    train_fp = os.path.join(folder, "claims_train.jsonl")
    valid_fp = os.path.join(folder, "claims_dev.jsonl")

    train_df = pd.read_json(train_fp, lines=True)
    valid_df = pd.read_json(valid_fp, lines=True)
    
    # Drop empty evidence
    train_df = train_df[train_df["evidence"].astype(bool)]
    valid_df = valid_df[valid_df["evidence"].astype(bool)]

    def label(evidence):
        support = 0
        contradict = 0

        for doc_evs in evidence.values():
            for ev in doc_evs:
                if ev["label"].lower() == "support":
                    support+=1
                else:
                    contradict+=1

        return 1 if support > contradict else 0

    # true false
    train_df["evidence"] = train_df["evidence"].apply(label)
    valid_df["evidence"] = valid_df["evidence"].apply(label)

    if not shuffle:
        df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
        return df
    
    #Return shuffled ver
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_pheme(folder="pheme", shuffle=False):
    fp = os.path.join(folder, "dataset.csv")

    df = pd.read_csv(fp)
    df = df.dropna(subset=["is_rumor"])
    df["is_rumor"] = df["is_rumor"].astype(int)

    # 1 is True, 0 is false
    df["is_rumor"] = df["is_rumor"].map({0:1, 1:0})

    if not shuffle:
        return df
    
    #Return shuffled ver
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def combine_fever_liar(fever_df, liar_df, shuffle=False):
    # remove cols
    fever_df = fever_df.drop(["id", "verifiable", "evidence"], axis=1)

    d_range = [i for i in range(3,13+1)]
    liar_df = liar_df.drop([0, *d_range], axis=1)
    liar_df = liar_df.rename(columns={1: "label", 2: "claim"})
    
    # remap: 1: True, 0: False
    fever_map = {"SUPPORTS": 1, "REFUTES": 0}
    fever_df["label"] = fever_df["label"].map(fever_map)
    fever_df = fever_df.dropna().reset_index(drop=True)
    fever_df["label"] = fever_df["label"].astype(int)

    #  half-true, false, barely-true and pants-fire labels to False
    liar_map = {"half-true": 0, "false": 0, "barely-true": 0, "pants-fire": 0, "true": 1, "mostly-true":1}
    liar_df["label"] = liar_df["label"].map(liar_map)
    liar_df = liar_df.dropna().reset_index(drop=True)
    
    df = pd.concat([fever_df, liar_df], axis=0, ignore_index=True)

    if not shuffle:
        return df
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def combine_fever_liar_scifact_pheme(parsed_fever_liar_df, scifact_df, pheme_df, shuffle=False):
    scifact_df = scifact_df.drop(["id", "cited_doc_ids"], axis=1)
    scifact_df = scifact_df.rename(columns={"evidence": "label"})
    scifact_df = scifact_df[["label", "claim"]]

    pheme_df = pheme_df.drop(["user.handle", "topic"], axis=1)
    pheme_df = pheme_df.rename(columns={"text":"claim", "is_rumor":"label"})
    pheme_df = pheme_df[["label", "claim"]]

    df = pd.concat([parsed_fever_liar_df, scifact_df, pheme_df], axis=0, ignore_index=True)
    
    if not shuffle:
        return df
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

