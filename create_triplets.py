import sys
import hashlib
import functools
import pandas as pd 
from itertools import combinations
from itertools import product
import random
import pickle 

df_cache = {}
def dataframe_checksum(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

def get_text_triplet(checksum, t):
    anchor = get_text(checksum, t[0])
    pos = get_text(checksum, t[1])
    neg = get_text(checksum, t[2])
    return (anchor, pos, neg)
    
@functools.lru_cache(maxsize=None)
def get_text(checksum, id):
    df = df_cache[checksum]
    return df[df["taskId"] == id ]["description"].values[0]
    
def extract_pairs(elements, labels):
    positive_pairs = []
    negative_pairs = []
    for e1, e2 in combinations(elements, 2):
        common_labels = set(labels[e1]).intersection(labels[e2])
        if len(common_labels) > 50:
            positive_pairs.append((e1, e2))
        if not common_labels:
            negative_pairs.append((e1, e2))
    return positive_pairs, negative_pairs

def get_id_triplet(combination):
        anchor_id = combination[0][0] 
        pos_id = combination[0][1] 
        neg_id = combination[1][1]

        return (anchor_id, pos_id, neg_id) 


if __name__=="__main__":
    print("Loading tasks")
    df = pd.read_csv("data/all_augmented_tasks_EN.csv") 
    df = df.dropna(subset=["description"])
    df.reset_index(inplace=True, drop=True) 

    print("Loading aspects")
    df_taskAspects = pd.read_csv("data/taskAspects.csv") 

    print("Merging")
    mapping_df = pd.merge(df[["taskId", "description"]], df_taskAspects[["taskId", "aspectId"]], how="inner", on=["taskId"])
    mapping_df = mapping_df.groupby(by="taskId")["aspectId"].apply(list).reset_index()
    elements = mapping_df["taskId"].to_list()
    
    labels = {}
    for index, row in mapping_df.iterrows():
        taskId = row["taskId"]
        labels[taskId] = row["aspectId"]

    print("Extracting pairs")
    positive_pairs, negative_pairs = extract_pairs(elements, labels)

    print("Computing combinations")
    combinations = list(product(positive_pairs, negative_pairs))
    
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        combinations = random.sample(combinations, n)
    
    # from [((anchor, pos), (anchor, neg))] to [(anchor, pos, neg)]
    print("Creating ID triplets from combinations")
    triplets_ids = map(get_id_triplet, combinations)
    
    checksum = dataframe_checksum(df)
    if checksum not in df_cache:
        df_cache[checksum] = df
        
    print("Getting texts")
    triplets_texts = map(lambda t: get_text_triplet(checksum, t), triplets_ids)

    print("Creating dictionary")
    anchors, positives, negatives = zip(*triplets_texts)
    triplets = {
        "anchor": anchors, 
        "positive": positives, 
        "negative": negatives
    } 

    print("Writing to file")
    with open('data/triplets.pkl', 'wb') as f:
        pickle.dump(triplets, f)

    print("OK.")


    