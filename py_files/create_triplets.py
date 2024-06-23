import argparse
import hashlib
import functools
import pandas as pd 
from itertools import combinations
from itertools import product
import random
import pickle 
from tqdm import tqdm

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
    
def extract_pairs(elements, labels, t):
    positive_pairs = []
    negative_pairs = []
    for e1, e2 in combinations(elements, 2):
        common_labels = set(labels[e1]).intersection(labels[e2])
        if len(common_labels) > t:
            positive_pairs.append((e1, e2))
        if not common_labels:
            negative_pairs.append((e1, e2))
    return positive_pairs, negative_pairs

def get_id_triplet(combination):
    anchor_id = combination[0][0] 
    pos_id = combination[0][1] 
    neg_id = combination[1][1]
    return (anchor_id, pos_id, neg_id) 


# Create the parser
parser = argparse.ArgumentParser()

if __name__=="__main__":
    print("Loading tasks")
    # Add named arguments
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    # parser.add_argument('--data', type=str, required=False, default= ,help='augmented or not')
    parser.add_argument('--t', type=int, required=False, default=50, help='Threshold for common labels')
    parser.add_argument('--n', type=int, required=False, help='Number of combinations to sample')
    
    # Parse the arguments
    args = parser.parse_args()
    lang = args.lang.upper()
    N = args.n
    t = args.t

    print("Loading data")
    df = pd.read_csv(f"data/preprocessed_open_tasks_{lang}.csv") 
    df = df.dropna(subset=["description"])
    df.reset_index(inplace=True, drop=True) 

    df_taskAspects = pd.read_csv(f"data/taskAspects_{lang}.csv") 
    mapping_df = df_taskAspects.groupby(by="taskId")["aspectId"].apply(list).reset_index()
    elements = mapping_df["taskId"].to_list()

    labels = {}
    for index, row in mapping_df.iterrows():
        taskId = row["taskId"]
        labels[taskId] = row["aspectId"]

    print("Extracting pairs")
    positive_pairs, negative_pairs = extract_pairs(elements, labels, t)

    anchors = {p[0] for p in positive_pairs}.intersection({p[0] for p in negative_pairs})
    
    if N != None:
        anchors = random.sample(list(anchors), N)
    print(len(list(anchors)))
        
    print("Computing combinations")
    combinations = []
    for a in tqdm(anchors): 
        a_positive_pairs = filter(lambda t: t[0] == a, positive_pairs)
        a_negative_pairs = filter(lambda t: t[0] == a, negative_pairs)
        combinations += list(product(a_positive_pairs, a_negative_pairs))
    
    
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
    with open(f"data/triplets_{lang}.pkl", 'wb') as f:
        pickle.dump(triplets, f)

    print("\nNumber of triplets generated: {}\n".format(len(triplets["anchor"])))

    