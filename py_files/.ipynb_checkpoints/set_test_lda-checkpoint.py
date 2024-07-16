import argparse
import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
import gensim.corpora as corpora
import json

parser = argparse.ArgumentParser()

def get_aspects(lda, aspects_per_topic, tokenized): 
    # Fit input to model
    id2word = corpora.Dictionary([tokenized])
    bow = id2word.doc2bow(tokenized)

    # Get topics
    num_topics = len(lda.get_topics())
    min_prob = 1/num_topics
    # topics: [(1, p1), (2, p2), ..]
    topics = lda.get_document_topics(bow, minimum_probability=min_prob)
    topics_ids = dict(topics).keys() 

    output = []
    for id in topics_ids:
        aspects = aspects_per_topic[str(id)]
        output += aspects
    return output



def get_bow(text_data):
    # create the vocabulary
    vectorizer = CountVectorizer(input='content', stop_words=None, ngram_range=(1,1))
    
    # fit the vocabulary to the text data
    vectorizer.fit(text_data)
    
    # create the bag-of-words model
    bow_model = vectorizer.transform(text_data)

    return vectorizer, bow_model

def jaccard_dissimilarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union
    
if __name__=="__main__": 
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    args = parser.parse_args()
    lang = args.lang.upper()
    
    # Load data 
    df = pd.read_csv(f"gen_files/{lang}/preprocessed/trimmed_test_tasks.csv")

    # Load model 
    temp_file = datapath(f"{lang}_ldamodel")
    lda = LdaModel.load(temp_file)
    
    # Run model
    mapping_file = open(f'results/LDA_mapped_aspects_{lang}.json')
    aspects_per_topic = json.load(mapping_file)
    df["description"] = df["description"].str.split()
    df["output_aspects"] = df["description"].apply(lambda t: get_aspects(lda, aspects_per_topic, t))

    # Concat true assigned aspects 
    df_taskaspects = pd.read_csv(f"gen_files/{lang}/concept_task_aspects.csv")
    df_taskaspects = df_taskaspects.groupby(by="taskId")["aspectId"].apply(list).reset_index()
    df_taskaspects.columns = ["taskId","true_aspects"]
    df = pd.merge(df, df_taskaspects, on="taskId", how="inner")

    # Compare sets of output_aspects and true_aspects
    df["diss_score"] = df.apply(lambda r: jaccard_dissimilarity(set(r["output_aspects"]), set(r["true_aspects"])), axis=1)

    df["issubset"] = df.apply(lambda r: set(r["true_aspects"]).issubset(set(r["output_aspects"])), axis=1)
    disjoint_sets = df[df["diss_score"] == 1]
    print("Portion of tasks with disjoint true and output aspects sets: ", len(disjoint_sets) / len(df))
    subset = df[df["issubset"] == True]
    print("Portion of tasks where true is subset of output: ", len(subset) / len(df))
    print("Average dissimilarity score between true and assigned aspects = ", df["diss_score"].mean())
