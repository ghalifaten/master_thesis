import argparse
import pandas as pd
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from collections import Counter
import itertools
from statistics import median 
from scipy.spatial.distance import pdist, squareform
import numpy as np
from gensim.test.utils import datapath
import csv
import json 

parser = argparse.ArgumentParser()

def get_corpus(data, min_len=3):
    # Create a Dictionary: a mapping between words and their integer IDs
    id2word = corpora.Dictionary(data)
    
    # Remove tokens of 1 or 2 letters
    del_ids = [k for k,v in id2word.items() if len(v)<min_len]
    id2word.filter_tokens(bad_ids=del_ids)
    
    # Create a corpus: a list of documents represented as a BoW
    corpus = [id2word.doc2bow(text) for text in data]
    
    return id2word, corpus

def get_model(corpus, id2word, num_topics=3, passes=10, decay=0.5, iterations=50):
    coh_scores = []
    lda_model = LdaModel(
        corpus=corpus, 
        id2word=id2word, 
        num_topics=num_topics, 
        distributed=False,
        passes=passes, 
        update_every=1,
        alpha='auto', 
        eta=None, 
        decay=decay,
        eval_every=5,
        iterations=iterations, 
        per_word_topics=True)
    
    coherence_model_lda = CoherenceModel(
        model=lda_model, 
        texts=data, 
        dictionary=id2word, 
        coherence='c_v')
        
    coherence_lda = coherence_model_lda.get_coherence()
    # TODO put in shape and save to txt file
    print(f"topics={num_topics}, passes={passes}: {coherence_lda}")

    return lda_model, coherence_lda

def plot_coh_score(coh_scores, x_range, title, language, save=True): 
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_range, coh_scores, marker='o', linestyle='--')
    ax.title.set_text(title)
    ax.set_ylabel("Coherence score")
    ax.set_xlabel('Number of topics')
    ax.grid(True)
    if save:
        ax.get_figure().savefig("figures/LDA_coh_"+language, bbox_inches="tight")

def get_best_model(corpus, id2word, title, language, plot=False, save_plot=False):
    coh_scores = []
    decay = 0.8 
    iterations = 100
    for num_topics, passes in itertools.product(range(3, 4), range(40, 41)):
        lda_model, coherence_lda = get_model(corpus, 
                                             id2word, 
                                             num_topics=num_topics, 
                                             passes=passes, 
                                             decay=decay, 
                                             iterations=iterations)
        coh_scores.append(coherence_lda)
        if coherence_lda == max(coh_scores):
            best_model = lda_model

    if plot:
        plot_coh_score(coh_scores, title, language, save_plot)

    return best_model 

## To evaluate dissimilarity of the sets of topics, we use the Jaccard similarity metric. 
def jaccard_dissimilarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union


if __name__=="__main__": 
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    parser.add_argument('--model', type=str, required=False, default='', help='A or empty')
    args = parser.parse_args()
    lang = args.lang.upper()
    model = args.model.upper()
    folder = f"gen_files/{lang}/"
    if model=="A":
        path = f"{folder}preprocessed/trimmed_open_tasks.csv"
    else:
        path = f"{folder}preprocessed/open_tasks.csv"
    df = pd.read_csv(path)
    print(f"Size of df: {len(df)}")
    df_taskaspects = pd.read_csv(f"{folder}concept_task_aspects.csv")
    
    df = pd.merge(df, df_taskaspects, on="taskId", how="inner") 
    df.reset_index(drop=True, inplace=True)

    _df = df[["taskId", "description", "topic_id"]].drop_duplicates("taskId")
    _df = _df.dropna(subset=["description"]).reset_index()
    data = _df["description"].str.split().to_list() 
    id2word, corpus = get_corpus(data)
    lda_model = get_best_model(corpus=corpus, 
                               id2word=id2word, 
                               title="", 
                               language="")
    
    # save model to file 
    temp_file = datapath(f"{lang}_ldamodel")
    lda_model.save(temp_file)
    print(f"\nModel saved in temp file {lang}_ldamodel\n")
    
    # Attribution of labels 
    documents = _df["description"].to_list()
    ## Infer topic distributions for each document
    topic_distributions = lda_model.get_document_topics(corpus)
    doc_to_topic = {}
    for (i, d) in enumerate(topic_distributions): 
        doc_to_topic[i] = {u:v for (u,v) in d} 
    df1 = pd.DataFrame.from_dict(doc_to_topic, orient='index').sort_index()
    ## Replace values that are less than 1/n by NaN 
    num_topics = len(lda_model.show_topics())
    df1 = df1.mask(df1 < 1/num_topics).reset_index() 

    ## df_tasks_topic: [taskId, 0, 1, ...] where 0, 1, ... are the topics
    df_tasks_topics = pd.concat([_df[["taskId"]], df1], axis=1) 
    ## df_task_to_aspects: [taskId, aspectId] where aspectId is the list of aspect Ids attributed to that task
    df_task_to_aspects = df_taskaspects.groupby(by="taskId")["aspectId"].apply(list).reset_index()
    ##df_task_topic_aspect: [taskId, 0,1,..., aspectId]
    df_task_topic_aspect = pd.merge(df_tasks_topics, 
                                    df_task_to_aspects, 
                                    on="taskId", 
                                    how="inner").drop(columns=["index"])
    
    print("BEFORE REDUCTION")
    retained_aspects_per_topic = []
    for i in range(num_topics): 
        df_topic = df_task_topic_aspect[[i, "aspectId"]].dropna(subset=[i]).reset_index(drop=True) 
        list_aspects = df_topic["aspectId"].to_list()  
        list_aspects = list(itertools.chain.from_iterable(list_aspects))
        retained_aspects_per_topic.append(set(list_aspects))  
        print(f"Topic {i}, #aspects = {len(set(list_aspects))}")
        
    ## Evaluation with jaccard dissimilarity 
    dissimilarities = []
    for s1, s2 in itertools.combinations(retained_aspects_per_topic, 2): 
        dis = jaccard_dissimilarity(s1, s2) 
        dissimilarities.append(dis) 
    print(f'score: {np.mean(dissimilarities)}') 

        #LABELS REDUCTION
        ## To reduce the number of aspects in each topic, and for the attribution of aspects to topics be as accurate as possible, we remove the least occurrent aspects in each topic. To do so, we define a threshold of occurrence (t) based on the values of the occurrences in each topic. t is defined by the median value. A fixed value of t risks having empty results.

    print("\nAFTER REDUCTION")
    retained_aspects_per_topic = []
    for i in range(num_topics): 
        df_topic = df_task_topic_aspect[[i, "aspectId"]].dropna(subset=[i]).reset_index(drop=True) 
        list_aspects = df_topic["aspectId"].to_list()  
        list_aspects = list(itertools.chain.from_iterable(list_aspects))
        
        aspects_occurrences = Counter(list_aspects) 
        t = median(aspects_occurrences.values())
        retained_occ = dict(filter(lambda x: x[1] > t, aspects_occurrences.items()))
        retained_aspects = list(retained_occ.keys())
        retained_aspects_per_topic.append(set(retained_aspects))
        print(f"Topic {i}, t = {t}, #aspects = {len(set(retained_aspects))}")
        
    ## Evaluation with jaccard dissimilarity 
    dissimilarities = []
    for s1, s2 in itertools.combinations(retained_aspects_per_topic, 2): 
        dis = jaccard_dissimilarity(s1, s2) 
        dissimilarities.append(dis) 
    print(f'score: {np.mean(dissimilarities)}') 

    result_as_dict = {topic: list(ids) for topic, ids in enumerate(retained_aspects_per_topic)}
    with open(f"results/LDA_mapped_aspects_{lang}.json", "w") as outfile: 
        json.dump(result_as_dict, outfile)
        
