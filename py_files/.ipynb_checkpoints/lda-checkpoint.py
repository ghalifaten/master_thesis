import argparse
import pandas as pd
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from itertools import product

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
    for num_topics, passes in product(range(2, 6), range(40, 120, 20)):
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

if __name__=="__main__": 
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    args = parser.parse_args()
    lang = args.lang.upper()

    folder = f"gen_files/{lang}/"
    df = pd.read_csv(f"{folder}preprocessed/open_tasks_{lang}.csv")
    print(f"Size of df: {len(df)}")
    df_taskaspects = pd.read_csv(f"{folder}taskAspects_{lang}.csv")
    
    # Keeping only the tasks that have one or more aspects of type CONCEPT
    df = pd.merge(df, df_taskaspects, on="taskId", how="inner") 
    df.reset_index(drop=True, inplace=True)

    if len(df_taskaspects.taskId.unique()) != len(df.taskId.unique()): 
        print("ERROR")
    else: 
        _df = df[["taskId", "description", "topic_id"]].drop_duplicates("taskId")
        _df = _df.dropna(subset=["description"]).reset_index()
        data = _df["description"].str.split().to_list() 
        id2word, corpus = get_corpus(data)
        lda_model = get_best_model(corpus=corpus, 
                                   id2word=id2word, 
                                   title="", 
                                   language="")
        
        lda_model.show_topics()