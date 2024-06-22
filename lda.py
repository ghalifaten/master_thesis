import argparse
import pandas as pd
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel

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

def get_best_model(corpus, id2word, title, language, plot=True, save_plot=True):
    coh_scores = []
    for num_topics in range(2, 11):
        lda_model = LdaModel(
            corpus=corpus, 
            id2word=id2word, 
            num_topics=num_topics, 
            # distributed=True,
            passes=80, 
            update_every=1,
            alpha='auto', 
            eta=None, 
            decay=0.9,
            eval_every=5,
            iterations=100, 
            per_word_topics=True)
        
        coherence_model_lda = CoherenceModel(
            model=lda_model, 
            texts=data, 
            dictionary=id2word, 
            coherence='c_v')
        
        coherence_lda = coherence_model_lda.get_coherence()
        print(f"{num_topics}: {coherence_lda}")
        coh_scores.append(coherence_lda)
        if coherence_lda == max(coh_scores):
            best_lda = lda_model
        
    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(range(2, 11), coh_scores, marker='o', linestyle='--')
        ax.title.set_text(title)
        ax.set_ylabel("Coherence score")
        ax.set_xlabel('Number of topics')
        ax.grid(True)
    if save_plot:
        ax.get_figure().savefig(f"figures/LDA_coh_{lang}", bbox_inches="tight")
        
    return best_lda 

if __name__=="__main__": 
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    args = parser.parse_args()
    lang = args.lang.upper()

    df = pd.read_csv(f"data/all_preprocessed_tasks_{lang}.csv") 
    df.dropna(subset=["description"], inplace=True)
    data = df["description"].str.split().to_list() 
    id2word, corpus = get_corpus(data)
    if lang == "DE": 
        language = "german"
    elif lang == "EN":
        language = "english"
    title = f"Coherence score by number of topics in {language} tasks" 
    lda_model = get_best_model(corpus, id2word, title=title, language=lang)

    lda_model.show_topics()
    
    print("OK")