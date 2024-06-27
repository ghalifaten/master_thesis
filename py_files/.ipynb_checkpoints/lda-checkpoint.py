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
parser = argparse.ArgumentParser()

def append_to_line(filename, line_number, append_str):
    # Read all lines from the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Check if the specified line number is within the bounds of the file
    if 0 <= line_number < len(lines):
        # Append the string to the specified line
        lines[line_number] = lines[line_number].rstrip('\n') + append_str + '\n'
    else:
        raise IndexError("Line number out of range")

    # Write all lines back to the file
    with open(filename, 'w') as file:
        file.writelines(lines)
        
def get_corpus(data, min_len=3):
    # Create a Dictionary: a mapping between words and their integer IDs
    id2word = corpora.Dictionary(data)
    
    # Remove tokens of 1 or 2 letters
    del_ids = [k for k,v in id2word.items() if len(v)<min_len]
    id2word.filter_tokens(bad_ids=del_ids)
    
    # Create a corpus: a list of documents represented as a BoW
    corpus = [id2word.doc2bow(text) for text in data]
    
    return id2word, corpus

def get_model(corpus, id2word, res_file, num_topics=3, passes=10, decay=0.5, iterations=50):
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
    # put in shape and save to txt file
    print(f"topics={num_topics}, passes={passes}: {coherence_lda}")
    append_str = " & {:.3f}".format(coherence_lda)
    line_number = passes//20 - 1 
    append_to_line(res_file, line_number, append_str)
    
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

def get_best_model(corpus, id2word, res_file, title, language, plot=False, save_plot=False):
    coh_scores = []
    decay = 0.8 
    iterations = 100
    for num_topics, passes in itertools.product(range(2, 6), range(20, 120, 20)):
        print(num_topics, passes)
        lda_model, coherence_lda = get_model(corpus, 
                                             id2word, 
                                             res_file,
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
        path = f"{folder}preprocessed/open_tasks_{lang}.csv"
    df = pd.read_csv(path)
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
        res_file = f"results/LDA_models_{lang}.txt"
        with open(res_file, "w") as f:  
            for p in range(20, 60, 20):  
                s = "& "+str(p)+"\n"
                f.write(s)
        lda_model = get_best_model(corpus=corpus, 
                                   id2word=id2word, 
                                   res_file=res_file,
                                   title="", 
                                   language="")
        for n in range(2):  
            append_to_line(res_file, n, " \\\\")
        # put in shape and save to txt file
        topics_prob = lda_model.show_topics() 
        with open(f"results/LDA_topics_{lang}.txt", "w") as f: 
            for i, s in topics_prob:
                f.write("\\textbf{Topic ") 
                f.write(str(i))
                f.write(":} \\textit{") 
                l = s.split("+")  
                for k in range(len(l)):  
                    if k>0: 
                        f.write(", ")
                    e = l[k]
                    w = e.split("*")[1].strip()[1:-1]
                    f.write(w) 
                f.write("}\n")
        
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
        ## Replace values that are less than 1/3 by NaN 
        df1 = df1.mask(df1 < 1/3).reset_index() 

        ## df_tasks_topic: [taskId, 0, 1, ...] where 0, 1, ... are the topics
        df_tasks_topics = pd.concat([_df[["taskId"]], df1], axis=1) 
        ## df_task_to_aspects: [taskId, aspectId] where aspectId is the list of aspect Ids attributed to that task
        df_task_to_aspects = df_taskaspects.groupby(by="taskId")["aspectId"].apply(list).reset_index()
        ##df_task_topic_aspect: [taskId, 0,1,..., aspectId]
        df_task_topic_aspect = pd.merge(df_tasks_topics, 
                                        df_task_to_aspects, 
                                        on="taskId", 
                                        how="inner").drop(columns=["index"])
        
        retained_aspects_per_topic = []
        num_topics = len(topics_prob)
        for i in range(num_topics): 
            df_topic = df_task_topic_aspect[[i, "aspectId"]].dropna(subset=[i]).reset_index(drop=True)             
            list_aspects = df_topic["aspectId"].to_list()  
            list_aspects = list(itertools.chain.from_iterable(list_aspects))
            with open(f"results/lda_aspects_topic_{i}", "w") as f: 
                write = csv.writer(f)
                write.writerow(set(list_aspects))
            retained_aspects_per_topic.append(set(list_aspects))  
            
            ## To reduce the number of aspects in each topic, and for the attribution of aspects to topics be as accurate as possible, we remove the least occurrent aspects in each topic. To do so, we define a threshold of occurrence (t) based on the values of the occurrences in each topic. t is defined by the median value. A fixed value of t risks having empty results.
            """           
            aspects_occurrences = Counter(list_aspects) 
            t = median(aspects_occurrences.values())
            retained_occ = dict(filter(lambda x: x[1] > t, aspects_occurrences.items()))
            retained_aspects = list(retained_occ.keys())
            retained_aspects_per_topic.append(set(retained_aspects))
            """

        ## Evaluation with jaccard dissimilarity 
        # Example usage
        print("Jaccard dissimilarity between the sets of aspects in each topic:")
        dissimilarities = []
        for s1, s2 in itertools.combinations(retained_aspects_per_topic, 2): 
            dis = jaccard_dissimilarity(s1, s2) 
            print(f"Topic {retained_aspects_per_topic.index(s1)}, Topic {retained_aspects_per_topic.index(s2)}: \t {dis}")
            dissimilarities.append(dis) 
        print(f'Average pairwise Jaccard dissimilarity: {np.mean(dissimilarities)}') 
