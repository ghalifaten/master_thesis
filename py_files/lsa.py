import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
import spacy
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import pickle 

parser = argparse.ArgumentParser()

def lsa(texts, num_topics=5):
    # Step 1: Convert texts to TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Step 2: Apply Singular Value Decomposition (SVD)
    svd = TruncatedSVD(n_components=num_topics)
    latent_semantic_analysis = svd.fit_transform(tfidf_matrix)
    
    # Step 3: Normalize the output of SVD
    normalizer = Normalizer(copy=False)
    latent_semantic_analysis = normalizer.fit_transform(latent_semantic_analysis)
    
    # Step 4: Print the topics and their most relevant terms
    terms = tfidf_vectorizer.get_feature_names_out()
    topics = []
    for i, topic in enumerate(svd.components_):
        top_terms_idx = topic.argsort()[:-21:-1] # Top 20 terms
        top_terms = [terms[idx] for idx in top_terms_idx]
        topics.append(top_terms)
    return tfidf_vectorizer, svd, normalizer, latent_semantic_analysis, topics

def preprocess_text(texts):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_vectorizer, tfidf_matrix

def transform_text(text, tfidf_vectorizer, svd, normalizer):
    text_tfidf = tfidf_vectorizer.transform([text])
    text_lsa = svd.transform(text_tfidf)
    text_lsa_normalized = normalizer.transform(text_lsa)
    return text_lsa_normalized

def assign_topic_to_text(text, topics_lsa_normalized):
    # Transform test text to LSA space
    text_lsa_normalized = transform_text(text, tfidf_vectorizer, svd, normalizer)
    # Compute cosine similarities and return the index of the best
    similarities = cosine_similarity(text_lsa_normalized, topics_lsa_normalized)
    most_similar_topic_index = np.argmax(similarities)
    return most_similar_topic_index

if __name__=="__main__": 
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    parser.add_argument('--e', type=str, required=True, help='tasks or aspects')
    # parser.add_argument('--model', type=str, required=False, default='', help='A or empty')
    args = parser.parse_args()
    lang = args.lang.upper()
    e = args.e.lower()
    # model = args.model.upper()
    folder = f"gen_files/{lang}/"
    if e == "tasks": 
        filename = "preprocessed/trimmed_open_tasks.csv"
        columns = ["taskId","description"]
    elif e == "aspects": 
        filename = "preprocessed/concept_aspects.csv"
        columns = ["aspectId","description"]
    
    # Load data 
    df = pd.read_csv(f"{folder}{filename}")
    if lang == "EN":
        stop_words = stopwords.words('english')
    elif lang == "DE": 
        stop_words = stopwords.words('german')

    df = df.dropna(subset=["description"]).reset_index(drop=True)
    data = df["description"].to_list() 

    # Tokenize documents for Gensim
    tokenized_docs = [doc.split() for doc in data]
    # Create a Gensim dictionary and corpus
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    coh_scores = []
    for num_topics in range(2,6):
        tfidf_vectorizer, svd, normalizer, lsa_output, topics = lsa(data, num_topics=num_topics)
        # Convert top words per topic into the format required by CoherenceModel
        cm_topics = [[dictionary.token2id[word] for word in topic] for topic in topics]
        # Compute Coherence Score using the 'u_mass' coherence measure
        coherence_model = CoherenceModel(topics=cm_topics, texts=tokenized_docs, dictionary=dictionary, coherence='u_mass')
        coherence_score = coherence_model.get_coherence() # mean of coherence scores per topic
        print(f"& {num_topics} & {coherence_score} ")
        coh_scores.append(coherence_score)
        if coherence_score == max(coh_scores):
            best_n = num_topics
            best_model = (tfidf_vectorizer, svd, normalizer, lsa_output, topics)

    # save model to file 
    filename = f"{lang}_{e}_lsamodel.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)
    
    print(f"\nModel saved in temp file {filename}\n")
        
    # Assign topics using best model
    (tfidf_vectorizer, svd, normalizer, lsa_output, topics) = best_model
    # Transform topics to LSA space
    list_topics_normalized = []
    for topic in topics:
        list_topics_normalized.append(transform_text(" ".join(topic), tfidf_vectorizer, svd, normalizer)[0].tolist())
        topics_lsa_normalized = np.array(list_topics_normalized)
        
    # Assign a topic to each task of df 
    df_topic = df[columns] 
    df_topic["topic"] = df_topic["description"].apply(lambda text: assign_topic_to_text(text, topics_lsa_normalized)) 

    # Save results 
    df_topic.to_csv(f"{folder}/LSA/{e}_topics.csv", index_label=False)



    

    