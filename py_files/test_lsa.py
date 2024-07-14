import argparse
from googletrans import Translator
import nltk
from nltk.tokenize import sent_tokenize
import pickle
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
import pandas as pd

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
stemmer = PorterStemmer()
parser = argparse.ArgumentParser()

def detect_and_translate(text, target_language):
    translator = Translator()
    sentences = sent_tokenize(text)
    translated_sentences = []

    for sentence in sentences:
        try:
            detected_language = translator.detect(sentence).lang
            # print(f"Detected language: {detected_language} for sentence: {sentence}")
            translation = translator.translate(sentence, dest=target_language)
            translated_sentences.append(translation.text)
        except Exception as e:
            print(f"Error translating sentence: {sentence}")
            print(e)
            translated_sentences.append(sentence)  # Append the original sentence if translation fails

    return ' '.join(translated_sentences)

def trim_to_sentences(text, max_sentences):
    sentences = sent_tokenize(text)
    trimmed_text = ' '.join(sentences[:max_sentences])
    return trimmed_text

def load_tools(language, stopwords=stopwords):
    """
    Load open tasks in DataFrame from csv file.
    Define stopwords and lemmatizer appropriate to given language of data.
    """
    if language == "DE": 
        stopwords = stopwords.words('german') 
        lemmatizer = spacy.load("de_core_news_sm") 
        stoplemmas = ["fur", "zwei", "drei", "geben", "bei", "immer", "gehen", "sehen"] 

    elif language == "EN": 
        stopwords = stopwords.words('english')
        lemmatizer = spacy.load('en_core_web_sm')
        stoplemmas = ["like", "use", "say", "go", "one", "see", "luke", "two", "three", "ga", "come", "tom", "gwen", "make", "get", "chri"]

    else:
        raise language + " language not supported." 
    return stopwords, lemmatizer, stoplemmas

def preprocess(text, lemmatizer, stopwords, stoplemmas, stemmer=stemmer):
    if isinstance(text, float):
        return ""
    # lowercase, remove punctuation, tokenize 
    words = simple_preprocess(text, deacc=True, min_len=1, max_len=50)
    # remove stopwords 
    tokens = [word for word in words if word not in stopwords]
    # stemming
    stems = [stemmer.stem(token) for token in tokens]
    # lemmatize
    stemmed_doc = lemmatizer(" ".join(stems))
    lemmas = [s.lemma_ for s in stemmed_doc]
    lemmas = [lemma for lemma in lemmas if lemma not in stoplemmas]
    # lowercase again (apparently stemming capitalizes the output) 
    lemmas = [lemma.lower() for lemma in lemmas] 
    return " ".join(lemmas)

def transform_text(text, tfidf_vectorizer, svd, normalizer):
    text_tfidf = tfidf_vectorizer.transform([text])
    text_lsa = svd.transform(text_tfidf)
    text_lsa_normalized = normalizer.transform(text_lsa)
    return text_lsa_normalized

def assign_topic_to_text(text, topics_lsa_normalized):
    # Transform test text to LSA space
    text_lsa_normalized = transform_text(text, tfidf_vectorizer, svd, normalizer)
    similarities = cosine_similarity(text_lsa_normalized, topics_lsa_normalized)
    most_similar_topic_index = np.argmax(similarities)
    return most_similar_topic_index

    
if __name__=="__main__": 
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    parser.add_argument('--doc', type=str, required=True, help='test document')
    args = parser.parse_args()
    lang = args.lang.upper()
    doc = args.doc
    
    # Load model 
    filename = f"{lang}_tasks_lsamodel.pkl"
    with open(filename, 'rb') as file:
        tfidf_vectorizer, svd, normalizer, lsa_output, topics = pickle.load(file)
    
    # Pre-process input document 
    ## Translate to lang
    detect_and_translate(doc, target_language=lang.lower())
    ## Truncate 
    max_sentences = 40 if lang=="DE" else 20
    truncated = trim_to_sentences(doc, max_sentences)
    ## Pre-process
    stopwords, lemmatizer, stoplemmas = load_tools(lang)
    preprocessed = preprocess(truncated, lemmatizer, stopwords, stoplemmas)

    # Assign topics using loaded model
    # Transform topics to LSA space
    list_topics_normalized = []
    for topic in topics:
        list_topics_normalized.append(transform_text(" ".join(topic), tfidf_vectorizer, svd, normalizer)[0].tolist())
        topics_lsa_normalized = np.array(list_topics_normalized)
        
    # Assign a topic to input 
    assigned_topic = assign_topic_to_text(preprocessed, topics_lsa_normalized)
    print(f"Assigned topic to input task: {assigned_topic}")

    # Assign the recommended topic(s) of aspects and then the list of aspects as saved in file
    folder = f"gen_files/{lang}/LSA/"
    df_mapping = pd.read_csv(f"{folder}mapping.csv")
    aspect_topics = df_mapping[df_mapping["taskTopic"] == assigned_topic]["aspectTopic"]
    aspect_topics = aspect_topics.to_list()
    df_aspects = pd.read_csv(f"{folder}aspects_topics.csv")[["aspectId", "topic"]]
    df_aspects = df_aspects.drop_duplicates().groupby("topic")["aspectId"].apply(list).reset_index()
    recommended_aspects = df_aspects[df_aspects["topic"].isin(aspect_topics)]["aspectId"] 
    output = list(set(item for sublist in recommended_aspects for item in sublist))

    print(f"Number of recommended aspects: {len(output)}")