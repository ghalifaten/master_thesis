import argparse
from googletrans import Translator
import nltk
from nltk.tokenize import sent_tokenize
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
import csv 

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

def get_bow(text_data):
    # create the vocabulary
    vectorizer = CountVectorizer(input='content', stop_words=None, ngram_range=(1,1))
    
    # fit the vocabulary to the text data
    vectorizer.fit(text_data)
    
    # create the bag-of-words model
    bow_model = vectorizer.transform(text_data)

    return vectorizer, bow_model
    
if __name__=="__main__": 
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    parser.add_argument('--doc', type=str, required=True, help='test document')
    args = parser.parse_args()
    lang = args.lang.upper()
    doc = args.doc
    
    # Load model 
    temp_file = datapath(f"{lang}_ldamodel")
    lda = LdaModel.load(temp_file)
    
    # Pre-process input document 
    ## Translate to lang
    detect_and_translate(doc, target_language=lang.lower())
    ## Truncate 
    max_sentences = 40 if lang=="DE" else 20
    truncated = trim_to_sentences(doc, max_sentences)
    ## Pre-process
    stopwords, lemmatizer, stoplemmas = load_tools(lang)
    preprocessed = preprocess(truncated, lemmatizer, stopwords, stoplemmas)
    tokenized = preprocessed.split()

    # Fit input to model
    id2word = corpora.Dictionary([tokenized])
    bow = id2word.doc2bow(tokenized)

    # Get topics
    num_topics = len(lda.get_topics())
    min_prob = 1/num_topics
    # topics: [(1, p1), (2, p2), ..]
    topics = lda.get_document_topics(bow, minimum_probability=min_prob)
    topics_ids = dict(topics).keys() 

    # output is the IDs of aspects to recommend
    output = []
    for id in topics_ids:  
        with open(f"results/lda_aspects_topic_{id}") as f:
            reader = csv.reader(f)
            aspects = list(reader)[0]
            output += aspects
    # print(output)
    print(f"Number of recommended aspects is: {len(output)}")

