import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from textaugment import EDA
from tqdm import tqdm 

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

t = EDA()

# Synonym replacement
def augment_by_sr(df, taskAspects_df, lang, n=2):
    _df = df.copy()
    _df['description'] = _df['description'].apply(lambda text: t.synonym_replacement(text, n=n))
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split()))
    _df["taskId"] = _df["taskId"] + "_SR"
    taskAspects_df["taskId"] = taskAspects_df["taskId"] + "_SR"
    _df.to_csv(f"data/augmented_SR_{lang}.csv", index_label=False)
    taskAspects_df.to_csv(f"data/taskAspects_SR_{lang}.csv", index_label=False)

# Random insertion
def augment_by_ri(df, taskAspects_df, lang, n=2):
    _df = df.copy()
    _df['description'] = _df['description'].apply(lambda text: t.random_insertion(text, n=n))
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split()))
    _df["taskId"] = _df["taskId"] + "_RI"
    taskAspects_df["taskId"] = taskAspects_df["taskId"] + "_RI"
    _df.to_csv(f"data/augmented_RI_{lang}.csv", index_label=False)
    taskAspects_df.to_csv(f"data/taskAspects_RI_{lang}.csv", index_label=False)

# Random swap
def augment_by_rs(df, taskAspects_df, lang):
    _df = df.copy()
    _df['description'] = _df['description'].apply(lambda text: t.random_swap(text))
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split()))
    _df["taskId"] = _df["taskId"] + "_RS"
    taskAspects_df["taskId"] = taskAspects_df["taskId"] + "_RS"
    _df.to_csv(f"data/augmented_RS_{lang}.csv", index_label=False)
    taskAspects_df.to_csv(f"data/taskAspects_RS_{lang}.csv", index_label=False)

# Random deletion
def augment_by_rd(df, taskAspects_df, lang, p=0.4):
    _df = df.copy()
    _df['description'] = _df['description'].apply(lambda text: t.random_deletion(text, p=p))
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split())) 
    _df["taskId"] = _df["taskId"] + "_RD"
    taskAspects_df["taskId"] = taskAspects_df["taskId"] + "_RD"
    _df.to_csv(f"data/augmented_RD_{lang}.csv", index_label=False)
    taskAspects_df.to_csv(f"data/taskAspects_RD_{lang}.csv", index_label=False)

# Back Translation
def augment_by_bt(df, taskAspects_df, lang):
    if lang == "EN": 
        from_model_name = 'facebook/wmt19-en-de'
        to_model_name = 'facebook/wmt19-de-en'
    elif lang == "DE": 
        from_model_name = 'facebook/wmt19-de-en' 
        to_model_name = 'facebook/wmt19-en-de'
    else:
        raise f"Language {lang} not supported."
        
    back_translation_aug = naw.BackTranslationAug(
        from_model_name=from_model_name,
        to_model_name=to_model_name
        )
    _df = df.copy()
    _df['description'] = _df['description'].apply(lambda text: back_translation_aug.augment(text)[0])
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split())) 
    _df["taskId"] = _df["taskId"] + "_BT"
    taskAspects_df["taskId"] = taskAspects_df["taskId"] + "_BT"
    _df.to_csv(f"data/augmented_BT_{lang}.csv", index_label=False)
    taskAspects_df.to_csv(f"data/taskAspects_BT_{lang}.csv", index_label=False)

if __name__=="__main__": 
    print("Loading data")
    df_en = pd.read_csv("data/open_tasks_EN.csv").dropna(subset=["description"]) 
    df_de = pd.read_csv("data/open_tasks_DE.csv").dropna(subset=["description"]) 
   
    # remove test descriptions before augmentation
    df_en = df_en[df_en["word_count"] > 4]
    df_de = df_de[df_de["word_count"] > 4]

    df_taskAspects_de = pd.read_csv("data/taskAspects_DE.csv") 
    df_taskAspects_en = pd.read_csv("data/taskAspects_EN.csv") 

    for df, taskAspects_df, lang in tqdm(zip([df_en, df_de], [df_taskAspects_en, df_taskAspects_de], ["EN", "DE"])):
        print(f"\n{lang}\n")
        print("Augmenting by SR...")
        augment_by_sr(df, taskAspects_df, lang)
        print("Augmenting by RI...")
        augment_by_ri(df, taskAspects_df, lang)
        print("Augmenting by RS...") 
        augment_by_rs(df, taskAspects_df, lang)
        print("Augmenting by RD...")
        augment_by_rd(df, taskAspects_df, lang)
        # print("Augmenting by BT...")
        # augment_by_bt(df, taskAspects_df, lang)
        
    print("OK")

