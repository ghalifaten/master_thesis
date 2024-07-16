import argparse
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

parser = argparse.ArgumentParser()
t = EDA()

# Synonym replacement
def augment_by_sr(df, taskAspects_df, lang, n=2):
    _df = df.copy()
    _taskAspects_df = taskAspects_df.copy()
    _df['description'] = _df['description'].apply(lambda text: t.synonym_replacement(text, n=n))
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split()))
    _df["taskId"] = _df["taskId"] + "_SR"
    _taskAspects_df["taskId"] = _taskAspects_df["taskId"] + "_SR"
    _df.to_csv(f"gen_files/{lang}/augmented/tasks_SR.csv", index_label=False)
    _taskAspects_df.to_csv(f"gen_files/{lang}/augmented/task_aspects_SR.csv", index_label=False)

# Random insertion
def augment_by_ri(df, taskAspects_df, lang, n=2):
    _df = df.copy()
    _taskAspects_df = taskAspects_df.copy()
    _df['description'] = _df['description'].apply(lambda text: t.random_insertion(text, n=n))
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split()))
    _df["taskId"] = _df["taskId"] + "_RI"
    _taskAspects_df["taskId"] = _taskAspects_df["taskId"] + "_RI"
    _df.to_csv(f"gen_files/{lang}/augmented/tasks_RI.csv", index_label=False)
    _taskAspects_df.to_csv(f"gen_files/{lang}/augmented/task_aspects_RI.csv", index_label=False)

# Random swap
def augment_by_rs(df, taskAspects_df, lang):
    _df = df.copy()
    _taskAspects_df = taskAspects_df.copy()
    _df['description'] = _df['description'].apply(lambda text: t.random_swap(text))
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split()))
    _df["taskId"] = _df["taskId"] + "_RS"
    _taskAspects_df["taskId"] = _taskAspects_df["taskId"] + "_RS"
    _df.to_csv(f"gen_files/{lang}/augmented/tasks_RS.csv", index_label=False)
    _taskAspects_df.to_csv(f"gen_files/{lang}/augmented/task_aspects_RS.csv", index_label=False)

# Random deletion
def augment_by_rd(df, taskAspects_df, lang, p=0.4):
    _df = df.copy()
    _taskAspects_df = taskAspects_df.copy()
    _df['description'] = _df['description'].apply(lambda text: t.random_deletion(text, p=p))
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split())) 
    _df["taskId"] = _df["taskId"] + "_RD"
    _taskAspects_df["taskId"] = _taskAspects_df["taskId"] + "_RD"
    _df.to_csv(f"gen_files/{lang}/augmented/tasks_RD.csv", index_label=False)
    _taskAspects_df.to_csv(f"gen_files/{lang}/augmented/task_aspects_RD.csv", index_label=False)


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
    _taskAspects_df = taskAspects_df.copy()
    _df['description'] = _df['description'].apply(lambda text: back_translation_aug.augment(text)[0])
    _df["word_count"] = _df["description"].apply(lambda s: len(s.split())) 
    _df["taskId"] = _df["taskId"] + "_BT"
    _taskAspects_df["taskId"] = _taskAspects_df["taskId"] + "_BT"
    _df.to_csv(f"gen_files/{lang}/augmented/tasks_BT.csv", index_label=False)
    _taskAspects_df.to_csv(f"gen_files/{lang}/augmented/task_aspects_BT.csv", index_label=False)


if __name__=="__main__": 
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    args = parser.parse_args()
    lang = args.lang.upper()
    
    print("Loading data")
    df = pd.read_csv(f"gen_files/{lang}/all_concept_open_tasks.csv").dropna(subset=["description"]) 
    df_taskAspects = pd.read_csv(f"gen_files/{lang}/concept_task_aspects.csv") 

    print("Augmenting by SR...")
    augment_by_sr(df, df_taskAspects, lang)
    print("Augmenting by RI...")
    augment_by_ri(df, df_taskAspects, lang)
    print("Augmenting by RS...") 
    augment_by_rs(df, df_taskAspects, lang)
    print("Augmenting by RD...")
    augment_by_rd(df, df_taskAspects, lang)
    print("Augmenting by BT...")
    augment_by_bt(df, df_taskAspects, lang)
        
    print("OK")

