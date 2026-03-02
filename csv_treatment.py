import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# 3. Augmenter le nombre de requêtes
# Ajoutez d'autres fichiers CSV ici pour dépasser les 3 requêtes de base
queries = {
    1: ("glucose in blood", "CSV/Loinc_2.82_Glucose.csv"),
    2: ("bilirubin in plasma", "CSV/Loinc_2.82_Bilirubin.csv"),
    3: ("white blood cells count", "CSV/Loinc_2.82_WhiteBlood.csv"),
}

df_list = []

for qid, (query_text, file_name) in queries.items():
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        continue
        
    df['qid'] = qid
    df['LONG_COMMON_NAME'] = df['LONG_COMMON_NAME'].fillna("")
    
    q_words = set(query_text.lower().split())
    q_len = len(q_words)
    
    # Calcul de la proportion de mots de la requête présents dans le document
    df['match_ratio'] = df['LONG_COMMON_NAME'].apply(
        lambda x: len(q_words.intersection(set(str(x).lower().split()))) / q_len if q_len > 0 else 0
    )
    
    # 1. Améliorer la logique d'étiquetage
    df['label'] = 0
    
    # Label 2 : La chaîne exacte est présente OU plus de 75% des mots correspondent
    df.loc[df['LONG_COMMON_NAME'].str.lower().str.contains(query_text) | (df['match_ratio'] >= 0.75), 'label'] = 2
    
    # Label 1 : Entre 40% et 75% des mots correspondent
    df.loc[(df['label'] == 0) & (df['match_ratio'] >= 0.40), 'label'] = 1
    
    # Préparation des features
    corpus = df['LONG_COMMON_NAME'].tolist()
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query = query_text.lower().split()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query_text])
    bm25 = BM25Okapi(tokenized_corpus)
    
    df['feature_1'] = df['LONG_COMMON_NAME'].apply(lambda x: len(q_words.intersection(set(str(x).lower().split()))))
    df['feature_2'] = df['LONG_COMMON_NAME'].apply(lambda x: sum(str(x).lower().split().count(w) for w in q_words))
    df['feature_3'] = df['LONG_COMMON_NAME'].apply(lambda x: len(str(x).split()))
    df['feature_4'] = (tfidf_matrix * query_vec.T).toarray().flatten()
    df['feature_5'] = bm25.get_scores(tokenized_query)
    
    # 2. Rééquilibrer le jeu de données (Under-sampling)
    df_relevant = df[df['label'] > 0]
    df_irrelevant = df[df['label'] == 0]
    
    # Conserver un maximum de 50 documents non pertinents par requête
    max_irrelevant = 50
    if len(df_irrelevant) > max_irrelevant:
        df_irrelevant = df_irrelevant.sample(n=max_irrelevant, random_state=42)
        
    df_balanced = pd.concat([df_relevant, df_irrelevant])
    df_list.append(df_balanced)

df_all = pd.concat(df_list, ignore_index=True)

with open("dataset_adarank.txt", "w") as f:
    for _, row in df_all.iterrows():
        line = (f"{int(row['label'])} qid:{row['qid']} "
                f"1:{row['feature_1']:.4f} "
                f"2:{row['feature_2']:.4f} "
                f"3:{row['feature_3']:.4f} "
                f"4:{row['feature_4']:.4f} "
                f"5:{row['feature_5']:.4f} "
                f"# {row['LOINC_NUM']}\n")
        f.write(line)