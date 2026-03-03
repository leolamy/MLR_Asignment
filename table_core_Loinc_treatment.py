import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# 1. Liste de vos requêtes (plus besoin de mapper à des fichiers)
queries = [
    "glucose in blood",
    "bilirubin in plasma",
    "white blood cells count",
    "calcium in urine",
    "sodium in serum",
    "creatinine in serum"
]

# 2. Chargement du dump complet LOINC (une seule fois)
df_loinc = pd.read_csv("CSV/LoincTableCore.csv", low_memory=False)
df_loinc['LONG_COMMON_NAME'] = df_loinc['LONG_COMMON_NAME'].fillna("")
df_loinc['COMPONENT'] = df_loinc['COMPONENT'].fillna("")

# 3. Préparation globale pour TF-IDF et BM25 sur la réalité de toute la base
corpus = df_loinc['LONG_COMMON_NAME'].tolist()
tokenized_corpus = [str(doc).lower().split() for doc in corpus]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
bm25 = BM25Okapi(tokenized_corpus)

df_list = []

# 4. Boucle sur les requêtes
for qid, query_text in enumerate(queries, start=1):
    print(f"Traitement de la requête : {query_text}")
    
    # On crée une copie de travail pour cette requête
    df = df_loinc[['LOINC_NUM', 'LONG_COMMON_NAME', 'COMPONENT']].copy()
    df['qid'] = qid
    
    substance = query_text.split()[0].lower()
    q_words = set(query_text.lower().split())
    
    # ÉTIQUETAGE (Vérité terrain basée sur la colonne COMPONENT)
    df['label'] = 0
    df.loc[df['COMPONENT'].astype(str).str.lower().str.contains(substance), 'label'] = 2
    df.loc[(df['label'] == 0) & (df['LONG_COMMON_NAME'].astype(str).str.lower().str.contains(substance)), 'label'] = 1
    
    # EXTRACTION DES FEATURES
    tokenized_query = query_text.lower().split()
    query_vec = vectorizer.transform([query_text])
    
    df['feature_1'] = df['LONG_COMMON_NAME'].apply(lambda x: len(q_words.intersection(set(str(x).lower().split()))))
    df['feature_2'] = df['LONG_COMMON_NAME'].apply(lambda x: sum(str(x).lower().split().count(w) for w in q_words))
    df['feature_3'] = df['LONG_COMMON_NAME'].apply(lambda x: len(str(x).split()))
    df['feature_4'] = (tfidf_matrix * query_vec.T).toarray().flatten()
    df['feature_5'] = bm25.get_scores(tokenized_query)
    df['feature_6'] = df['LONG_COMMON_NAME'].apply(lambda x: 1 if substance in str(x).lower() else 0)
    df['feature_7'] = df['LONG_COMMON_NAME'].apply(lambda x: difflib.SequenceMatcher(None, query_text, str(x).lower()).ratio())
    
    # SOUS-ÉCHANTILLONNAGE (Crucial avec 100 000 lignes)
    df_relevant = df[df['label'] > 0]
    df_irrelevant = df[df['label'] == 0]
    
    # On ne garde que 150 documents non-pertinents au hasard par requête pour ne pas noyer le modèle
    if len(df_irrelevant) > 150:
        df_irrelevant = df_irrelevant.sample(n=150, random_state=42)
        
    df_balanced = pd.concat([df_relevant, df_irrelevant])
    df_list.append(df_balanced)

# 5. Exportation
df_all = pd.concat(df_list, ignore_index=True)

with open("dataset_adarank.txt", "w") as f:
    for _, row in df_all.iterrows():
        line = (f"{int(row['label'])} qid:{row['qid']} "
                f"1:{row['feature_1']:.4f} "
                f"2:{row['feature_2']:.4f} "
                f"3:{row['feature_3']:.4f} "
                f"4:{row['feature_4']:.4f} "
                f"5:{row['feature_5']:.4f} "
                f"6:{row['feature_6']:.4f} "
                f"7:{row['feature_7']:.4f} "
                f"# {row['LOINC_NUM']}\n")
        f.write(line)

print("Génération terminée avec succès.")