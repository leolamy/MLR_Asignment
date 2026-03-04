import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.model_selection import train_test_split

# 1. Définition des requêtes
train_queries = [
    "glucose in blood",
    "bilirubin in plasma",
    "white blood cells count",
    "potassium in serum",
    "chloride in serum",
    "urea nitrogen in blood",
    "hemoglobin in blood",
    "platelet count in blood",
    "thyroid stimulating hormone in serum",
    "alanine aminotransferase in serum",
    "creatine kinase in serum",
    "c reactive protein in serum",
    "lactate dehydrogenase in serum",
]

test_queries = [
    "calcium in urine",
    "sodium in serum",
    "creatinine in serum",
    "magnesium in serum",
    "alkaline phosphatase in serum",
    "albumin in serum",
    "glucose in urine",
    "bilirubin in urine",
]

# 2. Chargement du corpus global
df_loinc = pd.read_csv("CSV/LoincTableCore.csv", low_memory=False)
df_loinc['LONG_COMMON_NAME'] = df_loinc['LONG_COMMON_NAME'].fillna("")
df_loinc['COMPONENT'] = df_loinc['COMPONENT'].fillna("")

# 3. Séparation des documents (LOINC_NUM) en train/test
all_loinc = df_loinc['LOINC_NUM'].unique()
train_loinc, test_loinc = train_test_split(all_loinc, test_size=0.2, random_state=42)

df_train = df_loinc[df_loinc['LOINC_NUM'].isin(train_loinc)].copy()
df_test  = df_loinc[df_loinc['LOINC_NUM'].isin(test_loinc)].copy()

print(f"Taille du corpus d'entraînement : {len(df_train)} documents")
print(f"Taille du corpus de test : {len(df_test)} documents")

# 4. Fonction de génération de dataset pour un ensemble de requêtes et un DataFrame
def process_queries(queries, df_corpus, output_file, max_irrelevant):
    """
    Génère un fichier au format SVM-Rank.
    - queries : liste de requêtes
    - df_corpus : DataFrame contenant les documents (doit contenir LOINC_NUM, LONG_COMMON_NAME, COMPONENT)
    - output_file : nom du fichier de sortie
    - max_irrelevant : nombre maximum de documents non pertinents par requête
    """
    # Préparation du corpus pour les features globales (TF-IDF, BM25)
    corpus = df_corpus['LONG_COMMON_NAME'].tolist()
    tokenized_corpus = [str(doc).lower().split() for doc in corpus]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    bm25 = BM25Okapi(tokenized_corpus)

    df_list = []

    for qid, query_text in enumerate(queries, start=1):
        print(f"Traitement de la requête {qid}: {query_text}")
        df = df_corpus[['LOINC_NUM', 'LONG_COMMON_NAME', 'COMPONENT']].copy()
        df['qid'] = qid

        substance = query_text.split()[0].lower()
        q_words = set(query_text.lower().split())

        # Attribution des labels (0,1,2) basée sur COMPONENT et LONG_COMMON_NAME
        df['label'] = 0
        df.loc[df['COMPONENT'].astype(str).str.lower().str.contains(substance), 'label'] = 2
        df.loc[(df['label'] == 0) & (df['LONG_COMMON_NAME'].astype(str).str.lower().str.contains(substance)), 'label'] = 1

        # Calcul des features
        tokenized_query = query_text.lower().split()
        query_vec = vectorizer.transform([query_text])

        df['feature_1'] = df['LONG_COMMON_NAME'].apply(lambda x: len(q_words.intersection(set(str(x).lower().split()))))
        df['feature_2'] = df['LONG_COMMON_NAME'].apply(lambda x: sum(str(x).lower().split().count(w) for w in q_words))
        df['feature_3'] = df['LONG_COMMON_NAME'].apply(lambda x: len(str(x).split()))
        # TF-IDF : produit scalaire entre le vecteur du document et celui de la requête
        df['feature_4'] = (tfidf_matrix * query_vec.T).toarray().flatten()
        # BM25 : score du document pour la requête
        df['feature_5'] = bm25.get_scores(tokenized_query)
        df['feature_6'] = df['LONG_COMMON_NAME'].apply(lambda x: 1 if substance in str(x).lower() else 0)
        df['feature_7'] = df['LONG_COMMON_NAME'].apply(lambda x: difflib.SequenceMatcher(None, query_text, str(x).lower()).ratio())

        # Séparation pertinents / non pertinents
        df_relevant = df[df['label'] > 0]
        df_irrelevant = df[df['label'] == 0]

        # Échantillonnage des non pertinents pour contrôler la difficulté
        if len(df_irrelevant) > max_irrelevant:
            df_irrelevant = df_irrelevant.sample(n=max_irrelevant, random_state=42)

        df_list.append(pd.concat([df_relevant, df_irrelevant]))

    df_all = pd.concat(df_list, ignore_index=True)

    # Écriture au format SVM-Rank
    with open(output_file, "w") as f:
        for _, row in df_all.iterrows():
            line = (f"{int(row['label'])} qid:{row['qid']} "
                    f"1:{row['feature_1']:.4f} 2:{row['feature_2']:.4f} "
                    f"3:{row['feature_3']:.4f} 4:{row['feature_4']:.4f} "
                    f"5:{row['feature_5']:.4f} 6:{row['feature_6']:.4f} "
                    f"7:{row['feature_7']:.4f} # {row['LOINC_NUM']}\n")
            f.write(line)

# 5. Génération des fichiers
process_queries(train_queries, df_train, "train_dataset.txt", max_irrelevant=150)
process_queries(test_queries, df_test, "test_dataset.txt", max_irrelevant=1000)

print("Fichiers générés avec succès.")