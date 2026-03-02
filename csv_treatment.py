import pandas as pd
import numpy as np
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


query_mapping = {
    "Loinc_2.82_Glucose.csv": "glucose in blood",
    "Loinc_2.82_Bilirubin.csv": "bilirubin in plasma",
    "Loinc_2.82_WhiteBlood.csv": "white blood cells count",
    "Loinc_2.82_calciumUrine.csv": "calcium in urine",
    "Loinc_2.82_Bilirubin.csv": "bilirubin in plasma",
    "Loinc_2.82_WhiteBlood.csv": "white blood cells count",
    "Loinc_2.82_calciumUrine.csv": "calcium in urine",
    "Loinc_2.82_Creatinine_in_serum.csv": "creatinine in serum",
    "Loinc_2.82_cholesterol_total_in serum.csv": "cholesterol total in serum",
    "Loinc_2.82_protein_in_urine.csv": "protein in urine",
    "Loinc_2.82_sodium_in_serum.csv": "sodium in serum",
    "Loinc_2.82_urea_nitrogen_in_blood.csv": "urea nitrogen in blood",
    "Loinc_2.82_potassium_in_serum.csv": "potassium in serum",
    "Loinc_2.82_chloride_in_serum.csv": "chloride in serum",
    "Loinc_2.82_anion_gap_in_serum.csv": "anion gap in serum",
    "Loinc_2.82_calcium_in_serum.csv": "calcium in serum",
    "Loinc_2.82_magnesium_in_serum.csv": "magnesium in serum",
    "Loinc_2.82_phosphate_in_serum.csv": "phosphate in serum",
    "Loinc_2.82_lactate_in_plasma.csv": "lactate in plasma",
    "Loinc_2.82_creatinine_clearance_in_urine.csv": "creatinine clearance in urine",
    "Loinc_2.82_albumin_in_urine.csv": "albumin in urine",
    "Loinc_2.82_microalbumin_in_urine.csv": "microalbumin in urine",
    "Loinc_2.82_albumin_creatinine_ratio_in_urine.csv": "albumin creatinine ratio in urine",
    "Loinc_2.82_alkaline_phosphatase_in_serum.csv": "alkaline phosphatase in serum",
    "Loinc_2.82_alanine_aminotransferase_in_serum.csv": "alanine aminotransferase in serum",
    "Loinc_2.82_aspartate_aminotransferase_in_serum.csv": "aspartate aminotransferase in serum",
    "Loinc_2.82_prothrombin_time_in_plasma.csv": "prothrombin time in plasma",
    "Loinc_2.82_hdl_cholesterol_in_serum.csv": "hdl cholesterol in serum",
    "Loinc_2.82_hemoglobin_in_blood.csv": "hemoglobin in blood",
    "Loinc_2.82_platelets_count.csv": "platelets count",
    "Loinc_2.82_c_reactive_protein_in_serum.csv": "c reactive protein in serum",
    "Loinc_2.82_erythrocyte_sedimentation_rate.csv": "erythrocyte sedimentation rate",
    "Loinc_2.82_procalcitonin_in_serum.csv": "procalcitonin in serum",
    "Loinc_2.82_hemoglobin_a1c_in_blood.csv": "hemoglobin a1c in blood",
    "Loinc_2.82_insulin_in_serum.csv": "insulin in serum",
    "Loinc_2.82_ferritin_in_serum.csv": "ferritin in serum",
    "Loinc_2.82_iron_in_serum.csv": "iron in serum",
    "Loinc_2.82_vitamin_b12_in_serum.csv": "vitamin b12 in serum",
    "Loinc_2.82_folate_in_serum.csv": "folate in serum",
    "Loinc_2.82_glucose_in_urine.csv": "glucose in urine",
    "Loinc_2.82_ketones_in_urine.csv": "ketones in urine",
    "Loinc_2.82_nitrite_in_urine.csv": "nitrite in urine",
    "Loinc_2.82_leukocytes_in_urine.csv": "leukocytes in urine",
    "Loinc_2.82_creatine_kinase_in_serum.csv": "creatine kinase in serum",
    "Loinc_2.82_myoglobin_in_serum.csv": "myoglobin in serum",
    "Loinc_2.82_amylase_in_serum.csv": "amylase in serum",
    "Loinc_2.82_procalcitonin_in_plasma.csv": "procalcitonin in plasma",
    "Loinc_2.82_fibrinogen_in_plasma.csv": "fibrinogen in plasma",
    "Loinc_2.82_d_dimer_in_plasma.csv": "d dimer in plasma",
    "Loinc_2.82_haptoglobin_in_serum.csv": "haptoglobin in serum",
    "Loinc_2.82_oxygen_saturation_in_arterial_blood.csv": "oxygen saturation in arterial blood",
    "Loinc_2.82_base_excess_in_blood.csv": "base excess in blood",
    "Loinc_2.82_glucose_in_plasma.csv": "glucose in plasma",
    "Loinc_2.82_thyroglobulin_in_serum.csv": "thyroglobulin in serum",
    "Loinc_2.82_prolactin_in_serum.csv": "prolactin in serum",
    "Loinc_2.82_estradiol_in_serum.csv": "estradiol in serum",
    "Loinc_2.82_progesterone_in_serum.csv": "progesterone in serum",
    "Loinc_2.82_testosterone_in_serum.csv": "testosterone in serum",
    "Loinc_2.82_cortisol_in_serum.csv": "cortisol in serum",
    "Loinc_2.82_aldosterone_in_serum.csv": "aldosterone in serum",
    "Loinc_2.82_renin_activity_in_plasma.csv": "renin activity in plasma",
    "Loinc_2.82_c_peptide_in_serum.csv": "c peptide in serum",
    "Loinc_2.82_zinc_in_serum.csv": "zinc in serum",
    "Loinc_2.82_copper_in_serum.csv": "copper in serum",
    "Loinc_2.82_alpha_fetoprotein_in_serum.csv": "alpha fetoprotein in serum",
    "Loinc_2.82_blood_in_urine.csv": "blood in urine",
    "Loinc_2.82_bilirubin_in_urine.csv": "bilirubin in urine",
    "Loinc_2.82_urobilinogen_in_urine.csv": "urobilinogen in urine",
    "Loinc_2.82_specific_gravity_of_urine.csv": "specific gravity of urine",
    "Loinc_2.82_ph_of_urine.csv": "ph of urine",
    "Loinc_2.82_creatinine_in_urine.csv": "creatinine in urine",
    "Loinc_2.82_sodium_in_urine.csv": "sodium in urine",
    "Loinc_2.82_potassium_in_urine.csv": "potassium in urine",
    "Loinc_2.82_protein_creatinine_ratio_in_urine.csv": "protein creatinine ratio in urine",
    "Loinc_2.82_thrombin_time_in_plasma.csv": "thrombin time in plasma",
    "Loinc_2.82_apolipoprotein_b_in_serum.csv": "apolipoprotein b in serum",
    "Loinc_2.82_lipoprotein_a_in_serum.csv": "lipoprotein a in serum",
}

df_list = []

csv_files = glob.glob("CSV/*.csv")

for qid, file_path in enumerate(csv_files, start=1):
    file_name = os.path.basename(file_path)
    
    query_text = query_mapping.get(file_name, file_name.replace('.csv', '').replace('_', ' ').lower())
    
    df = pd.read_csv(file_path)
    
    df['qid'] = qid
    df['LONG_COMMON_NAME'] = df['LONG_COMMON_NAME'].fillna("")
    
    q_words = set(query_text.lower().split())
    q_len = len(q_words)
    
    df['match_ratio'] = df['LONG_COMMON_NAME'].apply(
        lambda x: len(q_words.intersection(set(str(x).lower().split()))) / q_len if q_len > 0 else 0
    )
    
    df['label'] = 0
    df.loc[df['LONG_COMMON_NAME'].str.lower().str.contains(query_text) | (df['match_ratio'] >= 0.75), 'label'] = 2
    df.loc[(df['label'] == 0) & (df['match_ratio'] >= 0.40), 'label'] = 1
    
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
    
    df_relevant = df[df['label'] > 0]
    df_irrelevant = df[df['label'] == 0]
    
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
