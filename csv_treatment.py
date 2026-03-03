import pandas as pd
import numpy as np
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import difflib


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
    "Loinc_2.82_creatinine_in_plasma.csv": "creatinine in plasma",
    "Loinc_2.82_urea_nitrogen_in_urine.csv": "urea nitrogen in urine",
    "Loinc_2.82_lactate_dehydrogenase_in_serum.csv": "lactate dehydrogenase in serum",
    "Loinc_2.82_lactate_dehydrogenase_in_plasma.csv": "lactate dehydrogenase in plasma",
    "Loinc_2.82_bilirubin_direct_in_plasma.csv": "bilirubin direct in plasma",
    "Loinc_2.82_bilirubin_indirect_in_plasma.csv": "bilirubin indirect in plasma",
    "Loinc_2.82_bilirubin_total_in_serum.csv": "bilirubin total in serum",
    "Loinc_2.82_bilirubin_direct_in_serum.csv": "bilirubin direct in serum",
    "Loinc_2.82_bilirubin_indirect_in_serum.csv": "bilirubin indirect in serum",
    "Loinc_2.82_cholinesterase_in_serum.csv": "cholinesterase in serum",
    "Loinc_2.82_hdl_cholesterol_in_plasma.csv": "hdl cholesterol in plasma",
    "Loinc_2.82_homocysteine_in_plasma.csv": "homocysteine in plasma",
    "Loinc_2.82_homocysteine_in_serum.csv": "homocysteine in serum",
    "Loinc_2.82_beta_hydroxybutyrate_in_serum.csv": "beta hydroxybutyrate in serum",
    "Loinc_2.82_beta_hydroxybutyrate_in_plasma.csv": "beta hydroxybutyrate in plasma",
    "Loinc_2.82_pyruvate_in_plasma.csv": "pyruvate in plasma",
    "Loinc_2.82_pyruvate_in_serum.csv": "pyruvate in serum",
    "Loinc_2.82_calcitonin_in_serum.csv": "calcitonin in serum",
    "Loinc_2.82_gastrin_in_serum.csv": "gastrin in serum",
    "Loinc_2.82_transferrin_in_serum.csv": "transferrin in serum",
    "Loinc_2.82_transferrin_in_plasma.csv": "transferrin in plasma",
    "Loinc_2.82_prealbumin_in_serum.csv": "prealbumin in serum",
    "Loinc_2.82_rheumatoid_factor_in_serum.csv": "rheumatoid factor in serum",
    "Loinc_2.82_complement_c4_in_serum.csv": "complement c4 in serum",
    "Loinc_2.82_17_hydroxyprogesterone_in_serum.csv": "17 hydroxyprogesterone in serum",
    "Loinc_2.82_androstenedione_in_serum.csv": "androstenedione in serum",
    "Loinc_2.82_free_testosterone_in_serum.csv": "free testosterone in serum",
    "Loinc_2.82_estrone_in_serum.csv": "estrone in serum",
    "Loinc_2.82_vasopressin_in_plasma.csv": "vasopressin in plasma",
    "Loinc_2.82_metanephrine_in_plasma.csv": "metanephrine in plasma",
    "Loinc_2.82_normetanephrine_in_plasma.csv": "normetanephrine in plasma",
    "Loinc_2.82_catecholamines_in_urine.csv": "catecholamines in urine",
    "Loinc_2.82_platelet_distribution_width_in_blood.csv": "platelet distribution width in blood",
    "Loinc_2.82_erythropoietin_in_serum.csv": "erythropoietin in serum",
    "Loinc_2.82_erythropoietin_in_plasma.csv": "erythropoietin in plasma",
    "Loinc_2.82_haptoglobin_in_plasma.csv": "haptoglobin in plasma",
    "Loinc_2.82_kappa_lambda_ratio_in_serum.csv": "kappa lambda ratio in serum",
    "Loinc_2.82_fibrinogen_in_serum.csv": "fibrinogen in serum",
    "Loinc_2.82_factor_viii_activity_in_plasma.csv": "factor viii activity in plasma",
    "Loinc_2.82_von_willebrand_factor_activity_in_plasma.csv": "von willebrand factor activity in plasma",
    "Loinc_2.82_lupus_anticoagulant_in_plasma.csv": "lupus anticoagulant in plasma",
    "Loinc_2.82_oxalate_in_urine.csv": "oxalate in urine",
    "Loinc_2.82_citrate_in_urine.csv": "citrate in urine",
    "Loinc_2.82_urine_glucose.csv": "urine glucose",
    "Loinc_2.82_digoxin_in_serum.csv": "digoxin in serum",
    "Loinc_2.82_lithium_in_serum.csv": "lithium in serum",
    "Loinc_2.82_phenytoin_in_serum.csv": "phenytoin in serum",
    "Loinc_2.82_carbamazepine_in_serum.csv": "carbamazepine in serum",
    "Loinc_2.82_phenobarbital_in_serum.csv": "phenobarbital in serum",
    "Loinc_2.82_theophylline_in_serum.csv": "theophylline in serum",
    "Loinc_2.82_acetaminophen_in_serum.csv": "acetaminophen in serum",
    "Loinc_2.82_ethanol_in_serum.csv": "ethanol in serum",
    "Loinc_2.82_methotrexate_in_serum.csv": "methotrexate in serum",
    "Loinc_2.82_norovirus_rna_in_stool.csv": "norovirus rna in stool",
    "Loinc_2.82_lactoferrin_in_stool.csv": "lactoferrin in stool",
    "Loinc_2.82_ova_and_parasites_in_stool.csv": "ova and parasites in stool",
    "Loinc_2.82_stool_culture.csv": "stool culture",
    "Loinc_2.82_shiga_toxin_in_stool.csv": "shiga toxin in stool",
}

df_list = []

# Récupération automatique des CSV
csv_files = glob.glob("CSV/*.csv")

for qid, file_path in enumerate(csv_files, start=1):
    file_name = os.path.basename(file_path)
    
    # Récupération du texte de la requête
    query_text = query_mapping.get(file_name, file_name.replace('.csv', '').replace('_', ' ').lower())
    substance = query_text.split()[0].lower() # Ex: "glucose"
    
    df = pd.read_csv(file_path)
    
    df['qid'] = qid
    df['LONG_COMMON_NAME'] = df['LONG_COMMON_NAME'].fillna("")
    df['COMPONENT'] = df['COMPONENT'].fillna("")
    
    # 1. ÉTIQUETAGE (Label) - Séparé des features pour éviter le Data Leakage
    df['label'] = 0
    # Label 2 : La substance analysée est explicitement la bonne (Vérité terrain)
    df.loc[df['COMPONENT'].astype(str).str.lower().str.contains(substance), 'label'] = 2
    # Label 1 : La substance est mentionnée dans le nom, mais ce n'est pas le composant principal
    df.loc[(df['label'] == 0) & (df['LONG_COMMON_NAME'].astype(str).str.lower().str.contains(substance)), 'label'] = 1
    
    # 2. EXTRACTION DES FEATURES (Sur le texte brut LONG_COMMON_NAME)
    q_words = set(query_text.lower().split())
    corpus = df['LONG_COMMON_NAME'].tolist()
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query = query_text.lower().split()
    
    vectorizer = TfidfVectorizer()
    
    # Prévention d'erreur si le corpus est vide
    if len(corpus) > 0:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query_text])
        tfidf_array = (tfidf_matrix * query_vec.T).toarray().flatten()
    else:
        tfidf_array = np.zeros(len(df))
        
    bm25 = BM25Okapi(tokenized_corpus)
    
    df['feature_1'] = df['LONG_COMMON_NAME'].apply(lambda x: len(q_words.intersection(set(str(x).lower().split()))))
    df['feature_2'] = df['LONG_COMMON_NAME'].apply(lambda x: sum(str(x).lower().split().count(w) for w in q_words))
    df['feature_3'] = df['LONG_COMMON_NAME'].apply(lambda x: len(str(x).split()))
    df['feature_4'] = tfidf_array
    df['feature_5'] = bm25.get_scores(tokenized_query)
    
    # Feature 6 : Alignement métier simple
    df['feature_6'] = df.apply(
        lambda row: 1 if substance in str(row['LONG_COMMON_NAME']).lower() else 0, 
        axis=1
    )

    # Feature 7 : Similarité Levenshtein
    df['feature_7'] = df['LONG_COMMON_NAME'].apply(
        lambda x: difflib.SequenceMatcher(None, query_text, str(x).lower()).ratio()
    )
        
    # 3. ÉQUILIBRAGE (Under-sampling)
    df_relevant = df[df['label'] > 0]
    df_irrelevant = df[df['label'] == 0]
    
    max_irrelevant = 50
    if len(df_irrelevant) > max_irrelevant:
        df_irrelevant = df_irrelevant.sample(n=max_irrelevant, random_state=42)
        
    df_balanced = pd.concat([df_relevant, df_irrelevant])
    df_list.append(df_balanced)

# 4. EXPORT SVM-RANK
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