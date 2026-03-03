# MLR_Assignment
MLR assignment in Information Retrieval subject.

## Project Files Description
- `CSV/` : Folder containing the raw data exports downloaded from the LOINC website.
- `csv_treatment.py` : Python script that processes the CSV files, extracts features, balances the classes, and generates the training dataset.
- `dataset_adarank.txt` : The final training dataset formatted in SVM-Rank format, ready for model consumption.
- `RankLib-2.16.jar` : The Java public library (from Lemur Project) used to train the AdaRank model.
- `model_adarank.txt` : The output file containing the trained AdaRank model parameters.

## Download CSV files
On the LOINC website, we searched for the 3 queries ("glucose in blood", "bilirubin in plasma", "White blood cells count") written in the guidelines and downloaded the data. we stored them in the `CSV/` folder.

Because the manual procedure take time, we downloaded the official LOINC 2.82 release and added it directly to the project. Instead of manually exporting results from the LOINC website for each query, we generate our query-specific datasets automatically from the full LOINC table (see extractQuery.py). 

Concretely, we define a list of queries (e.g., “glucose in blood”, “bilirubin in plasma”, “white blood cells count”), then run our script to filter the LOINC table and export one CSV per query. Each exported file is stored in the CSV/ folder and is named following the pattern:

Loinc_2.82_<query_slug>.csv

This makes the dataset creation reproducible and allows us to scale to many queries without repetitions or manual downloads.

## Treatment of CSV files
To construct the training dataset, I extracted the most effective metrics from the foundational AdaRank paper. These features include:
- **Word Overlap:** The count of exact query words present in the document.
- **Term Frequency (TF):** The total number of times the query terms appear in the document.
- **Document Length:** Used to normalize scores, acknowledging that shorter documents containing the keyword are often more relevant.
- **TF-IDF:** A metric that weights the term frequency by the rarity of the word across the entire document collection.
- **BM25 Score:** A state-of-the-art information retrieval metric, which is directly integrated as a feature for the AdaRank algorithm.

## How to Run the Project

**1. Generate the training dataset**
Run the Python script to parse the CSVs and extract features:
```bash
python3 csv_treatment.py
java -jar RankLib-2.16.jar -train dataset_adarank.txt -ranker 3 -metric2t NDCG@5 -norm zscore -save model_adarank.txt
```
## Future improvements 
expand dataset in queries (adding new data to the dataset)
-> overfit a lot

## Results 
with 3 queries 
-> overfits a lot, stop at iteration one,  0.8687 training accuracy 

## Queries 
- Glucose in blood
- bilirubin in plasma 
- White blood cells count
- Calcium Urine
