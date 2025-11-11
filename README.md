# Topic-Clustering-of-Political-Debates

This project applies **unsupervised text clustering** to transcripts from 2022 political debates.  
The goal was to automatically **group similar debates** and uncover **dominant topics or discussion themes** using text-mining and machine learning techniques.

---

## Data Cleaning and Preprocessing

The raw text data (column `talk_text`) was cleaned to remove noise and standardize formatting.  
Main steps:
- Converted all text to lowercase  
- Removed numbers, punctuation, and percentages  
- Normalized whitespaces  
- Handled missing or empty entries  

This ensured that the dataset was consistent and ready for vectorization.

---

## TF-IDF Vectorization

The cleaned text was converted into numerical form using **TF-IDF**, which highlights informative terms.  
Key parameters:
- 3000 most relevant terms  
- Unigrams and bigrams included  
- English stop words removed  

The output was a sparse matrix representing word importance across debates.

---

## K-Means Clustering and Silhouette Score

To identify natural groupings, I used the **K-Means** algorithm.  
The optimal number of clusters (*k*) was selected based on the **silhouette score**, which measures how well each document fits within its cluster compared to others.  
The best-scoring *k* defined the final number of clusters, and the **top keywords** were extracted for each one to describe its theme.

---

## Visualization with PCA

For visual interpretation, **Principal Component Analysis (PCA)** reduced the high-dimensional TF-IDF data to two components.  
The resulting 2D plot shows how debates group together, with colors representing clusters.

---

##  Summary

1. Cleaned and normalized debate transcripts  
2. Represented text with TF-IDF  
3. Determined optimal cluster number via silhouette score  
4. Applied K-Means for clustering  
5. Used PCA for visualization  

This process revealed meaningful topic structures across political debates and demonstrated how text mining can uncover hidden thematic patterns.
