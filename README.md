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

After preprocessing, the text was transformed into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.  
TF-IDF measures how important a word is within a document relative to the entire collection of documents.  
- Words that appear frequently in one debate but rarely across others receive higher weights,  
- While common words that appear in nearly all documents are downweighted.  

This representation captures the distinctiveness of terms and provides a meaningful basis for clustering.

Key parameters:
- Limited to 3000 most informative terms  
- Included both unigrams and bigrams  
- Removed English stop words  

The result was a sparse matrix where each debate is represented by a weighted vector of term importance.

---

## 🔍 K-Means Clustering and Silhouette Score

To discover hidden patterns in the debates, I applied **K-Means clustering**, an algorithm that groups documents based on their textual similarity.  
Because the number of clusters (*k*) was not known in advance, I tested multiple values and evaluated them using the **silhouette score** — a metric that indicates how well each text fits into its assigned cluster compared to others.  

The value of *k* with the highest silhouette score was chosen as optimal.  
After clustering, I analyzed the **top keywords** for each group to interpret the main discussion themes represented within them.

---

## 🎨 Visualization with PCA

To better understand the structure of the clusters, I used **Principal Component Analysis (PCA)** to reduce the high-dimensional TF-IDF vectors into two components.  
The resulting **2D scatter plot** clearly illustrates how debates form distinct clusters, where each color corresponds to a different thematic group.  
This visualization provides an intuitive overview of the similarities and separations between clusters.

---

##  Summary

1. Cleaned and normalized debate transcripts  
2. Represented text with TF-IDF  
3. Determined optimal cluster number via silhouette score  
4. Applied K-Means for clustering  
5. Used PCA for visualization  

This process revealed meaningful topic structures across political debates and demonstrated how text mining can uncover hidden thematic patterns.
