from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv("data/debates_2022.csv")
# print(data['talk_text'][2])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+[\.,]?\d*\s*%?', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip() 
    return text


##Preprocessing
data["talk_text"] = data["talk_text"].fillna("").astype(str)
data["talk_text"] = data['talk_text'].apply(clean_text)
# print(data['talk_text'][2])

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer="word", stop_words='english', min_df=5, max_df=0.8, max_features=3000, ngram_range=(1,2))
tf_idf_matrix = vectorizer.fit_transform(data["talk_text"])
# row_sums = tf_idf_matrix.sum(axis=1)


# non_zero_indices = np.where(row_sums > 0)[0]
# tf_idf_nonzero = tf_idf_matrix[non_zero_indices]

features = vectorizer.get_feature_names_out()
print(list(features))
print(len(features))


def find_best_k(matrix, k_min=2, k_max=15):
    best_score = -1
    best_k = None
    scores = []

    for k in range(k_min, k_max+1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(matrix)
        score = silhouette_score(matrix, labels, metric='cosine')
        scores.append(score)
        print(f" k = {k}, silhouette score: {score}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Best k: {best_k}, silhouette score: {best_score}")
    return best_k, best_score

best_k, _ = find_best_k(tf_idf_matrix)

kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(tf_idf_matrix)
data['cluster'] = labels


##cluster-wise average tf-idf matrix
cluster_centers = np.zeros((best_k, tf_idf_matrix.shape[1]))
for k in range(best_k):
    cluster_docs = tf_idf_matrix[labels == k]
    cluster_centers[k] = cluster_docs.mean(axis=0)

##top 10 features for each cluster
for i in range(best_k):
    print(f"\nCluster {i}:")
    top_indices = np.argsort(cluster_centers[i])[::-1][:10]
    top_terms = [features[j] for j in top_indices]
    print(", ".join(top_terms))


output_columns = ['talk_text', 'cluster']
data[output_columns].to_csv("clustered_debates_results.csv", index=False)


## Task 3: Dimensionality Reduction
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(tf_idf_matrix.toarray())

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
plt.title("Visualization of Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()



