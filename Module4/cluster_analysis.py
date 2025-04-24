from matplotlib import colors as mcolors
import os
import time
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from adjustText import adjust_text
from matplotlib.colors import ListedColormap

# === STEP 0: Setup ===
titles = [
    "2024 United States presidential election",
    "2020 United States presidential election",
    "2016 United States presidential election",
    "2012 United States presidential election",
    "2008 United States presidential election",
    "2004 United States presidential election",
    "2000 United States presidential election",
    "1996 United States presidential election",
    "1992 United States presidential election",
    "1988 United States presidential election",
    "1984 United States presidential election",
    "1980 United States presidential election",
    "1976 United States presidential election",
    "1972 United States presidential election",
    "1968 United States presidential election",
    "1964 United States presidential election",
    "1960 United States presidential election",
    "1956 United States presidential election",
    "1952 United States presidential election",
    "1948 United States presidential election",
    "1944 United States presidential election",
]

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
os.makedirs("cache", exist_ok=True)

# === STEP 1: Fetch Wikipedia article content (cached) ===
def fetch_wikipedia_intro(title):
    """Fetch and cache the lead intro of a Wikipedia article."""
    safe_title = title.replace(" ", "_").replace("/", "_")
    path = f"cache/{safe_title}.json"

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title
    }

    try:
        response = requests.get(WIKI_API_URL, params=params, timeout=10)
        pages = response.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(text, f)
        return text
    except Exception as e:
        print(f"Error fetching {title}: {e}")
        return ""


print("Fetching Wikipedia article intros...")
contents = []
for title in titles:
    print(f"Fetching: {title}")
    summary = fetch_wikipedia_intro(title)
    contents.append(summary)
    time.sleep(0.5)

# === STEP 2: Generate Sentence-BERT embeddings ===
print("Generating Sentence-BERT embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(contents, show_progress_bar=True)

# === STEP 3: Elbow method (using embeddings) ===
def elbow_method(vectors, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(vectors)
        wcss.append(kmeans.inertia_)
    return wcss


wcss = elbow_method(embeddings)
optimal_k = np.diff(wcss, 2).argmin() + 2

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
plt.axvline(optimal_k, color='red', linestyle='--')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method_graph.png", dpi=300)
print(f"Optimal number of clusters: {optimal_k}")

# === STEP 4: Agglomerative Clustering with cosine distance ===
distance_matrix = cosine_distances(embeddings)
cluster_model = AgglomerativeClustering(
    n_clusters=optimal_k, metric='precomputed', linkage='average')
labels = cluster_model.fit_predict(distance_matrix)

# === STEP 5: Save cluster labels ===
df = pd.DataFrame({
    "Article Title": titles,
    "Cluster Label": labels
})
df.to_csv("clustered_wikipedia_articles.csv", index=False)
print("Saved: clustered_wikipedia_articles.csv")

# === STEP 6: Top TF-IDF terms per cluster ===
def top_terms_per_cluster(texts, labels, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    results = []
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        cluster_mean = tfidf_matrix[idx].mean(axis=0)
        top_indices = np.argsort(cluster_mean.A1)[::-1][:top_n]
        top_terms = terms[top_indices]
        results.append((cluster_id, ", ".join(top_terms)))
    return pd.DataFrame(results, columns=["Cluster", "Top Terms"])


terms_df = top_terms_per_cluster(contents, labels)
terms_df.to_csv("top_terms_per_cluster.csv", index=False)
print("Saved: top_terms_per_cluster.csv")

# === STEP 7: PCA visualization ===
reduced = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(12, 7))

# Use bright, distinct Tableau colors
distinct_colors = list(mcolors.TABLEAU_COLORS.values())  # 10 bright colors

# If more clusters than colors, raise an error or define more distinct colors manually
unique_labels = sorted(np.unique(labels))
if len(unique_labels) > len(distinct_colors):
    raise ValueError(f"Only {len(distinct_colors)} distinct colors available, "
                     f"but you have {len(unique_labels)} clusters. Please add more colors.")

# Assign consistent, distinct colors
color_map = {label: distinct_colors[i]
             for i, label in enumerate(unique_labels)}
point_colors = [color_map[label] for label in labels]

# Plot with consistent, high-contrast colors
scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                      c=point_colors, s=100, edgecolors='k')

texts = []
for i, title in enumerate(titles):
    label = f"{title}\n(Cluster {labels[i]})"
    texts.append(plt.text(
        reduced[i, 0], reduced[i, 1], label,
        fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    ))

adjust_text(
    texts,
    only_move={'points': 'y', 'texts': 'xy'},
    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5)
)

plt.title("Agglomerative Clustering of Wikipedia Articles on U.S. Presidential Elections (1944 - 2024)")
plt.xlabel("Principal Component 1 (Historical → Modern Framing)")
plt.ylabel("Principal Component 2 (Policy Focus → Conflict Focus)")
plt.grid(True)
plt.tight_layout()
plt.savefig("pres_election_clusters.png", dpi=300)
