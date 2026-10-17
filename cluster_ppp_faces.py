import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\openparliament-scraper\Images\Religious"

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "PPP_face_embeddings"
)

EMBEDDINGS_FILE = os.path.join(
    EMBEDDING_DIR,
    "PPP_embeddings.npy"
)

METADATA_FILE = os.path.join(
    EMBEDDING_DIR,
    "PPP_embedding_metadata.csv"
)

OUTPUT_FILE = os.path.join(
    EMBEDDING_DIR,
    "PPP_face_clusters.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)
metadata = pd.read_csv(METADATA_FILE)

print(f"Embeddings loaded: {len(embeddings)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")

# Keep only rows that actually have embeddings
metadata_ok = metadata[metadata["status"] == "OK"].copy()

if len(metadata_ok) != len(embeddings):
    raise ValueError(
        f"Mismatch: metadata has {len(metadata_ok)} OK rows "
        f"but embeddings has {len(embeddings)} rows."
    )

# ============================================================
# RECOVER POLITICIAN NAME FROM FILENAME
# ============================================================

metadata_ok["politician"] = (
    metadata_ok["Image.filename"]
    .str.split("__")
    .str[0]
)

# ============================================================
# IMPORTANT:
# We are NOT clustering all politicians together.
#
# We cluster each politician's embeddings separately.
# ============================================================

results = []

politicians = sorted(metadata_ok["politician"].unique())

print(f"\nNumber of politicians found: {len(politicians)}")

# ============================================================
# CLUSTER EACH POLITICIAN
# ============================================================

for i, politician in enumerate(politicians, start=1):

    print(
        f"\n[{i}/{len(politicians)}] "
        f"{politician}"
    )

    # Get rows belonging to this politician
    person_rows = metadata_ok[
        metadata_ok["politician"] == politician
    ].copy()

    # Get their corresponding embeddings
    indices = person_rows["embedding_index"].astype(int).values
    person_embeddings = embeddings[indices]

    print(f"  Usable images: {len(person_embeddings)}")

    # --------------------------------------------------------
    # If too few images, don't cluster
    # --------------------------------------------------------

    if len(person_embeddings) < 5:

        person_rows["cluster"] = -1
        person_rows["cluster_size"] = len(person_embeddings)
        person_rows["is_dominant_cluster"] = True

        results.append(person_rows)
        continue

    # --------------------------------------------------------
    # DBSCAN using cosine distance
    #
    # eps = maximum cosine distance for two points
    # to be considered neighbors.
    #
    # min_samples = minimum number of nearby observations
    # needed to form a cluster.
    # --------------------------------------------------------

    clustering = DBSCAN(
        eps=0.30,
        min_samples=5,
        metric="cosine"
    )

    labels = clustering.fit_predict(person_embeddings)

    person_rows["cluster"] = labels

    # --------------------------------------------------------
    # Determine size of each cluster
    # -1 means DBSCAN considered it noise
    # --------------------------------------------------------

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .to_dict()
    )

    person_rows["cluster_size"] = (
        person_rows["cluster"]
        .map(cluster_counts)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Identify dominant cluster
    #
    # Ignore -1 (noise) when possible.
    # --------------------------------------------------------

    valid_clusters = {
        cluster: count
        for cluster, count in cluster_counts.items()
        if cluster != -1
    }

    if valid_clusters:

        dominant_cluster = max(
            valid_clusters,
            key=valid_clusters.get
        )

        person_rows["is_dominant_cluster"] = (
            person_rows["cluster"] == dominant_cluster
        )

        print(
            f"  Dominant cluster: {dominant_cluster} "
            f"({valid_clusters[dominant_cluster]} images)"
        )

    else:

        person_rows["is_dominant_cluster"] = False

        print("  No coherent cluster found.")

    results.append(person_rows)

# ============================================================
# COMBINE RESULTS
# ============================================================

final = pd.concat(results, ignore_index=True)

# Put useful columns first
columns = [
    "Image.filename",
    "politician",
    "status",
    "embedding_index",
    "face_count",
    "det_score",
    "cluster",
    "cluster_size",
    "is_dominant_cluster"
]

final = final[columns]

# ============================================================
# SAVE
# ============================================================

final.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("CLUSTERING COMPLETE")
print("========================================")

print(f"Rows saved: {len(final)}")
print(f"Output:")
print(OUTPUT_FILE)

print("\nDominant-cluster summary:")

summary = (
    final[final["is_dominant_cluster"]]
    .groupby("politician")
    .size()
    .sort_values(ascending=False)
)

print(summary)