import os
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\openparliament-scraper\Images\Religious"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "PPP"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "PPP_face_embeddings"
)

EMBEDDINGS_FILE = os.path.join(
    EMBEDDING_DIR,
    "PPP_embeddings.npy"
)

CLUSTER_FILE = os.path.join(
    EMBEDDING_DIR,
    "PPP_face_clusters.csv"
)

OUTPUT_DIR = os.path.join(
    EMBEDDING_DIR,
    "cluster_inspection"
)

# Number of images to select per politician
N_IMAGES = 20

# ============================================================
# LOAD DATA
# ============================================================

print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_FILE)

print("Loading cluster information...")
metadata = pd.read_csv(CLUSTER_FILE)

# Only images in the dominant cluster
dominant = metadata[
    metadata["is_dominant_cluster"] == True
].copy()

politicians = sorted(
    dominant["politician"].unique()
)

print(f"\nPoliticians found: {len(politicians)}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PROCESS EACH POLITICIAN
# ============================================================

for i, politician in enumerate(politicians, start=1):

    print(
        f"\n[{i}/{len(politicians)}] "
        f"{politician}"
    )

    person = dominant[
        dominant["politician"] == politician
    ].copy()

    if len(person) == 0:
        print("  No dominant-cluster images.")
        continue

    # --------------------------------------------------------
    # Get embeddings belonging to this politician's
    # dominant cluster
    # --------------------------------------------------------

    indices = (
        person["embedding_index"]
        .astype(int)
        .values
    )

    person_embeddings = embeddings[indices]

    # --------------------------------------------------------
    # Calculate the centroid of the dominant cluster
    # --------------------------------------------------------

    centroid = np.mean(
        person_embeddings,
        axis=0,
        keepdims=True
    )

    # Normalize centroid
    centroid = centroid / np.linalg.norm(
        centroid,
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # Calculate distance from every image to centroid
    # --------------------------------------------------------

    distances = cosine_distances(
        person_embeddings,
        centroid
    ).flatten()

    person["centroid_distance"] = distances

    # --------------------------------------------------------
    # Select the 20 images closest to centroid
    # --------------------------------------------------------

    selected = person.sort_values(
        "centroid_distance"
    ).head(N_IMAGES)

    # --------------------------------------------------------
    # Create politician-specific folder
    # --------------------------------------------------------

    politician_dir = os.path.join(
        OUTPUT_DIR,
        politician
    )

    os.makedirs(
        politician_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    copied = 0

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        filename = row["Image.filename"]

        source = os.path.join(
            IMAGE_DIR,
            filename
        )

        destination = os.path.join(
            politician_dir,
            f"{rank:02d}_{filename}"
        )

        if os.path.exists(source):

            shutil.copy2(
                source,
                destination
            )

            copied += 1

    # --------------------------------------------------------
    # Save a small CSV showing what was selected
    # --------------------------------------------------------

    selected_file = os.path.join(
        politician_dir,
        "selected_images.csv"
    )

    selected[
        [
            "Image.filename",
            "embedding_index",
            "cluster",
            "cluster_size",
            "centroid_distance",
            "det_score"
        ]
    ].to_csv(
        selected_file,
        index=False
    )

    print(
        f"  Dominant cluster: "
        f"{len(person)} images"
    )

    print(
        f"  Copied for inspection: "
        f"{copied}"
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("INSPECTION SET COMPLETE")
print("========================================")

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nEach politician has up to 20 images.")
print("These are the images closest to their")
print("dominant face-cluster centroid.")