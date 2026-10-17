import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\openparliament-scraper\Images\Religious"
EMBED_DIR = os.path.join(BASE_DIR, "PPP_face_embeddings")

EMBEDDINGS_FILE = os.path.join(
    EMBED_DIR,
    "PPP_embeddings.npy"
)

CLUSTERS_FILE = os.path.join(
    EMBED_DIR,
    "PPP_face_clusters.csv"
)

OUTPUT_DIR = os.path.join(
    EMBED_DIR,
    "PPP_identity_references"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings loaded: {len(embeddings)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")

print("\nLoading cluster information...")
clusters = pd.read_csv(CLUSTERS_FILE)

print(f"Cluster rows: {len(clusters)}")


# ============================================================
# KEEP ONLY VALIDATED DOMINANT CLUSTERS
# ============================================================

reference_data = clusters[
    clusters["is_dominant_cluster"] == True
].copy()

print(
    f"\nValidated dominant-cluster rows: "
    f"{len(reference_data)}"
)


# ============================================================
# BUILD ONE REFERENCE PROFILE PER POLITICIAN
# ============================================================

politicians = sorted(
    reference_data["politician"].unique()
)

print(
    f"Politicians with validated reference clusters: "
    f"{len(politicians)}"
)

summary = []


for i, politician in enumerate(politicians, start=1):

    print(f"\n[{i}/{len(politicians)}] {politician}")

    politician_data = reference_data[
        reference_data["politician"] == politician
    ].copy()

    # Get embedding rows from the original .npy file
    indices = politician_data[
        "embedding_index"
    ].astype(int).values

    politician_embeddings = embeddings[indices]

    # --------------------------------------------------------
    # Normalize each embedding
    # --------------------------------------------------------

    norms = np.linalg.norm(
        politician_embeddings,
        axis=1,
        keepdims=True
    )

    normalized_embeddings = (
        politician_embeddings / norms
    )

    # --------------------------------------------------------
    # Calculate normalized centroid
    # --------------------------------------------------------

    centroid = normalized_embeddings.mean(axis=0)

    centroid_norm = np.linalg.norm(centroid)

    centroid = centroid / centroid_norm

    # --------------------------------------------------------
    # Save politician reference profile
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{politician}.npz"
    )

    np.savez_compressed(
        output_file,
        embeddings=normalized_embeddings,
        centroid=centroid
    )

    n_reference = len(normalized_embeddings)

    summary.append({
        "politician": politician,
        "n_reference_embeddings": n_reference,
        "reference_file": output_file
    })

    print(
        f"  Reference embeddings: {n_reference}"
    )

    print(
        f"  Saved: {output_file}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = os.path.join(
    OUTPUT_DIR,
    "reference_library_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 50)
print("REFERENCE LIBRARY COMPLETE")
print("=" * 50)

print(
    f"Politicians: {len(summary_df)}"
)

print(
    f"Total reference embeddings: "
    f"{summary_df['n_reference_embeddings'].sum()}"
)

print("\nReference counts:")

print(
    summary_df[
        ["politician", "n_reference_embeddings"]
    ].to_string(index=False)
)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nSummary file:")
print(summary_file)