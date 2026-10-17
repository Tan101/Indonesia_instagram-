import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis
import cv2


# ============================================================
# PATHS
# ============================================================

# PPP solo-image Excel
SOLO_EXCEL = (
    r"C:\openparliament-scraper\Images\Religious"
    r"\PPP_face_embed_solo.xlsx"
)

# PPP image folder
IMAGE_DIR = (
    r"C:\openparliament-scraper\Images\Religious\PPP"
)

# Everything produced by this script goes here
OUTPUT_DIR = (
    r"C:\openparliament-scraper\Images\Religious"
    r"\PPP_face_embeddings"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDDINGS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "PPP_embeddings.npy"
)

METADATA_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "PPP_embedding_metadata.csv"
)


# ============================================================
# LOAD PPP SOLO IMAGE LIST
# ============================================================

print("=" * 60)
print("PPP FACE EMBEDDING EXTRACTION")
print("=" * 60)

print("\nLoading PPP solo Excel...")

df = pd.read_excel(SOLO_EXCEL)

if "Image.filename" not in df.columns:
    raise ValueError(
        "Image.filename column not found in PPP solo Excel."
    )

print(f"Total solo images: {len(df):,}")


# ============================================================
# INITIALIZE INSIGHTFACE
# ============================================================

print("\nLoading InsightFace buffalo_l...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("InsightFace ready.")


# ============================================================
# PROCESS IMAGES
# ============================================================

embeddings = []
metadata = []

print("\nProcessing PPP solo images...\n")

for i, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="PPP solo images"
):

    filename = str(row["Image.filename"]).strip()

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    # --------------------------------------------------------
    # IMAGE NOT FOUND
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        metadata.append({
            "Image.filename": filename,
            "status": "IMAGE_NOT_FOUND",
            "embedding_index": np.nan,
            "face_count": 0,
            "det_score": np.nan
        })

        continue

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    img = cv2.imread(image_path)

    if img is None:

        metadata.append({
            "Image.filename": filename,
            "status": "IMAGE_READ_ERROR",
            "embedding_index": np.nan,
            "face_count": 0,
            "det_score": np.nan
        })

        continue

    # --------------------------------------------------------
    # DETECT FACES
    # --------------------------------------------------------

    try:
        faces = app.get(img)

    except Exception as e:

        print(f"\nError processing {filename}: {e}")

        metadata.append({
            "Image.filename": filename,
            "status": "ERROR",
            "embedding_index": np.nan,
            "face_count": 0,
            "det_score": np.nan
        })

        continue

    face_count = len(faces)

    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    if face_count == 0:

        metadata.append({
            "Image.filename": filename,
            "status": "NO_FACE",
            "embedding_index": np.nan,
            "face_count": 0,
            "det_score": np.nan
        })

        continue

    # --------------------------------------------------------
    # MULTIPLE FACES
    # --------------------------------------------------------

    if face_count > 1:

        metadata.append({
            "Image.filename": filename,
            "status": "MULTIPLE_FACES",
            "embedding_index": np.nan,
            "face_count": face_count,
            "det_score": np.nan
        })

        continue

    # --------------------------------------------------------
    # EXACTLY ONE FACE
    # --------------------------------------------------------

    face = faces[0]
    embedding = face.embedding

    if embedding is None:

        metadata.append({
            "Image.filename": filename,
            "status": "NO_EMBEDDING",
            "embedding_index": np.nan,
            "face_count": 1,
            "det_score": float(face.det_score)
        })

        continue

    # --------------------------------------------------------
    # SAVE EMBEDDING IN MEMORY
    # --------------------------------------------------------

    embedding_index = len(embeddings)
    embeddings.append(embedding)

    metadata.append({
        "Image.filename": filename,
        "status": "OK",
        "embedding_index": embedding_index,
        "face_count": 1,
        "det_score": float(face.det_score)
    })

    # --------------------------------------------------------
    # PROGRESS MESSAGE
    # --------------------------------------------------------

    if (i + 1) % 100 == 0:

        print(
            f"\n[{i + 1:,}/{len(df):,}] processed | "
            f"usable embeddings: {len(embeddings):,}"
        )


# ============================================================
# CONVERT EMBEDDINGS TO NUMPY ARRAY
# ============================================================

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    EMBEDDINGS_OUTPUT,
    embeddings
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(
    METADATA_OUTPUT,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PPP EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

print(f"Total Excel rows:       {len(df):,}")
print(f"Usable embeddings:      {len(embeddings):,}")

for status in [
    "NO_FACE",
    "MULTIPLE_FACES",
    "IMAGE_NOT_FOUND",
    "IMAGE_READ_ERROR",
    "NO_EMBEDDING",
    "ERROR"
]:
    count = (metadata_df["status"] == status).sum()
    print(f"{status + ':':24s}{count:,}")

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nSaved files:")
print(EMBEDDINGS_OUTPUT)
print(METADATA_OUTPUT)

print("\nAll PPP embedding files are stored in:")
print(OUTPUT_DIR)

print("\nDone.")