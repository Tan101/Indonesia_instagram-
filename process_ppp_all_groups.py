import os
import pandas as pd
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# PATHS
# ============================================================

EXCEL_PATH = r"C:\openparliament-scraper\Pilot sample 9000\ppp_codes_flashlite.xlsx"

IMAGE_DIR = r"C:\openparliament-scraper\Images\Religious\PPP"

REFERENCE_DIR = (
    r"C:\openparliament-scraper\Images\Religious"
    r"\PPP_face_embeddings\PPP_identity_references"
)

VALIDATION_DIR = (
    r"C:\openparliament-scraper\Images\Religious"
    r"\PPP_face_embeddings\group_identity_test"
)

OUTPUT_DIR = (
    r"C:\openparliament-scraper\Images\Religious"
    r"\PPP_face_embeddings\PPP_full_group_identity"
)

ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_images")

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "PPP_full_group_identity_results.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)


# ============================================================
# LOAD EXCEL
# ============================================================

print("\nLoading Excel file...")

df = pd.read_excel(EXCEL_PATH)

print(f"Total rows in Excel: {len(df):,}")

if "group_type" not in df.columns:
    raise ValueError(
        "ERROR: 'group_type' column was not found in the Excel file."
    )

if "Image.filename" not in df.columns:
    raise ValueError(
        "ERROR: 'Image.filename' column was not found."
    )

group_df = df[
    df["group_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "group"
].copy()

print(f"Group images in Excel: {len(group_df):,}")


# ============================================================
# LOAD ALREADY-VALIDATED FILENAMES
# ============================================================

print("\nLoading previously validated images...")

already_done = set()

validation_files = [
    os.path.join(
        VALIDATION_DIR,
        "group_identity_validation.csv"
    ),
    os.path.join(
        VALIDATION_DIR,
        "three_politicians_identity_validation.csv"
    )
]

for validation_file in validation_files:

    if os.path.exists(validation_file):

        temp = pd.read_csv(validation_file)

        if "Image.filename" in temp.columns:

            names = (
                temp["Image.filename"]
                .dropna()
                .astype(str)
                .tolist()
            )

            already_done.update(names)

            print(
                f"Loaded {len(names):,} already-tested images "
                f"from {os.path.basename(validation_file)}"
            )

        else:
            print(
                f"WARNING: Image.filename not found in "
                f"{validation_file}"
            )

    else:
        print(
            f"WARNING: validation file not found:\n"
            f"{validation_file}"
        )


# ============================================================
# REMOVE ALREADY-VALIDATED IMAGES
# ============================================================

group_df["already_validated"] = (
    group_df["Image.filename"]
    .astype(str)
    .isin(already_done)
)

remaining_df = group_df[
    ~group_df["already_validated"]
].copy()

print("\n============================================")
print("PROCESSING SUMMARY")
print("============================================")
print(f"Total Excel rows:              {len(df):,}")
print(f"Group images:                  {len(group_df):,}")
print(f"Already validated:             {len(group_df[group_df['already_validated']]):,}")
print(f"Remaining to process:          {len(remaining_df):,}")
print("============================================\n")


# ============================================================
# LOAD REFERENCE PROFILES
# ============================================================

print("Loading politician reference profiles...")

reference_profiles = {}

for filename in os.listdir(REFERENCE_DIR):

    if not filename.endswith(".npz"):
        continue

    politician = os.path.splitext(filename)[0]

    path = os.path.join(
        REFERENCE_DIR,
        filename
    )

    data = np.load(path)

    references = data["embeddings"]
    centroid = data["centroid"]

    reference_profiles[politician] = {
        "references": references,
        "centroid": centroid
    }

print(
    f"Reference profiles loaded: "
    f"{len(reference_profiles)}"
)

print(
    "Politicians with references:"
)

for politician in sorted(reference_profiles):
    print(f"  - {politician}")


# ============================================================
# INITIALIZE INSIGHTFACE
# ============================================================

print("\nLoading InsightFace...")

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
# HELPER FUNCTIONS
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / (a_norm * b_norm)
    )


def politician_from_filename(filename):

    return str(filename).split("__")[0]


def annotate_image(
    image_path,
    faces,
    best_face_index,
    best_similarity,
    politician,
    output_path
):

    image = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            28
        )
    except:
        font = ImageFont.load_default()

    for i, face in enumerate(faces):

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        if i == best_face_index:

            width = 8

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=width
            )

            label = (
                f"POLITICIAN CANDIDATE | "
                f"{politician} | "
                f"sim={best_similarity:.3f}"
            )

            draw.text(
                (x1, max(0, y1 - 35)),
                label,
                fill="red",
                font=font
            )

        else:

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="yellow",
                width=3
            )

            label = f"FACE {i}"

            draw.text(
                (x1, max(0, y1 - 30)),
                label,
                fill="yellow",
                font=font
            )

    image.save(
        output_path,
        quality=95
    )


# ============================================================
# PROCESS IMAGES
# ============================================================

results = []

total = len(remaining_df)

print("\nStarting full group-image processing...\n")

for counter, (_, row) in enumerate(
    remaining_df.iterrows(),
    start=1
):

    filename = str(
        row["Image.filename"]
    )

    politician = politician_from_filename(
        filename
    )

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    result = {
        "Image.filename": filename,
        "politician": politician,
        "group_type": "Group",
        "status": None,
        "face_count": None,
        "best_face_index": None,
        "best_similarity": None,
        "top5_similarity": None,
        "centroid_similarity": None,
        "det_score": None,
        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None
    }

    # --------------------------------------------------------
    # Check reference profile
    # --------------------------------------------------------

    if politician not in reference_profiles:

        result["status"] = "NO_REFERENCE"

        results.append(result)

        print(
            f"[{counter:,}/{total:,}] "
            f"{filename} -> NO_REFERENCE"
        )

        continue


    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        result["status"] = "IMAGE_NOT_FOUND"

        results.append(result)

        print(
            f"[{counter:,}/{total:,}] "
            f"{filename} -> IMAGE_NOT_FOUND"
        )

        continue


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    try:

        import cv2

        image = cv2.imread(
            image_path
        )

        if image is None:

            result["status"] = "IMAGE_READ_ERROR"

            results.append(result)

            continue

    except Exception as e:

        result["status"] = "IMAGE_READ_ERROR"

        results.append(result)

        continue


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    try:

        faces = app.get(image)

    except Exception as e:

        result["status"] = "FACE_DETECTION_ERROR"

        results.append(result)

        print(
            f"[{counter:,}/{total:,}] "
            f"{filename} -> FACE_DETECTION_ERROR"
        )

        continue


    result["face_count"] = len(faces)


    if len(faces) == 0:

        result["status"] = "NO_FACE"

        results.append(result)

        print(
            f"[{counter:,}/{total:,}] "
            f"{filename} -> NO_FACE"
        )

        continue


    # --------------------------------------------------------
    # Get politician references
    # --------------------------------------------------------

    profile = reference_profiles[
        politician
    ]

    references = profile["references"]

    centroid = profile["centroid"]


    # --------------------------------------------------------
    # Compare every detected face against THIS politician
    # --------------------------------------------------------

    face_scores = []

    for face_index, face in enumerate(faces):

        embedding = face.embedding

        # Best similarity against any reference image
        similarities = [
            cosine_similarity(
                embedding,
                reference
            )
            for reference in references
        ]

        similarities = np.asarray(
            similarities
        )

        best_similarity = float(
            np.max(similarities)
        )

        # Mean of top 5 references
        top_n = min(
            5,
            len(similarities)
        )

        top5_similarity = float(
            np.mean(
                np.sort(similarities)[-top_n:]
            )
        )

        centroid_similarity = cosine_similarity(
            embedding,
            centroid
        )

        face_scores.append({
            "face_index": face_index,
            "best_similarity": best_similarity,
            "top5_similarity": top5_similarity,
            "centroid_similarity": centroid_similarity,
            "det_score": float(
                face.det_score
            )
        })


    # --------------------------------------------------------
    # Select best matching face
    # --------------------------------------------------------

    best = max(
        face_scores,
        key=lambda x: x["best_similarity"]
    )

    best_face_index = best["face_index"]

    best_face = faces[
        best_face_index
    ]

    bbox = best_face.bbox.astype(int)

    x1, y1, x2, y2 = bbox


    result["status"] = "CANDIDATE"
    result["best_face_index"] = best_face_index
    result["best_similarity"] = best[
        "best_similarity"
    ]
    result["top5_similarity"] = best[
        "top5_similarity"
    ]
    result["centroid_similarity"] = best[
        "centroid_similarity"
    ]
    result["det_score"] = best[
        "det_score"
    ]

    result["bbox_x1"] = int(x1)
    result["bbox_y1"] = int(y1)
    result["bbox_x2"] = int(x2)
    result["bbox_y2"] = int(y2)


    # --------------------------------------------------------
    # Create annotated image
    # --------------------------------------------------------

    output_image = os.path.join(
        ANNOTATED_DIR,
        filename
    )

    try:

        annotate_image(
            image_path=image_path,
            faces=faces,
            best_face_index=best_face_index,
            best_similarity=best[
                "best_similarity"
            ],
            politician=politician,
            output_path=output_image
        )

    except Exception as e:

        print(
            f"WARNING: annotation failed for "
            f"{filename}: {e}"
        )


    results.append(result)


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if counter % 25 == 0:

        print(
            f"[{counter:,}/{total:,}] "
            f"processed"
        )

        pd.DataFrame(
            results
        ).to_csv(
            RESULTS_CSV,
            index=False
        )


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)

print("\n============================================")
print("FINISHED")
print("============================================")
print(
    f"Images processed: {len(results_df):,}"
)

print(
    "\nStatus counts:"
)

print(
    results_df["status"]
    .value_counts()
)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_CSV}"
)

print(
    f"\nAnnotated images saved to:\n"
    f"{ANNOTATED_DIR}"
)

print("============================================")