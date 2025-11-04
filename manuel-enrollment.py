# crop_my_face.py
import os
import cv2
from insightface.app import FaceAnalysis

# === 1. Input image ===
INPUT_PATH = "1.png"            # your photo file
OUTPUT_DIR = "data/gallery/Amine"   # folder for your face
OUTPUT_NAME = "Amine_0000.jpg"      # saved face file name
TARGET_SIZE = (224, 224)            # same as dataset size

# === 2. Load the image ===
img = cv2.imread(INPUT_PATH)
assert img is not None, f"❌ Cannot read input image: {INPUT_PATH}"

# === 3. Initialize the InsightFace model (CPU mode) ===
print("Loading InsightFace models (ArcFace)...")
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256, 256))

# === 4. Detect the face ===
faces = app.get(img)
if len(faces) == 0:
    raise RuntimeError("❌ No face detected in the image!")

# Pick the largest detected face
f = max(faces, key=lambda z: z.bbox[2] * z.bbox[3])
x1, y1, x2, y2 = map(int, f.bbox)
crop = img[y1:y2, x1:x2]

# === 5. Resize and save ===
crop = cv2.resize(crop, TARGET_SIZE)

# Make sure the output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
cv2.imwrite(out_path, crop)
print(f"✅ Saved cropped face: {out_path}")
