# -*- coding: utf-8 -*-
import os, sys, numpy as np, cv2, faiss
from insightface.app import FaceAnalysis
from src.backend.config import Paths, Thresholds

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

test_image = r"image/ihan/IMG_4352.jpg"

index = faiss.read_index(Paths.FAISS_INDEX)
labels = np.load(Paths.FAISS_LABELS, allow_pickle=True)

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(832, 832))
img = cv2.imread(test_image)
if img is None:
    print(f"?대?吏 濡쒕뱶 ?ㅽ뙣: {test_image}")
    raise SystemExit(1)
faces = app.get(img)
if not faces:
    print("?쇨뎬 誘명깘吏")
    raise SystemExit(0)
face = faces[0]
emb = face.normed_embedding
emb = emb / np.linalg.norm(emb)
D, I = index.search(np.array([emb]).astype('float32'), 1)
sim = 1 - D[0][0]
name = labels[I[0][0]]
print(f"Top-1: {name}, similarity: {sim:.3f}, pass: {sim >= Thresholds.SIMILARITY}")
