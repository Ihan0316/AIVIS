# -*- coding: utf-8 -*-
"""
Video Face Recognition Test (Person Crop + buffalo_l)
Threshold: 0.70 (higher = stricter matching)
"""
import os
import sys
import cv2
import numpy as np
import time

# NVIDIA library PATH
nvidia_paths = [
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\cublas\bin",
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\cudnn\bin",
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\cufft\bin",
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\cusparse\bin",
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\cusolver\bin",
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\curand\bin",
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\cuda_runtime\bin",
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\cuda_nvrtc\bin",
    r"C:\Users\ihan\anaconda3\envs\aivis-gpu\lib\site-packages\nvidia\nvjitlink\bin",
]
for p in nvidia_paths:
    if os.path.exists(p):
        os.environ['PATH'] = p + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(p)
        except:
            pass

from ultralytics import YOLO
from insightface.app import FaceAnalysis
import faiss
from PIL import Image, ImageDraw, ImageFont

print("=" * 70)
print("Video Face Recognition Test (Person Crop + buffalo_l)")
print("Threshold: 0.70 (strict matching)")
print("=" * 70)

# Load models
model_dir = r"C:\Users\ihan\Desktop\aivis-project-mac-windows\model"

print("\n[1] Loading models...")

# Pose model (Person detection)
pose_model = YOLO(os.path.join(model_dir, "yolo11n-pose.engine"), task='pose')
print("   Pose model loaded")

# buffalo_l (Face detection + embedding)
face_analyzer = FaceAnalysis(
    name='buffalo_l',
    providers=[
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ],
    allowed_modules=['detection', 'recognition']
)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
print("   buffalo_l loaded")

# FAISS load
faiss_path = r"C:\Users\ihan\Desktop\aivis-project-mac-windows\face\data\face_index.faiss"
labels_path = r"C:\Users\ihan\Desktop\aivis-project-mac-windows\face\data\face_index.faiss.labels.npy"

faiss_index = faiss.read_index(faiss_path)
faiss_labels = np.load(labels_path, allow_pickle=True)
print(f"   FAISS loaded: {faiss_index.ntotal} embeddings, {len(set(faiss_labels))} persons")

# Korean font
font_path = r"C:\Windows\Fonts\malgun.ttf"
font = ImageFont.truetype(font_path, 24)
font_small = ImageFont.truetype(font_path, 18)

# THRESHOLD - Higher = stricter matching
# IndexFlatIP uses Inner Product = Cosine Similarity for normalized vectors
# D value IS the similarity (not distance!)
SIMILARITY_THRESHOLD = 0.30  # IP value 0.30 = 30% similarity (strict for different people)

def find_best_match(embedding):
    """FAISS search with strict threshold for IndexFlatIP"""
    embedding = embedding.reshape(1, -1).astype('float32')
    D, I = faiss_index.search(embedding, 5)
    
    if I[0][0] == -1:
        return "Unknown", 0.0
    
    best_idx = I[0][0]
    # For IndexFlatIP: D IS the similarity (inner product)
    best_similarity = D[0][0]
    
    # Strict threshold - for IndexFlatIP, higher D = more similar
    if best_similarity < SIMILARITY_THRESHOLD:
        return "Unknown", best_similarity
    
    return faiss_labels[best_idx], best_similarity

def draw_text_korean(img, text, position, color=(0, 255, 0), font_size=24):
    """Draw Korean text"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font if font_size >= 24 else font_small, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Video processing
video_path = r"C:\Users\ihan\Desktop\aivis-project-mac-windows\동영상.mp4"
output_path = r"C:\Users\ihan\Desktop\aivis-project-mac-windows\test_face_output.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"\n[2] Processing video... ({total_frames} frames)")
print(f"   Input: {video_path}")
print(f"   Output: {output_path}")
print(f"   Threshold: {SIMILARITY_THRESHOLD}")

frame_count = 0
recognition_stats = {"total": 0, "recognized": 0, "unknown": 0}
name_counts = {}

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Processing: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
        
        start_time = time.perf_counter()
        
        # 1. Pose model - Person detection
        pose_results = pose_model(frame, verbose=False)
        
        persons = []
        if pose_results and len(pose_results) > 0:
            boxes = pose_results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        persons.append((x1, y1, x2, y2, conf))
        
        # 2. Face recognition for each person (crop method)
        for px1, py1, px2, py2, pconf in persons:
            # Person box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 200, 0), 2)
            
            # Person crop
            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size == 0:
                continue
            
            # buffalo_l face detection
            faces = face_analyzer.get(person_crop)
            
            recognition_stats["total"] += 1
            
            if len(faces) > 0:
                # Select largest face
                best_face = max(faces, key=lambda f: f.det_score)
                
                # Crop coords -> Original coords
                bbox = best_face.bbox.astype(int)
                fx1 = px1 + bbox[0]
                fy1 = py1 + bbox[1]
                fx2 = px1 + bbox[2]
                fy2 = py1 + bbox[3]
                
                # Embedding extraction + FAISS search
                embedding = best_face.embedding
                if embedding is not None:
                    embedding = embedding / np.linalg.norm(embedding)
                    name, similarity = find_best_match(embedding)
                    
                    if name != "Unknown":
                        recognition_stats["recognized"] += 1
                        name_counts[name] = name_counts.get(name, 0) + 1
                        color = (0, 255, 0)  # Green
                    else:
                        recognition_stats["unknown"] += 1
                        color = (0, 165, 255)  # Orange
                    
                    label = f"{name} ({similarity:.2f})"
                    
                    # Face box
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                    
                    # Name label
                    frame = draw_text_korean(frame, label, (fx1, fy1 - 30), color)
        
        # FPS display
        elapsed = (time.perf_counter() - start_time) * 1000
        fps_text = f"FPS: {1000/elapsed:.1f}"
        frame = draw_text_korean(frame, fps_text, (10, 30), (255, 255, 255))
        
        out.write(frame)

except KeyboardInterrupt:
    print("\n   Interrupted")

cap.release()
out.release()

print(f"\n[3] Complete!")
print(f"   Total frames: {frame_count}")
print(f"   Person detections: {recognition_stats['total']}")
print(f"   Recognized: {recognition_stats['recognized']}")
print(f"   Unknown: {recognition_stats['unknown']}")
if recognition_stats['total'] > 0:
    rate = 100 * recognition_stats['recognized'] / recognition_stats['total']
    print(f"   Recognition rate: {rate:.1f}%")
print(f"\n   Name counts:")
for name, count in sorted(name_counts.items(), key=lambda x: -x[1]):
    print(f"     - {name}: {count}")
print(f"\n   Output: {output_path}")
print("=" * 70)