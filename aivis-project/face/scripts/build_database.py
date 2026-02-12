import os
import sys
import platform
import cv2
import numpy as np
import onnxruntime
from ultralytics import YOLO  # YOLOv8n-Face: 키포인트 제공 + 3배 빠름!
import time
import faiss  # Faiss 임포트

# config 임포트 경로 수정
base_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
parent_dir = os.path.dirname(base_dir)  # final/
backend_dir = os.path.join(parent_dir, 'src', 'backend')

# sys.path에 경로 추가
sys.path.insert(0, os.path.abspath(backend_dir))
sys.path.insert(0, os.path.abspath(parent_dir))

# config 임포트 (경로 설정용)
try:
    import config
except ImportError:
    # config가 없으면 기본 경로 사용
    config = None

# --- 설정 ---
# data/images를 참조하도록 경로 수정
base_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
parent_dir = os.path.dirname(base_dir)  # face/
project_root = os.path.dirname(parent_dir)  # aivis-project/ (프로젝트 루트)

# 얼굴 이미지 데이터베이스 경로 (PPE 합성 이미지 포함)
# new_faces 폴더 우선 확인 → data/images → image 순서로 확인
new_faces_dir = os.path.join(parent_dir, "data", "new_faces")  # new_faces 폴더
image_dir = os.path.join(parent_dir, "image")  # 원본 이미지 폴더
data_images_dir = os.path.join(parent_dir, "data", "images")  # PPE 합성 이미지 포함 폴더

# new_faces 폴더 처리: new_faces의 이미지를 data/images로 이동
if os.path.exists(new_faces_dir) and os.listdir(new_faces_dir):
    print(f"📁 new_faces 폴더 발견: {new_faces_dir}")
    print(f"   → data/images로 이동하여 처리합니다...")
    
    # data/images 폴더 생성
    os.makedirs(data_images_dir, exist_ok=True)
    
    # new_faces의 각 작업자 폴더를 data/images로 이동
    import shutil
    moved_count = 0
    for person_folder in os.listdir(new_faces_dir):
        person_path = os.path.join(new_faces_dir, person_folder)
        if os.path.isdir(person_path):
            dest_path = os.path.join(data_images_dir, person_folder)
            if os.path.exists(dest_path):
                # 이미 존재하면 파일만 이동
                for filename in os.listdir(person_path):
                    src_file = os.path.join(person_path, filename)
                    if os.path.isfile(src_file):
                        dest_file = os.path.join(dest_path, filename)
                        if not os.path.exists(dest_file):
                            shutil.move(src_file, dest_file)
                            moved_count += 1
                # 빈 폴더 삭제
                try:
                    if not os.listdir(person_path):
                        os.rmdir(person_path)
                except:
                    pass
            else:
                # 폴더 전체 이동
                shutil.move(person_path, dest_path)
                moved_count += 1
                print(f"   ✅ '{person_folder}' 폴더 이동 완료")
    
    if moved_count > 0:
        print(f"   ✅ 총 {moved_count}개 폴더/파일 이동 완료")

# DB_PATH 우선순위: data/images > image > new_faces
if os.path.exists(data_images_dir) and os.listdir(data_images_dir):
    DB_PATH = data_images_dir  # PPE 합성 이미지 포함 폴더 사용 (원본 + PPE 합성 모두 포함)
    print(f"✅ PPE 합성 이미지 포함 폴더 사용: {DB_PATH}")
elif os.path.exists(image_dir) and os.listdir(image_dir):
    DB_PATH = image_dir  # 원본 이미지 폴더 사용 (PPE 합성 없을 때)
    print(f"✅ 원본 이미지 폴더 사용: {DB_PATH}")
elif os.path.exists(new_faces_dir) and os.listdir(new_faces_dir):
    DB_PATH = new_faces_dir  # new_faces 폴더 직접 사용 (폴백)
    print(f"✅ new_faces 폴더 사용: {DB_PATH}")
else:
    DB_PATH = data_images_dir  # 폴백
    print(f"✅ 복사된 이미지 폴더 사용: {DB_PATH}")

# FAISS 파일 저장 경로: face/data 폴더에 저장 (백엔드와 통일)
# face/data: face/data/face_index.faiss, face/data/face_index.faiss.labels.npy
face_data_dir = os.path.join(parent_dir, "data")  # face/data 폴더
os.makedirs(face_data_dir, exist_ok=True)
FAISS_INDEX_FILE = os.path.join(face_data_dir, "face_index.faiss")  # face/data에 저장
FAISS_LABELS_FILE = os.path.join(face_data_dir, "face_index.faiss.labels.npy")  # face/data에 저장

# 백업용 원본 임베딩 (face/data/embeddings에 저장 - 참고용)
OUTPUT_EMBEDDINGS = os.path.join(parent_dir, "data", "embeddings", "face_embeddings.npy")  # 백업용 원본 (절대 경로)
embeddings_dir = os.path.join(parent_dir, "data", "embeddings")
os.makedirs(embeddings_dir, exist_ok=True)

# 프로젝트 루트에 FAISS 파일 저장을 위한 디렉토리 확인
os.makedirs(project_root, exist_ok=True)

# 증분 업데이트 설정
INCREMENTAL_UPDATE = False  # True: 새 이미지만 추가, False: 전체 재구축 (최고 성능 재생성)
PROCESSED_IMAGES_FILE = "processed_images.txt"  # 처리된 이미지 기록 파일

# PPE 합성 자동 실행 설정
AUTO_PPE_SYNTHESIS = True  # True: new_faces에서 이동한 이미지에 대해 자동으로 PPE 합성 수행

# CCTV 환경 최적화 설정
USE_SMART_AUGMENTATION = True  # True: 스마트 증강 (품질에 따라 선택적), False: 모든 증강
AUGMENTATION_MODE = "full"  # "full": 12가지, "balanced": 8가지, "fast": 5가지 (속도 우선)
# ⭐ full 모드: 최고 품질 (12가지 증강, 인식률 최대화) - 권장!
# balanced 모드: 품질과 속도 균형 (8가지 증강, 약 30-40% 시간 단축 예상)
# fast 모드: 속도 우선 (5가지 증강, 약 50-60% 시간 단축 예상)

# 증강 이미지 저장 설정 (시각화/결과 정리용)
SAVE_AUGMENTED_IMAGES = True  # True: 증강된 이미지 저장, False: 저장 안 함
AUGMENTED_IMAGES_DIR = os.path.join(parent_dir, "data", "augmented")  # 증강 이미지 저장 경로

# 대표 임베딩(센트로이드) 사용 설정
# ⭐ 등록 인원이 적으면(~10명) 모든 임베딩 저장이 인식률 향상에 유리!
USE_REPRESENTATIVE_EMBEDDING = False  # False: 모든 임베딩 저장 (인식률 최대화)
STORE_CENTROID_ONLY = False           # (USE_REPRESENTATIVE_EMBEDDING=False면 무시됨)
TOP_N_PER_PERSON = 15                 # (USE_REPRESENTATIVE_EMBEDDING=False면 무시됨)

# 품질 향상 옵션
ENABLE_CLAHE = True                   # 조명 표준화 (Y 채널 CLAHE)
ENABLE_TTA_FLIP = True                # Test-Time Augmentation: 좌우 플립 평균
MIN_BLUR_VARIANCE = 40.0              # 흐림(블러) 임계값 (Variance of Laplacian)
MIN_BRIGHTNESS = 20.0                 # 최소 밝기 (0~255, 평균)
MAX_BRIGHTNESS = 235.0                # 최대 밝기 (0~255, 평균)

def is_low_quality(img: np.ndarray) -> bool:
    """간단한 품질 체크: 블러/밝기 기준으로 저품질 판정"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 블러 측정
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 밝기 측정
        mean_brightness = float(np.mean(gray))
        if var < MIN_BLUR_VARIANCE:
            return True
        if mean_brightness < MIN_BRIGHTNESS or mean_brightness > MAX_BRIGHTNESS:
            return True
        return False
    except Exception:
        return False

def apply_clahe_bgr(img: np.ndarray) -> np.ndarray:
    """Y 채널 CLAHE로 조명 표준화"""
    if img is None or img.size == 0:
        return img
    try:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        ycrcb = cv2.merge([y, cr, cb])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return out
    except Exception:
        return img

def extract_embedding_with_tta(rec_model, face_img: np.ndarray) -> np.ndarray:
    """
    rec_model.get_feat()에 대해 TTA(flip) 적용 후 평균 임베딩 반환.
    반환은 float32 512차원, L2 정규화됨.
    
    ⚠️ 주의: 이 함수는 InsightFace 모델용입니다. AdaFace는 FastIndustrialRecognizer를 사용합니다.
    """
    try:
        embs = []
        # 원본
        emb0 = rec_model.get_feat(face_img)
        if emb0 is not None:
            embs.append(emb0.astype(np.float32))
        # 좌우 플립
        if ENABLE_TTA_FLIP:
            flipped = cv2.flip(face_img, 1)
            emb1 = rec_model.get_feat(flipped)
            if emb1 is not None:
                embs.append(emb1.astype(np.float32))
        if not embs:
            return None
        embs = np.array(embs, dtype=np.float32)
        avg = np.mean(embs, axis=0)
        # L2 정규화
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = (avg / norm).astype(np.float32)
        return avg
    except Exception:
        return None


def extract_embedding_adaface_tta(fast_recognizer, frame: np.ndarray, kps: np.ndarray, face_analyzer=None) -> tuple:
    """
    AdaFace 모델에 TTA(flip) 적용 후 평균 임베딩 반환.
    실시간 인식과 동일한 전처리 방식 사용.
    반환: (embedding, aligned_face) - embedding은 float32 512차원, L2 정규화됨
    """
    try:
        embs = []
        aligned_face = None
        
        # 원본 임베딩
        result = fast_recognizer.get_embedding_fast(frame, kps, face_analyzer=face_analyzer)
        if result is not None and len(result) >= 2:
            emb0, aligned_face = result[0], result[1]
            if emb0 is not None and isinstance(emb0, np.ndarray):
                embs.append(emb0.astype(np.float32))
        
        # 좌우 플립 임베딩
        if ENABLE_TTA_FLIP:
            flipped_frame = cv2.flip(frame, 1)
            # 키포인트도 좌우 반전 (x 좌표 반전)
            h, w = frame.shape[:2]
            flipped_kps = kps.copy()
            flipped_kps[:, 0] = w - flipped_kps[:, 0]
            # 왼쪽/오른쪽 눈, 입꼬리 스왑 (0<->1, 3<->4)
            flipped_kps[[0, 1]] = flipped_kps[[1, 0]]
            flipped_kps[[3, 4]] = flipped_kps[[4, 3]]
            
            result = fast_recognizer.get_embedding_fast(flipped_frame, flipped_kps, face_analyzer=face_analyzer)
            if result is not None and len(result) >= 2:
                emb1, _ = result[0], result[1]
                if emb1 is not None and isinstance(emb1, np.ndarray):
                    embs.append(emb1.astype(np.float32))
        
        if not embs:
            return None, None
        
        embs = np.array(embs, dtype=np.float32)
        avg = np.mean(embs, axis=0)
        
        # L2 정규화
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = (avg / norm).astype(np.float32)
        return avg, aligned_face
        
    except Exception as e:
        print(f"  ⚠️ AdaFace TTA 오류: {e}")
        return None, None


def extract_rotated_embeddings(fast_recognizer, aligned_face: np.ndarray) -> list:
    """
    정렬된 얼굴(112x112)을 90도 회전시켜서 추가 임베딩 생성.
    넘어진 사람 인식을 위한 증강.
    반환: 회전된 임베딩 리스트 [(embedding, rotation_label), ...]
    """
    rotated_embeddings = []
    
    if aligned_face is None or aligned_face.size == 0:
        return rotated_embeddings
    
    try:
        # 시계방향 90도 회전 (오른쪽으로 넘어짐)
        rotated_cw = cv2.rotate(aligned_face, cv2.ROTATE_90_CLOCKWISE)
        # TensorRT 우선, ONNX 폴백
        emb_cw = None
        if fast_recognizer.use_tensorrt:
            emb_cw = fast_recognizer._get_embedding_from_tensorrt(rotated_cw)
        if emb_cw is None and fast_recognizer.use_direct_onnx:
            emb_cw = fast_recognizer._get_embedding_from_onnx(rotated_cw)
        if emb_cw is not None:
            norm_val = np.linalg.norm(emb_cw)
            if norm_val > 0:
                emb_cw = (emb_cw / norm_val).astype(np.float32)
                rotated_embeddings.append((emb_cw, "rotate_90_cw"))
        
        # 반시계방향 90도 회전 (왼쪽으로 넘어짐)
        rotated_ccw = cv2.rotate(aligned_face, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # TensorRT 우선, ONNX 폴백
        emb_ccw = None
        if fast_recognizer.use_tensorrt:
            emb_ccw = fast_recognizer._get_embedding_from_tensorrt(rotated_ccw)
        if emb_ccw is None and fast_recognizer.use_direct_onnx:
            emb_ccw = fast_recognizer._get_embedding_from_onnx(rotated_ccw)
        if emb_ccw is not None:
            norm_val = np.linalg.norm(emb_ccw)
            if norm_val > 0:
                emb_ccw = (emb_ccw / norm_val).astype(np.float32)
                rotated_embeddings.append((emb_ccw, "rotate_90_ccw"))
        
    except Exception as e:
        print(f"  ⚠️ 90도 회전 임베딩 추출 오류: {e}")
    
    return rotated_embeddings


def load_processed_images():
    """이미 처리된 이미지 목록을 로드합니다."""
    if not os.path.exists(PROCESSED_IMAGES_FILE):
        return set()
    
    processed_set = set()
    base_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
    parent_dir = os.path.dirname(base_dir)  # project root/
    
    with open(PROCESSED_IMAGES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 상대 경로를 절대 경로로 변환
            if line.startswith('../'):
                abs_path = os.path.normpath(os.path.join(base_dir, line))
            else:
                abs_path = os.path.normpath(line)
            processed_set.add(abs_path)
    
    return processed_set


def save_processed_image(image_path):
    """처리된 이미지 경로를 기록합니다."""
    with open(PROCESSED_IMAGES_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{image_path}\n")


def update_processed_images(image_paths):
    """여러 이미지를 일괄 기록합니다."""
    with open(PROCESSED_IMAGES_FILE, 'a', encoding='utf-8') as f:
        for path in image_paths:
            f.write(f"{path}\n")


def create_augmentation_grid(images, labels, output_path, max_size=200):
    """증강 이미지들을 그리드 형태로 합쳐서 저장합니다."""
    if not images:
        return
    
    # 이미지 크기 조정 (그리드용)
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img.copy()
        resized_images.append(resized)
    
    # 그리드 크기 계산 (가로 4개씩)
    cols = 4
    rows = (len(resized_images) + cols - 1) // cols
    
    # 각 이미지 크기 통일
    target_h = max_size
    target_w = max_size
    
    # 빈 그리드 이미지 생성
    grid_img = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    
    # 이미지들을 그리드에 배치
    for idx, (img, label) in enumerate(zip(resized_images, labels)):
        row = idx // cols
        col = idx % cols
        
        h, w = img.shape[:2]
        y_offset = (target_h - h) // 2
        x_offset = (target_w - w) // 2
        
        # 이미지 배치
        grid_img[row * target_h + y_offset:row * target_h + y_offset + h,
                 col * target_w + x_offset:col * target_w + x_offset + w] = img
        
        # 레이블 텍스트 추가
        cv2.putText(grid_img, label, 
                   (col * target_w + 5, row * target_h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 그리드 이미지 저장 (한글 경로 지원)
    try:
        # cv2.imwrite는 한글 경로를 제대로 처리하지 못하므로 imencode + tofile 사용
        ext = os.path.splitext(output_path)[1]
        result, encoded_img = cv2.imencode(ext, grid_img)
        if result:
            encoded_img.tofile(output_path)
    except Exception as e:
        # 폴백: 일반 imwrite 시도
        cv2.imwrite(output_path, grid_img)


def load_existing_database():
    """기존 FAISS 인덱스와 라벨을 로드합니다."""
    # face/data 폴더에서 FAISS 파일 로드 (백엔드와 통일)
    index_path = FAISS_INDEX_FILE
    labels_path = FAISS_LABELS_FILE
    
    # face/data에 없으면 프로젝트 루트도 확인 (하위 호환성)
    if not os.path.exists(index_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        project_index = os.path.join(project_root, "face_index.faiss")
        project_labels = os.path.join(project_root, "face_index.faiss.labels.npy")
        if os.path.exists(project_index):
            index_path = project_index
            labels_path = project_labels
            print(f"✅ 기존 인덱스 발견 (하위 호환성): {index_path}")
    
    # 프로젝트 루트에도 없으면 face/data/embeddings 경로도 확인 (더 오래된 버전 호환성)
    if not os.path.exists(index_path):
        embeddings_dir = os.path.join(parent_dir, "data", "embeddings")
        old_index_path = os.path.join(embeddings_dir, "face_index.faiss")
        old_labels_path = os.path.join(embeddings_dir, "face_index.faiss.labels.npy")
        if os.path.exists(old_index_path):
            index_path = old_index_path
            labels_path = old_labels_path
            print(f"✅ 기존 인덱스 발견 (하위 호환성): {index_path}")
    
    if not os.path.exists(index_path) or not os.path.exists(labels_path):
        return None, None, set()
    
    try:
        index = faiss.read_index(index_path)
        labels = np.load(labels_path, allow_pickle=True)
        
        # 기존 처리된 이미지 목록 로드
        processed = load_processed_images()
        
        print(f"✅ 기존 인덱스 로드 완료: {index.ntotal}개 임베딩, {len(processed)}개 이미지 처리됨 (경로: {index_path})")
        return index, labels, processed
    except Exception as e:
        print(f"⚠️ 기존 인덱스 로드 실패: {e}")
        return None, None, set()


def build_database():
    """
    DB_PATH에 있는 모든 이미지로부터 얼굴 특징(임베딩)을 추출하여 Faiss 인덱스를 생성합니다.
    각 이미지에 대해 원본, 좌우 반전, 밝기 조절 등 데이터 증강을 적용합니다.
    new_faces 폴더의 이미지는 자동으로 data/images로 이동하고, PPE 합성을 수행합니다.
    """
    # Windows 콘솔 유니코드 출력 대응
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    # PPE 합성 자동 실행 (new_faces에서 이동한 이미지에 대해)
    if AUTO_PPE_SYNTHESIS and DB_PATH == data_images_dir:
        print("\n" + "=" * 70)
        print("🛠️ PPE 합성 자동 실행")
        print("=" * 70)
        try:
            # ppe_synthesis_and_embedding.py의 PPE 합성 함수 임포트
            ppe_script_path = os.path.join(os.path.dirname(__file__), "ppe_synthesis_and_embedding.py")
            if os.path.exists(ppe_script_path):
                # PPE 합성 스크립트를 서브프로세스로 실행
                import subprocess
                print(f"   PPE 합성 스크립트 실행: {ppe_script_path}")
                result = subprocess.run(
                    [sys.executable, ppe_script_path],
                    cwd=os.path.dirname(ppe_script_path),
                    capture_output=False,  # 실시간 출력을 위해 False
                    text=True
                )
                if result.returncode == 0:
                    print("✅ PPE 합성 완료")
                else:
                    print(f"⚠️ PPE 합성 실패 (계속 진행): returncode={result.returncode}")
            else:
                print(f"⚠️ PPE 합성 스크립트를 찾을 수 없습니다: {ppe_script_path}")
                print("   (계속 진행합니다)")
        except Exception as e:
            print(f"⚠️ PPE 합성 실행 중 오류 (계속 진행): {e}")
            import traceback
            traceback.print_exc()
        print("=" * 70 + "\n")

    print("모델을 로딩합니다. 몇 초 정도 소요될 수 있습니다...")
    
    # 1. YOLOv8n-Face 모델 로드 (Mac MPS 지원)
    import torch
    # Mac MPS 디바이스 확인
    use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    device = 'mps' if use_mps else 'cpu'
    
    yolo_face_engine_path = os.path.join(project_root, "model", "yolov8n-face.engine")
    yolo_face_pt_path = os.path.join(project_root, "model", "yolov8n-face.pt")
    
    if os.path.exists(yolo_face_pt_path):
        print(f"YOLOv8n-Face PT 모델 로딩 중: {yolo_face_pt_path} (디바이스: {device})")
        yolo_face_model = YOLO(yolo_face_pt_path)
        # YOLO 모델은 device 파라미터를 추론 시에 전달하므로 여기서는 로드만
        print(f"✅ YOLOv8n-Face PT 로드 완료 (추론 시 {device.upper()} 사용)")
    elif os.path.exists(yolo_face_engine_path):
        # TensorRT 엔진은 Mac에서 작동하지 않지만, 폴백으로 시도
        print(f"⚠️ TensorRT 엔진은 Mac에서 지원되지 않습니다. PT 모델을 사용하세요.")
        print(f"YOLOv8n-Face TensorRT 엔진 로딩 시도: {yolo_face_engine_path}")
        yolo_face_model = YOLO(yolo_face_engine_path, task='pose')
        print(f"✅ YOLOv8n-Face TensorRT 로드 완료")
    else:
        print(f"❌ YOLOv8n-Face 모델 파일을 찾을 수 없습니다")
        print(f"   확인 경로: {yolo_face_engine_path} 또는 {yolo_face_pt_path}")
        return
    print(f"✅ 모델 로딩 완료")
    
    # 2. AdaFace 모델 (임베딩 추출용) - TensorRT 엔진 우선
    adaface_engine_path = os.path.join(project_root, "model", "adaface_ir50_ms1mv2.engine")
    adaface_onnx_path = os.path.join(project_root, "model", "adaface_ir50_ms1mv2.onnx")
    
    # FastIndustrialRecognizer는 .onnx 경로를 받으면 자동으로 .engine을 찾음
    # 하지만 .engine만 있는 경우를 위해 .engine 경로 직접 전달
    if os.path.exists(adaface_engine_path):
        adaface_model_path = adaface_engine_path
        print(f"✅ AdaFace TensorRT 엔진 사용: {adaface_model_path}")
    elif os.path.exists(adaface_onnx_path):
        adaface_model_path = adaface_onnx_path
        print(f"✅ AdaFace ONNX 모델 사용: {adaface_model_path}")
    else:
        print(f"❌ AdaFace 모델 파일을 찾을 수 없습니다")
        print(f"   확인 경로: {adaface_engine_path} 또는 {adaface_onnx_path}")
        return
    
    # 3. FastIndustrialRecognizer 초기화 (임베딩 추출)
    fast_recognizer = None
    try:
        backend_dir = os.path.join(project_root, "src", "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        from fast_face_recognizer import FastIndustrialRecognizer
        fast_recognizer = FastIndustrialRecognizer(
            model_path=adaface_model_path,
            ctx_id=0,  # GPU 사용 (TensorRT 우선)
            use_adaface=True
        )
        print(f"✅ FastIndustrialRecognizer 초기화 완료 (AdaFace TensorRT/GPU)")
    except Exception as e:
        print(f"❌ FastIndustrialRecognizer 초기화 실패: {e}")
        return
    
    print(f"✅ 모델 로딩 완료 (YOLOv8n-Face 키포인트 + AdaFace 임베딩 - 실시간 시스템과 동일!)")
    
    # 증강 이미지 저장 폴더 생성
    if SAVE_AUGMENTED_IMAGES:
        os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)
        print(f"📸 증강 이미지 저장 활성화: {AUGMENTED_IMAGES_DIR}")
    
    # 증분 업데이트 모드 확인
    if INCREMENTAL_UPDATE:
        index, existing_labels, processed_images = load_existing_database()
        if index is not None and existing_labels is not None:
            print(f"📌 증분 업데이트 모드: 기존 {index.ntotal}개 임베딩 유지")
        else:
            print("📌 새 인덱스 생성 모드: 전체 재구축")
            index = None
            existing_labels = None
            processed_images = set()
    else:
        print("📌 전체 재구축 모드")
        index = None
        existing_labels = None
        processed_images = set()

    # DB의 모든 이미지를 처리하여 임베딩 추출
    face_database = {}  # 임시 저장용
    start_time = time.time()
    processed_files_count = 0
    new_files_count = 0
    embedding_count = 0
    new_image_paths = []
    # 얼굴 감지 실패 추적
    person_face_detection_stats = {}  # {person_name: {'total_images': 0, 'faces_found': 0, 'faces_not_found': 0}}

    # os.walk를 사용하여 하위 폴더까지 모두 탐색
    for root, dirs, files in os.walk(DB_PATH):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue

        for file in image_files:
            image_path = os.path.join(root, file)
            # 절대 경로로 변환
            image_path = os.path.abspath(os.path.normpath(image_path))
            
            # 증분 업데이트 모드에서 이미 처리된 이미지는 건너뛰기
            if image_path in processed_images:
                continue
            
            person_name = os.path.basename(root)  # 폴더 이름을 사람 이름으로 사용
            new_files_count += 1
            new_image_paths.append(image_path)
            
            # 통계 초기화
            if person_name not in person_face_detection_stats:
                person_face_detection_stats[person_name] = {'total_images': 0, 'faces_found': 0, 'faces_not_found': 0, 'low_quality': 0}
            person_face_detection_stats[person_name]['total_images'] += 1

            print(f"처리 중: {image_path} (원본 + 증강 3종)")

            # OpenCV로 이미지 읽기 (한글 경로 지원)
            # cv2.imread는 한글 경로를 제대로 처리하지 못하므로 numpy로 먼저 읽음
            try:
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                img = None
                print(f"  [경고] 이미지 디코딩 실패: {e}")
            
            if img is None:
                # 폴백: 일반 imread 시도
                img = cv2.imread(image_path)
            
            if img is None:
                print(f"  [경고] 이미지를 읽을 수 없습니다: {image_path}")
                continue

            # ⭐ 실시간 카메라와 동일한 해상도로 리사이즈 (640x480)
            # 비율 유지하면서 640x480에 맞추고 검정 패딩
            TARGET_W, TARGET_H = 640, 480
            h, w = img.shape[:2]
            scale = min(TARGET_W / w, TARGET_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 검정 배경에 중앙 배치
            canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
            x_offset = (TARGET_W - new_w) // 2
            y_offset = (TARGET_H - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            img = canvas
            print(f"  ✅ 640x480 리사이즈 완료 (원본: {w}x{h} → {new_w}x{new_h}, 패딩 적용)")

            processed_files_count += 1
            new_files_count += 1
            new_image_paths.append(image_path)

            # --- [CCTV 최적화] 산업현장 CCTV 환경에 맞춘 데이터 증강 (스마트 모드) ---
            # 처리할 이미지들을 리스트에 담습니다.
            images_to_process = []
            augmentation_labels = []  # 증강 타입 레이블 (저장용)

            # 항상 포함: 원본과 좌우반전 (핵심)
            images_to_process.append(img)
            augmentation_labels.append("original")
            images_to_process.append(cv2.flip(img, 1))
            augmentation_labels.append("flip")
            
            # ⭐ 90도 회전 추가 (넘어진 사람 인식용)
            # 시계방향 90도 (오른쪽으로 넘어짐)
            rotated_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            images_to_process.append(rotated_cw)
            augmentation_labels.append("rotate_90_cw")
            
            # 반시계방향 90도 (왼쪽으로 넘어짐)
            rotated_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            images_to_process.append(rotated_ccw)
            augmentation_labels.append("rotate_90_ccw")
            
            # 증강 모드에 따라 선택적 적용
            if AUGMENTATION_MODE == "full":
                # 전체 증강 (12가지) - 최고 품질, 느림
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=25))
                augmentation_labels.append("bright_+25")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=45))
                augmentation_labels.append("bright_+45")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=-25))
                augmentation_labels.append("bright_-25")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=-45))
                augmentation_labels.append("bright_-45")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.15, beta=0))
                augmentation_labels.append("contrast_+1.15")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=0.85, beta=0))
                augmentation_labels.append("contrast_0.85")
                blurred = cv2.GaussianBlur(img, (3, 3), 0)
                images_to_process.append(blurred)
                augmentation_labels.append("blur")
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(img, -1, kernel)
                images_to_process.append(sharpened)
                augmentation_labels.append("sharpen")
                h, w = img.shape[:2]
                # 카메라 해상도 시뮬레이션 (640 기준) - 실시간 인식과 동일 조건
                if h > 400 or w > 400:
                    # 640 픽셀 기준으로 다운스케일 (카메라 시뮬레이션)
                    target_size = 640
                    scale = target_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    downscaled_cam = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    upscaled_cam = cv2.resize(downscaled_cam, (w, h), interpolation=cv2.INTER_LINEAR)
                    images_to_process.append(upscaled_cam)
                    augmentation_labels.append("cam_sim_640")
                    
                    # 추가: 480 픽셀 기준 (더 저해상도)
                    scale_480 = 480 / max(h, w)
                    new_h_480, new_w_480 = int(h * scale_480), int(w * scale_480)
                    downscaled_480 = cv2.resize(img, (new_w_480, new_h_480), interpolation=cv2.INTER_AREA)
                    upscaled_480 = cv2.resize(downscaled_480, (w, h), interpolation=cv2.INTER_LINEAR)
                    images_to_process.append(upscaled_480)
                    augmentation_labels.append("cam_sim_480")
                if len(img.shape) == 2:
                    equalized = cv2.equalizeHist(img)
                else:
                    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                images_to_process.append(equalized)
                augmentation_labels.append("hist_equal")
                
                # ===== 🔆 추가 조명 증강 (9가지) =====
                
                # 1. 감마 보정 (실내/실외 조명 차이)
                # 감마 < 1: 밝게 (실외/창가)
                # 감마 > 1: 어둡게 (실내/그늘)
                for gamma_val, gamma_label in [(0.7, "gamma_0.7"), (1.5, "gamma_1.5"), (2.0, "gamma_2.0")]:
                    inv_gamma = 1.0 / gamma_val
                    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    gamma_img = cv2.LUT(img, table)
                    images_to_process.append(gamma_img)
                    augmentation_labels.append(gamma_label)
                
                # 2. 색온도 변경 (형광등/백열등/자연광)
                # 따뜻한 색온도 (백열등, 2700K) - R 증가, B 감소
                warm_img = img.copy()
                warm_img[:,:,2] = np.clip(warm_img[:,:,2] * 1.1, 0, 255).astype(np.uint8)  # R 증가
                warm_img[:,:,0] = np.clip(warm_img[:,:,0] * 0.9, 0, 255).astype(np.uint8)  # B 감소
                images_to_process.append(warm_img)
                augmentation_labels.append("color_warm")
                
                # 차가운 색온도 (형광등, 6500K) - B 증가, R 감소
                cool_img = img.copy()
                cool_img[:,:,0] = np.clip(cool_img[:,:,0] * 1.15, 0, 255).astype(np.uint8)  # B 증가
                cool_img[:,:,2] = np.clip(cool_img[:,:,2] * 0.85, 0, 255).astype(np.uint8)  # R 감소
                images_to_process.append(cool_img)
                augmentation_labels.append("color_cool")
                
                # 3. 그림자 합성 (부분 조명 - 좌/우 그림자)
                h, w = img.shape[:2]
                # 왼쪽 그림자 (왼쪽이 어두움)
                shadow_left = img.copy().astype(np.float32)
                for col in range(w // 2):
                    factor = 0.5 + (col / (w // 2)) * 0.5  # 0.5 ~ 1.0
                    shadow_left[:, col] = shadow_left[:, col] * factor
                shadow_left = np.clip(shadow_left, 0, 255).astype(np.uint8)
                images_to_process.append(shadow_left)
                augmentation_labels.append("shadow_left")
                
                # 오른쪽 그림자 (오른쪽이 어두움)
                shadow_right = img.copy().astype(np.float32)
                for col in range(w // 2, w):
                    factor = 1.0 - ((col - w // 2) / (w // 2)) * 0.5  # 1.0 ~ 0.5
                    shadow_right[:, col] = shadow_right[:, col] * factor
                shadow_right = np.clip(shadow_right, 0, 255).astype(np.uint8)
                images_to_process.append(shadow_right)
                augmentation_labels.append("shadow_right")
                
                # 4. 노이즈 추가 (저조도 시뮬레이션)
                # 가우시안 노이즈 (저조도 카메라 시뮬레이션)
                noise_img = img.copy().astype(np.float32)
                noise = np.random.normal(0, 15, noise_img.shape).astype(np.float32)  # 표준편차 15
                noise_img = np.clip(noise_img + noise, 0, 255).astype(np.uint8)
                images_to_process.append(noise_img)
                augmentation_labels.append("noise_low_light")
                
                # 더 심한 노이즈 (매우 어두운 환경)
                noise_img_heavy = img.copy().astype(np.float32)
                noise_heavy = np.random.normal(0, 25, noise_img_heavy.shape).astype(np.float32)  # 표준편차 25
                noise_img_heavy = np.clip(noise_img_heavy + noise_heavy, 0, 255).astype(np.uint8)
                images_to_process.append(noise_img_heavy)
                augmentation_labels.append("noise_very_low")
                # ===== 추가 조명 증강 완료 =====
                
            elif AUGMENTATION_MODE == "balanced":
                # 균형 모드 (8가지) - 품질과 속도 균형
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=30))  # 밝기 증가
                augmentation_labels.append("bright_+30")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=-30))  # 밝기 감소
                augmentation_labels.append("bright_-30")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.2, beta=0))  # 대비 증가
                augmentation_labels.append("contrast_+1.2")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=0.8, beta=0))  # 대비 감소
                augmentation_labels.append("contrast_0.8")
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(img, -1, kernel)
                images_to_process.append(sharpened)
                augmentation_labels.append("sharpen")
                if len(img.shape) == 2:
                    equalized = cv2.equalizeHist(img)
                else:
                    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                images_to_process.append(equalized)
                augmentation_labels.append("hist_equal")
                
            elif AUGMENTATION_MODE == "fast":
                # 빠른 모드 (5가지) - 속도 우선
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=30))
                augmentation_labels.append("bright_+30")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=-30))
                augmentation_labels.append("bright_-30")
                images_to_process.append(cv2.convertScaleAbs(img, alpha=1.2, beta=0))
                augmentation_labels.append("contrast_+1.2")
            # --- [CCTV 최적화 완료] ---
            
            # 증강 이미지 저장 (시각화용)
            if SAVE_AUGMENTED_IMAGES:
                # 증강 이미지 저장 폴더 생성
                person_aug_dir = os.path.join(AUGMENTED_IMAGES_DIR, person_name)
                os.makedirs(person_aug_dir, exist_ok=True)
                
                # 원본 이미지 파일명 (확장자 제거)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # 각 증강 이미지 저장
                for aug_img, aug_label in zip(images_to_process, augmentation_labels):
                    aug_filename = f"{base_name}_{aug_label}.jpg"
                    aug_filepath = os.path.join(person_aug_dir, aug_filename)
                    # 한글 경로 지원
                    try:
                        result, encoded_img = cv2.imencode('.jpg', aug_img)
                        if result:
                            encoded_img.tofile(aug_filepath)
                    except Exception:
                        cv2.imwrite(aug_filepath, aug_img)
                
                # 증강 이미지 그리드 생성 (한눈에 보기)
                try:
                    grid_path = os.path.join(person_aug_dir, f"{base_name}_grid.jpg")
                    create_augmentation_grid(images_to_process, augmentation_labels, grid_path)
                    print(f"  ✅ 증강 이미지 저장 완료: {len(images_to_process)}개 (그리드 포함)")
                except Exception as e:
                    print(f"  [경고] 증강 그리드 생성 실패: {e}")

            # 원본 및 증강된 모든 이미지에서 특징 추출 (InsightFace 사용)
            # ⭐ InsightFace로 얼굴 감지 → 랜드마크 추출 → AdaFace로 임베딩 추출
            face_found_in_any_augmentation = False
            for aug_idx, augmented_img in enumerate(images_to_process):
                aug_label = augmentation_labels[aug_idx] if aug_idx < len(augmentation_labels) else "unknown"
                
                # ⭐ YOLOv8n-Face로 얼굴 감지 (키포인트 포함! + 3배 빠름)
                face_bbox = None
                kps_for_adaface = None
                try:
                    # conf 임계값을 낮춰서 더 많은 얼굴 감지 시도 (0.1로 낮춤)
                    # Mac MPS 사용 시 device 파라미터 추가
                    yolo_results = yolo_face_model(augmented_img, conf=0.1, verbose=False, device=device)
                    if yolo_results and len(yolo_results) > 0:
                        result = yolo_results[0]
                        if result.boxes is not None and len(result.boxes) > 0:
                            # 가장 큰 얼굴 선택
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else None
                            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                            best_idx = np.argmax(areas)
                            
                            fx1, fy1, fx2, fy2 = boxes[best_idx].astype(int)
                            face_bbox = (fx1, fy1, fx2, fy2)
                            face_found_in_any_augmentation = True
                            
                            # 디버깅: 얼굴 감지 성공 로그 (원본 이미지에서만)
                            if aug_label == "original":
                                conf_str = f", conf={confidences[best_idx]:.3f}" if confidences is not None else ""
                                print(f"  ✅ 얼굴 감지 성공 ({aug_label}): bbox=({fx1},{fy1},{fx2},{fy2}){conf_str}")
                            
                            # YOLOv8n-Face 키포인트 추출 (5개 포인트!)
                            if result.keypoints is not None and len(result.keypoints) > best_idx:
                                kps = result.keypoints[best_idx].xy.cpu().numpy()
                                # kps shape: (1, 5, 2) - 첫번째 차원 제거
                                if kps is not None:
                                    if len(kps.shape) == 3:
                                        kps = kps[0]  # (1, 5, 2) -> (5, 2)
                                    if len(kps) >= 5:
                                        kps_for_adaface = kps[:5].astype(np.float32)
                except Exception as e:
                    if aug_label == "original":
                        print(f"  ⚠️ 얼굴 감지 오류 ({aug_label}): {e}")
                
                if face_bbox is None:
                    # 얼굴을 찾지 못한 경우
                    continue
                
                # 랜드마크 기반 임베딩 추출 (FastIndustrialRecognizer + AdaFace)
                embedding = None
                aligned_face_for_rotation = None
                
                try:
                    if kps_for_adaface is not None and fast_recognizer is not None:
                        try:
                            # AdaFace + TTA (Test-Time Augmentation) 적용
                            # 원본 + 좌우반전 평균으로 더 안정적인 임베딩 생성
                            embedding, aligned_face_for_rotation = extract_embedding_adaface_tta(
                                fast_recognizer, 
                                augmented_img, 
                                kps_for_adaface, 
                                face_analyzer=None
                            )
                            if embedding is not None:
                                print(f"  ✅ AdaFace+TTA 임베딩 추출 성공 ({aug_label})")
                        except Exception as e:
                            print(f"  ⚠️ AdaFace+TTA 처리 실패: {e}")
                            import traceback
                            traceback.print_exc()
                            embedding = None
                            aligned_face_for_rotation = None
                    
                    # 랜드마크가 없는 경우 - 품질 필터 적용 후 스킵
                    if embedding is None:
                        fx1, fy1, fx2, fy2 = face_bbox
                        img_h, img_w = augmented_img.shape[:2]
                        fx1 = max(0, fx1)
                        fy1 = max(0, fy1)
                        fx2 = min(img_w, fx2)
                        fy2 = min(img_h, fy2)
                        face_img = augmented_img[fy1:fy2, fx1:fx2]
                        if face_img.size > 0 and is_low_quality(face_img):
                            person_face_detection_stats[person_name]['low_quality'] += 1
                        continue
                            
                except Exception as e:
                    print(f"  ⚠️ 임베딩 추출 오류: {e}")
                    continue
                
                # embedding이 성공적으로 생성된 경우에만 저장
                if embedding is not None:
                    # face_database 딕셔너리에 저장
                    if person_name not in face_database:
                        face_database[person_name] = []
                    face_database[person_name].append(embedding)
                    embedding_count += 1
                    person_face_detection_stats[person_name]['faces_found'] += 1
                    
                    # ⭐ 90도 회전 임베딩 추가 (넘어진 사람 인식용)
                    # 원본 이미지에서만 90도 회전 임베딩 생성 (증강 이미지에서는 생략)
                    if aug_label == "original" and aligned_face_for_rotation is not None:
                        rotated_embs = extract_rotated_embeddings(fast_recognizer, aligned_face_for_rotation)
                        for rot_emb, rot_label in rotated_embs:
                            face_database[person_name].append(rot_emb)
                            embedding_count += 1
                            print(f"  ✅ 90도 회전 임베딩 추가: {rot_label}")
            
            # 이미지 처리 후 얼굴 감지 실패 여부 확인
            if not face_found_in_any_augmentation:
                person_face_detection_stats[person_name]['faces_not_found'] += 1
                print(f"  ⚠️ {person_name}: 이 이미지에서 얼굴을 찾지 못했습니다: {os.path.basename(image_path)}")

    if new_files_count == 0 and INCREMENTAL_UPDATE:
        print("✅ 새로 처리할 이미지가 없습니다. 인덱스가 최신 상태입니다.")
        # 기존 인덱스에서 임베딩을 로드하여 npy 파일 생성
        if index is not None and index.ntotal > 0:
            print("기존 인덱스에서 임베딩을 로드하여 npy 파일 생성 중...")
            # 기존 인덱스의 모든 임베딩 추출
            all_embeddings = index.reconstruct_n(0, index.ntotal)
            all_labels = existing_labels if existing_labels is not None else []
            
            # 딕셔너리로 변환
            existing_db = {}
            for i, label in enumerate(all_labels):
                if label not in existing_db:
                    existing_db[label] = []
                existing_db[label].append(all_embeddings[i])
            
            # npy 파일 저장
            np.save(OUTPUT_EMBEDDINGS, existing_db)
            print(f"✅ face_embeddings.npy 파일 생성 완료: {OUTPUT_EMBEDDINGS}")
        return

    # 얼굴 감지 통계 출력
    print("\n" + "=" * 70)
    print("📊 얼굴 감지 통계")
    print("=" * 70)
    for person_name, stats in sorted(person_face_detection_stats.items()):
        total = stats['total_images']
        found = stats['faces_found']
        not_found = stats['faces_not_found']
        low_quality = stats['low_quality']
        success_rate = (found / total * 100) if total > 0 else 0
        status = "✅" if found > 0 else "❌"
        print(f"{status} {person_name}: 이미지 {total}개, 얼굴 발견 {found}개, 미발견 {not_found}개, 저품질 {low_quality}개 (성공률: {success_rate:.1f}%)")
    print("=" * 70 + "\n")

    if not face_database:
        print(f"오류: 처리할 새 이미지를 찾을 수 없습니다.")
        print(f"디버깅 정보:")
        print(f"  - 처리한 이미지 파일 수: {new_files_count}개")
        print(f"  - 임베딩 추출 성공: {embedding_count}개")
        print(f"  - face_database에 저장된 인물 수: {len(face_database)}명")
        if new_files_count > 0 and embedding_count == 0:
            print(f"  ⚠️ 이미지는 처리되었지만 얼굴을 찾지 못했습니다.")
            print(f"     이미지 품질이나 얼굴이 명확하게 보이는지 확인해주세요.")
        return

    # 대표 임베딩(센트로이드) 생성으로 품질 향상 및 노이즈 감소
    if USE_REPRESENTATIVE_EMBEDDING:
        refined_database = {}
        for name, embeddings in face_database.items():
            if not embeddings:
                continue
            embs = np.array(embeddings, dtype=np.float32)
            # L2 정규화 재확인
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms

            # 센트로이드 계산 및 정규화
            centroid = embs.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm

            if STORE_CENTROID_ONLY:
                refined_database[name] = [centroid]
            else:
                # 센트로이드와의 유사도 기준 상위 N개 + 센트로이드
                sims = embs @ centroid.astype(np.float32)
                top_idx = np.argsort(-sims)[:TOP_N_PER_PERSON]
                top_embs = embs[top_idx].tolist()
                refined_database[name] = [centroid] + top_embs
        face_database = refined_database

    # Faiss 인덱스 구축/업데이트 로직
    print("DB 구축 완료. Faiss 인덱스를 업데이트합니다...")
    labels_list = []
    embeddings_list = []

    # 딕셔너리를 Faiss가 사용할 리스트로 변환
    for name, embeddings in face_database.items():
        for embedding in embeddings:
            labels_list.append(name)
            embeddings_list.append(embedding)

    if not embeddings_list:
        print("오류: 추출된 임베딩이 없습니다.")
        return

    # 임베딩 배열 정규화 (모든 임베딩을 1차원 배열로 변환)
    normalized_embeddings = []
    for emb in embeddings_list:
        if emb is None:
            continue
        emb = np.array(emb, dtype=np.float32)
        # 1차원 배열로 변환 (512,)
        if emb.ndim > 1:
            emb = emb.flatten()
        # 차원 확인 (512차원)
        if emb.shape[0] != 512:
            print(f"⚠️ 임베딩 차원 오류: {emb.shape}, 건너뜀")
            continue
        # L2 정규화
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        normalized_embeddings.append(emb)
    
    if not normalized_embeddings:
        print("오류: 유효한 임베딩이 없습니다.")
        return
    
    embeddings_array = np.array(normalized_embeddings).astype('float32')
    labels_array = np.array(labels_list)
    
    # shape 확인 및 수정
    if embeddings_array.ndim == 1:
        # 1차원 배열인 경우 (512,) -> (1, 512)로 변환
        embeddings_array = embeddings_array.reshape(1, -1)
    elif embeddings_array.ndim > 2:
        # 3차원 이상인 경우 평탄화
        embeddings_array = embeddings_array.reshape(len(normalized_embeddings), -1)
    
    d = embeddings_array.shape[1]  # 임베딩 차원 (512)

    # 인덱스 처리
    if index is None:
        # 새 인덱스 생성
        print("새 인덱스 생성 중...")
        index = faiss.IndexFlatIP(d)
        index.add(embeddings_array)
        final_labels = labels_array
    else:
        # 기존 인덱스에 추가
        print(f"기존 인덱스에 {len(embeddings_list)}개 임베딩 추가 중...")
        index.add(embeddings_array)
        
        # 라벨 병합
        final_labels = np.concatenate([existing_labels, labels_array])
    
    # Faiss 인덱스와 라벨 배열 저장 (프로젝트 루트에 저장)
    print(f"💾 FAISS 인덱스 저장 중: {FAISS_INDEX_FILE}")
    print(f"💾 FAISS 레이블 저장 중: {FAISS_LABELS_FILE}")
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(FAISS_LABELS_FILE, final_labels)
    print(f"✅ 저장 완료: face/data 폴더에 저장됨")
    
    # 처리된 이미지 목록 업데이트
    update_processed_images(new_image_paths)

    # 기존 .npy 저장 (백업용)
    if os.path.exists(OUTPUT_EMBEDDINGS):
        existing_db = np.load(OUTPUT_EMBEDDINGS, allow_pickle=True).item()
    else:
        existing_db = {}
    
    # 새로운 데이터 병합
    for name, embeddings in face_database.items():
        if name not in existing_db:
            existing_db[name] = []
        existing_db[name].extend(embeddings)
    
    np.save(OUTPUT_EMBEDDINGS, existing_db)

    end_time = time.time()
    print("-" * 30)
    print("✅ Faiss 인덱스 및 데이터베이스 업데이트 완료!")
    print(f"총 처리 시간: {end_time - start_time:.2f}초")
    print(f"처리한 새 이미지 수: {new_files_count}개")
    print(f"새로 추가된 인물 수: {len(face_database)}명")
    print(f"새로 추가된 임베딩 수 (증강 포함): {len(labels_list)}개")
    print(f"인덱스 총 임베딩 수: {index.ntotal}개")
    print(f"인덱스 총 인물 수: {len(np.unique(final_labels))}명")
    print(f"저장된 인덱스: {FAISS_INDEX_FILE}")
    print(f"저장된 라벨: {FAISS_LABELS_FILE}")
    print(f"(참고용) 원본 DB: {OUTPUT_EMBEDDINGS}")
    print("-" * 30)


if __name__ == "__main__":
    build_database()