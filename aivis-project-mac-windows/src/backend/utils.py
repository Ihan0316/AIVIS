# utils.py (최종 수정본)
import datetime
import logging
import os
from typing import Tuple, Optional, List, Dict, Any, Union
import cv2

# FAISS 조건부 import (conda DLL 충돌 방지)
# venv를 사용하는 경우 venv의 faiss를 우선 사용하도록 함
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logging.warning("⚠️ FAISS를 찾을 수 없습니다. 얼굴 인식 기능이 제한될 수 있습니다.")
except Exception as e:
    # DLL 로드 실패 등 기타 오류 처리
    faiss = None
    FAISS_AVAILABLE = False
    logging.warning(f"⚠️ FAISS 로드 실패 (DLL 오류 가능성): {e}. 얼굴 인식 기능이 제한될 수 있습니다.")
    logging.info("💡 해결 방법: venv에서 faiss-cpu 재설치 또는 conda 경로 제거")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Keypoints

import config


def setup_logging():
    """표준 로깅 시스템을 설정합니다. 모든 로그를 파일에 저장합니다."""
    import logging.handlers
    import sys

    # 로그 폴더 생성
    os.makedirs(config.Paths.LOG_FOLDER, exist_ok=True)

    # 로그 파일 핸들러 설정 (로테이션 포함)
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(config.Paths.LOG_FOLDER, "system.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )

    # 콘솔 핸들러 설정 (터미널에도 출력)
    console_handler = logging.StreamHandler(sys.stdout)

    # 포맷터 설정 (간소화: 시간, 레벨, 메시지만)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 파일용 상세 포맷터
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(formatter)
    
    # ⭐ 콘솔은 INFO 이상만, 파일은 DEBUG 포함 모두 저장
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)

    # 루트 로거 설정 - 모든 로거의 기본 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers.clear()
    
    # 파일과 콘솔 핸들러 추가
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 모든 서브 로거가 루트 로거를 사용하도록 설정
    for logger_name in ['', '__main__', 'core', 'utils', 'frame_processor', 'camera_worker', 
                        'storage_manager', 'database', 'main', 'state', 'pipeline_manager']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

    # 외부 라이브러리 로거 레벨 조정 (너무 많은 로그 방지)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('pymongo.connection').setLevel(logging.WARNING)
    logging.getLogger('pymongo.topology').setLevel(logging.WARNING)
    logging.getLogger('pymongo.serverSelection').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
    logging.getLogger('onnxruntime').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)

    logging.info("로깅 시스템 초기화 완료 (콘솔: INFO, 파일: DEBUG)")


def create_standard_response(data=None, status="success", message="", error_code=None):
    """표준화된 API 응답 형식을 생성합니다."""
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat(),
        "data": data
    }
    if error_code:
        response["error_code"] = error_code
    return response


# --- [수정] OS에 맞는 폰트를 자동으로 로드하도록 개선 ---
try:
    if not os.path.exists(config.Paths.FONT_PATH):
        # OS별 대체 폰트 경로 시도
        import platform
        fallback_fonts = []

        if platform.system() == "Windows":
            # Windows 한글 폰트 대체 목록
            fallback_fonts = [
                "C:/Windows/Fonts/malgunbd.ttf",  # 맑은 고딕 Bold
                "C:/Windows/Fonts/gulim.ttc",      # 굴림
                "C:/Windows/Fonts/batang.ttc",     # 바탕
                "C:/Windows/Fonts/NanumGothic.ttf", # 나눔고딕
                "C:/Windows/Fonts/arial.ttf"       # Arial (한글 지원 안함)
            ]
        elif platform.system() == "Darwin":
            # macOS 대체 폰트
            fallback_fonts = [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/Library/Fonts/NanumGothic.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
            ]
        else:
            # Linux 대체 폰트
            fallback_fonts = [
                "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ]

        font_found = False
        for fallback_font in fallback_fonts:
            if os.path.exists(fallback_font):
                logging.warning(f"기본 폰트({config.Paths.FONT_PATH}) 없음. 대체 폰트 사용: {fallback_font}")
                config.Paths.FONT_PATH = fallback_font
                font_found = True
                break

        if not font_found:
            raise IOError(f"'{config.Paths.FONT_PATH}' 폰트 파일을 찾을 수 없으며, 대체 폰트도 없습니다.")

    KOREAN_FONT = ImageFont.truetype(config.Paths.FONT_PATH, 14)
    logging.info(f"✅ 폰트 로드 성공: {config.Paths.FONT_PATH}")
except IOError as e:
    logging.warning("=" * 80)
    logging.warning(f"⚠️  폰트 로드 실패: {e}")
    logging.warning("⚠️  기본 폰트를 사용합니다. (한글이 깨질 수 있습니다)")
    logging.warning("⚠️  권장: Windows에 한글 폰트(맑은 고딕, 굴림 등)를 설치하세요")
    logging.warning("=" * 80)
    KOREAN_FONT = ImageFont.load_default()
# --- [수정 완료] ---


class TextRenderer:
    """프레임의 모든 텍스트를 한 번에 그려 성능을 최적화하는 클래스."""

    def __init__(self, frame_shape: Tuple[int, int, int]):
        self.text_layer = Image.new("RGBA", (frame_shape[1], frame_shape[0]), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.text_layer)

    def add_text(self, text: str, pos: Tuple[int, int], bgr_color: Tuple[int, int, int]):
        x, y = pos
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])

        try:
            # 텍스트 크기 계산
            text_bbox = self.draw.textbbox((0, 0), text, font=KOREAN_FONT)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        except Exception as e:
            logging.warning(f"텍스트 렌더링 크기 계산 오류: {e} (텍스트: {text})")
            text_w, text_h = 50, 10 # 기본 크기

        # 화면 경계 정보 가져오기
        frame_h, frame_w = self.text_layer.size[1], self.text_layer.size[0]

        # 패딩 증가 (더 넓은 여백)
        padding_x = 6
        padding_y = 4

        # 텍스트 위치를 화면 경계 내로 제한
        # x 좌표 제한 (텍스트가 오른쪽 경계를 넘지 않도록)
        if x + text_w + padding_x * 2 > frame_w:
            x = frame_w - text_w - padding_x * 2
        if x < padding_x:
            x = padding_x

        # y 좌표 제한 (텍스트가 위쪽 경계를 넘지 않도록)
        bg_y1 = y - text_h - padding_y * 2
        if bg_y1 < 0:
            bg_y1 = 0
            y = bg_y1 + text_h + padding_y * 2

        # 배경 사각형 그리기 (둥근 모서리 효과, 더 진한 배경)
        bg_rect = (x - padding_x, bg_y1, x + text_w + padding_x, y)

        # 배경: 색상과 유사한 진한 색 (더 선명하게)
        bg_color = (
            min(255, int(rgb_color[0] * 0.3)),
            min(255, int(rgb_color[1] * 0.3)),
            min(255, int(rgb_color[2] * 0.3)),
            200  # 더 불투명하게 (128 -> 200)
        )
        self.draw.rectangle(bg_rect, fill=bg_color)

        # 텍스트 그리기 (더 밝게)
        try:
            text_color = (*rgb_color, 255)
            self.draw.text((x, bg_y1 + padding_y), text, font=KOREAN_FONT, fill=text_color)
        except Exception as e:
            logging.warning(f"텍스트 렌더링 그리기 오류: {e}")


    def render_on(self, frame: np.ndarray) -> np.ndarray:
        try:
            text_layer_rgba = np.array(self.text_layer)
            alpha_channel = text_layer_rgba[:, :, 3]

            # 알파 채널에 내용이 있는지 확인
            if not np.any(alpha_channel > 0):
                return frame

            y_coords, x_coords = np.where(alpha_channel > 0)

            # 좌표가 비어있는 극단적인 경우 방지
            if len(y_coords) == 0 or len(x_coords) == 0:
                return frame

            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_min, x_max = np.min(x_coords), np.max(x_coords)

            # 프레임 경계를 넘지 않도록 보정
            y_max = int(min(y_max, frame.shape[0] - 1))
            x_max = int(min(x_max, frame.shape[1] - 1))
            y_min = int(max(0, y_min))
            x_min = int(max(0, x_min))

            text_patch_rgba = text_layer_rgba[y_min:y_max + 1, x_min:x_max + 1]
            frame_patch = frame[y_min:y_max + 1, x_min:x_max + 1]

            # 크기 일치 확인
            if text_patch_rgba.shape[:2] != frame_patch.shape[:2]:
                 logging.warning(f"TextRenderer: 패치 크기 불일치! Text={text_patch_rgba.shape}, Frame={frame_patch.shape}. 렌더링 건너뜀.")
                 return frame

            alpha = (text_patch_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
            text_patch_bgr = text_patch_rgba[:, :, :3][:, :, ::-1] # RGBA -> BGR

            blended_patch = (frame_patch * (1 - alpha) + text_patch_bgr * alpha).astype(np.uint8)
            frame[y_min:y_max + 1, x_min:x_max + 1] = blended_patch
            return frame
        except Exception as e:
            logging.error(f"텍스트 렌더링 적용(render_on) 오류: {e}", exc_info=True)
            return frame


def draw_modern_bbox(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    corner_length: int = 20,
    alpha: float = 0.3
) -> None:
    """
    현대적인 스타일의 바운딩 박스 그리기 (모서리 라인 + 반투명 배경)

    Args:
        frame: 프레임
        x1, y1, x2, y2: 바운딩 박스 좌표
        color: BGR 색상
        thickness: 선 두께
        corner_length: 모서리 라인 길이
        alpha: 배경 투명도 (0~1)
    """
    try:
        # 반투명 배경 레이어 제거 (바운딩 박스 내 색상 제거)
        # overlay = frame.copy()
        # cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        # cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 모서리 라인 그리기 (더 굵고 선명하게)
        corner_thickness = thickness + 1

        # 왼쪽 위
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)

        # 오른쪽 위
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)

        # 왼쪽 아래
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)

        # 오른쪽 아래
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)

        # 테두리 라인 (얇게)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    except Exception as e:
        logging.warning(f"현대적 바운딩박스 그리기 오류: {e}")
        # 오류 시 기본 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_fast_bbox(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2
) -> None:
    """
    프로덕션용 빠른 바운딩 박스 그리기 (모서리 라인 제거, 단순 사각형)
    
    Args:
        frame: 프레임
        x1, y1, x2, y2: 바운딩 박스 좌표
        color: BGR 색상
        thickness: 선 두께
    """
    try:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    except Exception as e:
        logging.warning(f"빠른 바운딩박스 그리기 오류: {e}")


def draw_keypoints(
    frame: np.ndarray,
    keypoints: Any,
    confidence_threshold: float = 0.1,
    point_radius: int = 3,
    line_thickness: int = 2
) -> None:
    """
    키포인트를 프레임에 그리기 (COCO 포맷 17개 키포인트)
    
    Args:
        frame: 프레임 (BGR)
        keypoints: Keypoints 객체 (ultralytics 포맷)
        confidence_threshold: 키포인트 신뢰도 임계값
        point_radius: 키포인트 원 반지름
        line_thickness: 연결선 두께
    """
    try:
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return
        
        points = keypoints.xy[0].cpu().numpy()  # (17, 2)
        confidences = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else None  # (17,)
        
        if confidences is None:
            confidences = np.ones(len(points))
        
        # COCO 포맷 키포인트 인덱스
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        
        # 키포인트 연결 정의 (스켈레톤)
        skeleton = [
            # 머리
            (0, 1), (0, 2),  # nose - eyes
            (1, 3), (2, 4),  # eyes - ears
            # 상체
            (5, 6),  # shoulders
            (5, 7), (7, 9),  # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12),  # shoulders - hips
            # 하체
            (11, 12),  # hips
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
        ]
        
        # 키포인트 색상 (BGR)
        keypoint_colors = [
            (255, 255, 255),  # 0: nose - 흰색
            (255, 0, 0), (0, 255, 0),  # 1-2: eyes - 빨강, 초록
            (0, 0, 255), (255, 255, 0),  # 3-4: ears - 파랑, 청록
            (255, 0, 255), (0, 255, 255),  # 5-6: shoulders - 자홍, 노랑
            (128, 0, 128), (128, 128, 0),  # 7-8: elbows - 보라, 올리브
            (255, 165, 0), (0, 128, 255),  # 9-10: wrists - 주황, 하늘색
            (128, 0, 0), (0, 128, 0),  # 11-12: hips - 진한 빨강, 진한 초록
            (0, 0, 128), (128, 128, 128),  # 13-14: knees - 진한 파랑, 회색
            (255, 192, 203), (192, 192, 192),  # 15-16: ankles - 분홍, 은색
        ]
        
        # 연결선 그리기
        for i, (start_idx, end_idx) in enumerate(skeleton):
            if start_idx < len(points) and end_idx < len(points):
                if confidences[start_idx] > confidence_threshold and confidences[end_idx] > confidence_threshold:
                    pt1 = (int(points[start_idx][0]), int(points[start_idx][1]))
                    pt2 = (int(points[end_idx][0]), int(points[end_idx][1]))
                    # 연결선 색상 (중간 색상 사용)
                    line_color = (
                        (keypoint_colors[start_idx][0] + keypoint_colors[end_idx][0]) // 2,
                        (keypoint_colors[start_idx][1] + keypoint_colors[end_idx][1]) // 2,
                        (keypoint_colors[start_idx][2] + keypoint_colors[end_idx][2]) // 2,
                    )
                    cv2.line(frame, pt1, pt2, line_color, line_thickness)
        
        # 키포인트 점 그리기
        for i, (point, conf) in enumerate(zip(points, confidences)):
            if conf > confidence_threshold:
                x, y = int(point[0]), int(point[1])
                color = keypoint_colors[i] if i < len(keypoint_colors) else (255, 255, 255)
                cv2.circle(frame, (x, y), point_radius, color, -1)
                # 신뢰도가 높으면 더 큰 원으로 표시
                if conf > 0.5:
                    cv2.circle(frame, (x, y), point_radius + 1, color, 1)
                    
    except Exception as e:
        logging.warning(f"키포인트 그리기 오류: {e}", exc_info=True)


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    try:
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if inter_area == 0: return 0.0
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception as e:
        logging.warning(f"IOU 계산 오류: {e} (box1={box1}, box2={box2})")
        return 0.0


def calculate_iou_batch(
    boxes1: np.ndarray,  # (N, 4) 형태
    boxes2: np.ndarray   # (M, 4) 형태
) -> np.ndarray:
    """
    여러 박스에 대한 IoU를 배치로 계산합니다.
    
    :param boxes1: (N, 4) 형태의 박스 배열 [x1, y1, x2, y2]
    :param boxes2: (M, 4) 형태의 박스 배열 [x1, y1, x2, y2]
    :return: (N, M) 형태의 IoU 행렬
    """
    try:
        if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
        
        # 박스 영역 계산
        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # 교집합 계산
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 합집합 계산
        union_area = boxes1_area[:, None] + boxes2_area[None, :] - inter_area
        
        # IoU 계산
        iou = inter_area / np.maximum(union_area, 1e-6)
        
        return iou.astype(np.float32)
    except Exception as e:
        logging.warning(f"배치 IoU 계산 오류: {e}")
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)


def find_best_matches_faiss_batch(
    embeddings: np.ndarray, 
    faiss_index: Union[Any, Tuple[Any, Optional[np.ndarray]]],
    threshold: float, 
    labels: Optional[np.ndarray] = None
) -> List[Tuple[str, float]]:
    """
    FAISS 인덱스에서 여러 임베딩에 대해 가장 좋은 매치를 찾습니다. (배치 처리)
    
    :param embeddings: (N, 512) 형태의 임베딩 numpy 배열
    :param faiss_index: FAISS 인덱스 또는 (인덱스, 레이블) 튜플
    :param threshold: 유사도 임계값
    :param labels: 레이블 배열 (faiss_index가 튜플이 아닌 경우)
    :return: [(이름, 유사도), ...] 형태의 결과 리스트
    """
    # FAISS 사용 가능 여부 확인
    if not FAISS_AVAILABLE or faiss is None:
        logging.warning("⚠️ FAISS를 사용할 수 없습니다. Unknown 반환")
        return [("Unknown", 0.0)] * len(embeddings) if embeddings is not None and len(embeddings) > 0 else []
    
    if isinstance(faiss_index, tuple):
        index, faiss_labels = faiss_index
    else:
        index = faiss_index
        faiss_labels = labels

    if index is None or embeddings is None or embeddings.shape[0] == 0:
        return [("Unknown", 0.0)] * len(embeddings)
    
    try:
        # 임베딩을 float32로 변환
        embeddings_array = embeddings.astype(np.float32)
        
        # 정규화 전 norm 확인 (디버깅)
        norms_before = np.linalg.norm(embeddings_array, axis=1)
        logging.debug(f"🔍 배치 FAISS 정규화 전: norms 범위=[{norms_before.min():.3f}, {norms_before.max():.3f}], 평균={norms_before.mean():.3f}")
        
        # L2 정규화 (IndexFlatIP는 코사인 유사도를 위해 정규화 필수)
        # 단일 검색과 동일하게 정규화하여 일관성 유지
        faiss.normalize_L2(embeddings_array)
        
        # 정규화 후 norm 확인 (디버깅)
        norms_after = np.linalg.norm(embeddings_array, axis=1)
        logging.debug(f"🔍 배치 FAISS 정규화 후: norms 범위=[{norms_after.min():.3f}, {norms_after.max():.3f}], 평균={norms_after.mean():.3f}")
        
        # 인덱스의 첫 번째 벡터 norm 확인 (디버깅)
        if index.ntotal > 0:
            try:
                # 인덱스에서 첫 번째 벡터 가져오기 (재구성)
                index_vector = index.reconstruct(0)
                index_norm = np.linalg.norm(index_vector)
                logging.debug(f"🔍 FAISS 인덱스 첫 번째 벡터 norm: {index_norm:.6f} (정규화되어야 1.0)")
            except:
                pass
        
        # k=2로 설정하여 가장 유사한 2개를 찾음 (오인식 방지용)
        distances, indices = index.search(embeddings_array, k=2)
        
        logging.info(f"🔍 배치 FAISS 검색: {len(embeddings)}개 임베딩, 인덱스 크기={index.ntotal}, 임계값={threshold}")
        logging.debug(f"🔍 배치 FAISS 검색 결과 범위: distances min={distances.min():.6f}, max={distances.max():.6f}, mean={distances.mean():.6f}")
        
        results = []
        for i in range(len(embeddings)):
            top1_idx = indices[i][0]
            top1_dist = distances[i][0]
            
            top2_dist = distances[i][1] if len(distances[i]) > 1 else -1.0
            
            # 인식률 향상: 임계값과 차이 검증 완화
            # 1위 점수가 임계값 이상이고, 차이가 충분하면 인정
            # 차이가 작아도 1위 점수가 충분히 높으면(임계값+0.05) 인정
            diff = top1_dist - top2_dist
            
            # 로깅: 검색 결과 상세 정보
            if top1_idx < len(faiss_labels):
                label_info = faiss_labels[top1_idx]
                if isinstance(label_info, dict):
                    matched_name = label_info.get('name', 'Unknown')
                else:
                    matched_name = str(label_info).strip()
            else:
                matched_name = "Unknown"
            
            # 인식률 최대화: 더 완화된 조건 적용
            # 1위 유사도가 임계값 이상이면 인정 (차이 검증 완화)
            if top1_dist >= threshold:
                # 차이 검증: 차이가 임계값 이상이거나, 1위 점수가 충분히 높으면 인정
                similarity_diff_threshold = getattr(config.Thresholds, 'SIMILARITY_DIFF_THRESHOLD', 0.05)
                if diff >= similarity_diff_threshold or top1_dist >= threshold + 0.03:
                    label_info = faiss_labels[top1_idx]
                    if isinstance(label_info, dict):
                        person_name = label_info.get('name', 'Unknown')
                    else:
                        person_name = str(label_info)
                    logging.info(f"✅ 배치 FAISS 매칭 성공 [{i}]: 인덱스={top1_idx}, 이름={person_name}, 유사도={top1_dist:.3f}, 2위={top2_dist:.3f}, 차이={diff:.3f}")
                    results.append((person_name, float(top1_dist)))
                else:
                    # 차이가 작아도 1위 점수가 임계값 이상이면 인정 (인식률 최대화)
                    if top1_dist >= threshold:
                        label_info = faiss_labels[top1_idx]
                        if isinstance(label_info, dict):
                            person_name = label_info.get('name', 'Unknown')
                        else:
                            person_name = str(label_info)
                        logging.info(f"✅ 배치 FAISS 매칭 성공 (차이 작음, 완화) [{i}]: 인덱스={top1_idx}, 이름={person_name}, 유사도={top1_dist:.3f}, 2위={top2_dist:.3f}, 차이={diff:.3f}")
                        results.append((person_name, float(top1_dist)))
                    else:
                        logging.warning(f"⚠️ 배치 FAISS 매칭 실패 (차이 작음) [{i}]: 인덱스={top1_idx}, 이름={matched_name}, 유사도={top1_dist:.3f} < 임계값={threshold:.3f}, 2위={top2_dist:.3f}, 차이={diff:.3f}")
                        results.append(("Unknown", float(top1_dist)))
            else:
                # 추가 검증: 1위 유사도가 임계값보다 낮지만, 1위와 2위 차이가 크고 1위 유사도가 임계값+0.03 이상이면 인정 (완화)
                similarity_diff_threshold = getattr(config.Thresholds, 'SIMILARITY_DIFF_THRESHOLD', 0.05)
                if top1_dist >= (threshold + 0.03) and diff >= similarity_diff_threshold:
                    label_info = faiss_labels[top1_idx]
                    if isinstance(label_info, dict):
                        person_name = label_info.get('name', 'Unknown')
                    else:
                        person_name = str(label_info)
                    logging.info(f"✅ 배치 FAISS 매칭 성공 (완화 조건) [{i}]: 인덱스={top1_idx}, 이름={person_name}, 유사도={top1_dist:.3f}, 임계값={threshold:.3f}, 2위={top2_dist:.3f}, 차이={diff:.3f}")
                    results.append((person_name, float(top1_dist)))
                # 추가 완화: 1위 유사도가 임계값보다 낮지만, 1위와 2위 차이가 충분히 크면 인정
                elif diff >= similarity_diff_threshold * 2 and top1_dist >= threshold * 0.7:
                    label_info = faiss_labels[top1_idx]
                    if isinstance(label_info, dict):
                        person_name = label_info.get('name', 'Unknown')
                    else:
                        person_name = str(label_info)
                    logging.info(f"✅ 배치 FAISS 매칭 성공 (추가 완화) [{i}]: 인덱스={top1_idx}, 이름={person_name}, 유사도={top1_dist:.3f}, 임계값={threshold:.3f}, 2위={top2_dist:.3f}, 차이={diff:.3f}")
                    results.append((person_name, float(top1_dist)))
                else:
                    logging.warning(f"⚠️ 배치 FAISS 매칭 실패 [{i}]: 인덱스={top1_idx}, 이름={matched_name}, 유사도={top1_dist:.3f} < 임계값={threshold:.3f} (차이: {threshold - top1_dist:.3f}), 2위={top2_dist:.3f}, 차이={diff:.3f}")
                    results.append(("Unknown", float(top1_dist)))
                
        return results
    except Exception as e:
        logging.error(f"배치 FAISS 검색 실패: {e}")
        return [("Unknown", 0.0)] * len(embeddings)


def find_best_match_faiss(
    embedding: np.ndarray, 
    faiss_index: Union[Any, Tuple[Any, Optional[np.ndarray]]],
    threshold: float, 
    labels: Optional[np.ndarray] = None
) -> Tuple[str, float]:
    """insightface 임베딩과 Faiss IndexFlatIP에 최적화된 검색 함수"""
    # FAISS 사용 가능 여부 확인
    if not FAISS_AVAILABLE or faiss is None:
        logging.warning("⚠️ FAISS를 사용할 수 없습니다. Unknown 반환")
        return "Unknown", 0.0
    
    try:
        logging.debug(f"🔍 find_best_match_faiss 호출: embedding shape={embedding.shape if embedding is not None else None}, faiss_index type={type(faiss_index)}")
        
        # faiss_index가 튜플인 경우 (core.py에서 반환하는 형태) 처리
        if isinstance(faiss_index, tuple):
            faiss_index, provided_labels = faiss_index
            logging.info(f"🔍 FAISS 튜플 언패킹: 인덱스 type={type(faiss_index)}, ntotal={faiss_index.ntotal if hasattr(faiss_index, 'ntotal') else 'N/A'}, 레이블 길이={len(provided_labels) if provided_labels is not None else 0}")
            if provided_labels is not None and len(provided_labels) > 0:
                labels = provided_labels
                logging.info(f"🔍 FAISS 레이블 사용: {len(labels)}개 레이블")
        
        # 레이블이 제공되지 않았거나 비어있는 경우 파일에서 로드
        if labels is None or len(labels) == 0:
            # 캐시된 레이블 확인 (함수 속성으로 저장된 경우)
            cached_labels = getattr(find_best_match_faiss, "labels", None)
            if cached_labels is None or len(cached_labels) == 0:
                # 파일에서 레이블 로드
                label_path = config.Paths.FAISS_LABELS
                if not os.path.exists(label_path):
                    label_path = os.path.normpath(os.path.join(config.BASE_DIR, "../..", "face_index.faiss.labels.npy"))

                if not os.path.exists(label_path):
                    logging.error(f"Faiss 레이블 파일 없음: {label_path}")
                    # 함수 속성에 기본값 저장
                    setattr(find_best_match_faiss, "labels", np.array(["Error"]))
                    labels = np.array(["Error"])
                else:
                    loaded_labels = np.load(label_path, allow_pickle=True)
                    # 함수 속성에 저장 (다음 호출 시 캐시 사용)
                    setattr(find_best_match_faiss, "labels", loaded_labels)
                    labels = loaded_labels
                    logging.info(f"Faiss 레이블 로드 완료: {label_path}")
            else:
                # 캐시된 레이블 사용
                labels = cached_labels

        # faiss_index가 None인 경우 빈 인덱스 처리
        if faiss_index is None:
            logging.error(f"❌ FAISS 인덱스가 None입니다!")
            return "Unknown", 0.0
        
        if hasattr(faiss_index, 'ntotal'):
            if faiss_index.ntotal == 0:
                logging.error(f"❌ FAISS 인덱스가 비어있음 (ntotal=0) - 얼굴 데이터베이스에 등록된 얼굴이 없습니다!")
                return "Unknown", 0.0
            logging.info(f"🔍 FAISS 인덱스 확인: ntotal={faiss_index.ntotal}")
        else:
            logging.error(f"❌ FAISS 인덱스에 ntotal 속성이 없습니다!")
            return "Unknown", 0.0

        # 임베딩 검증
        if embedding is None or embedding.size == 0:
            logging.error(f"❌ 임베딩이 None이거나 비어있습니다!")
            return "Unknown", 0.0
        
        logging.debug(f"🔍 임베딩 검증: shape={embedding.shape}, dtype={embedding.dtype}, norm={np.linalg.norm(embedding):.3f}")

        # 임베딩을 2D 배열로 변환 (FAISS는 (n, d) 형태를 요구)
        embedding_array = np.array(embedding, dtype='float32')
        # 1D 또는 2D 배열을 (1, 512) 형태로 정규화
        if embedding_array.ndim == 1:
            # 1D 배열: (512,) -> (1, 512)
            query_embedding = embedding_array.reshape(1, -1)
        elif embedding_array.ndim == 2:
            # 2D 배열: (1, 512) -> 그대로 사용, (512, 1) -> (1, 512)
            if embedding_array.shape[0] == 1:
                query_embedding = embedding_array
            elif embedding_array.shape[1] == 1:
                query_embedding = embedding_array.T
            else:
                # 이미 올바른 형태
                query_embedding = embedding_array
        else:
            # 3D 이상: squeeze 후 reshape
            embedding_array = np.squeeze(embedding_array)
            query_embedding = embedding_array.reshape(1, -1) if embedding_array.ndim == 1 else embedding_array
        
        # 최종 shape 확인 및 정규화
        if query_embedding.shape[0] != 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # L2 정규화 (ArcFace는 코사인 유사도 사용)
        faiss.normalize_L2(query_embedding)
        
        logging.debug(f"🔍 FAISS 검색 실행: query shape={query_embedding.shape}, 인덱스 크기={faiss_index.ntotal}")

        # 오인식 방지: top-5 검색 후 2차 검증 (더 많은 후보 확인)
        k = min(5, faiss_index.ntotal)  # 최대 5개 후보 검색 (3 -> 5, 더 정확한 매칭)
        similarities, indices = faiss_index.search(query_embedding, k)
        best_similarity = float(similarities[0][0])
        best_idx = int(indices[0][0])
        
        logging.info(f"🔍 FAISS 검색 결과 (Top-{k}): best_idx={best_idx}, best_similarity={best_similarity:.4f}, threshold={threshold}")

        # 오인식 방지: 1위와 2위의 유사도 차이 확인
        if k > 1:
            second_similarity = float(similarities[0][1])
            second_idx = int(indices[0][1])
            similarity_gap = best_similarity - second_similarity
            
            # 1위와 2위가 같은 사람인지 확인 (원본 + 좌우 반전 임베딩이 DB에 있을 수 있음)
            same_person = False
            if labels is not None and best_idx < len(labels) and second_idx < len(labels):
                best_name_raw = labels[best_idx]
                second_name_raw = labels[second_idx]
                
                # 이름 추출 (객체 또는 문자열)
                def extract_name(name_obj):
                    if isinstance(name_obj, dict):
                        return name_obj.get('name') or name_obj.get('id') or 'Unknown'
                    elif isinstance(name_obj, str):
                        return name_obj.strip()
                    else:
                        return str(name_obj).strip()
                
                best_name = extract_name(best_name_raw)
                second_name = extract_name(second_name_raw)
                
                # 같은 사람이면 차이 검증 건너뛰기
                if best_name == second_name and best_name != 'Unknown':
                    same_person = True
                    logging.debug(f"🔍 1위와 2위가 같은 사람 ({best_name}): 차이 검증 건너뜀")
            
            # 1위와 2위의 차이가 임계값 미만이면 불확실한 매칭으로 간주 (오인식 방지)
            # 단, 같은 사람이면 차이 검증 건너뛰기
            # 추가 조건: 최소 유사도 임계값 확인 (너무 낮은 유사도는 Unknown)
            diff_threshold = config.Thresholds.SIMILARITY_DIFF_THRESHOLD
            # 최소 유사도 임계값을 config의 SIMILARITY의 20%로 설정 (더 관대하게, 인식률 향상)
            min_similarity_threshold = threshold * 0.2  # 0.3 -> 0.2 (더 관대하게, 인식률 향상)
            
            # 최소 유사도 검증 (너무 낮은 유사도는 Unknown, 하지만 너무 엄격하지 않게)
            if best_similarity < min_similarity_threshold:
                logging.debug(f"🔍 FAISS 매칭 유사도 낮음: 1위={best_similarity:.3f} < {min_similarity_threshold:.3f} (Unknown)")
                return "Unknown", best_similarity
            
            # 1위와 2위 차이 검증: 오인식 방지 (조정됨)
            # 차이가 작으면 유사도가 높아야만 매칭 허용
            if not same_person:
                # 조건 1: 유사도가 0.70 이상이면 무조건 허용 (확실한 매칭)
                if best_similarity >= 0.70:
                    logging.info(f"✅ FAISS 매칭 성공 (고유사도): 1위={best_similarity:.3f} >= 0.70")
                    # 확실한 매칭, 계속 진행
                # 조건 2: 유사도 0.60~0.70이고 차이가 0.05 이상이면 허용
                elif best_similarity >= 0.60 and similarity_gap >= 0.05:
                    logging.info(f"✅ FAISS 매칭 성공 (중유사도+차이): 1위={best_similarity:.3f}, 2위={second_similarity:.3f}, 차이={similarity_gap:.3f} >= 0.05")
                    # 계속 진행
                # 조건 3: 유사도 0.55~0.60이고 차이가 0.08 이상이면 허용
                elif best_similarity >= 0.55 and similarity_gap >= 0.08:
                    logging.info(f"✅ FAISS 매칭 성공 (저유사도+큰차이): 1위={best_similarity:.3f}, 2위={second_similarity:.3f}, 차이={similarity_gap:.3f} >= 0.08")
                    # 계속 진행
                else:
                    # 위 조건 모두 불충족 → Unknown
                    logging.warning(f"⚠️ FAISS 매칭 불확실 (오인식 방지): 1위={best_similarity:.3f}, 2위={second_similarity:.3f}, 차이={similarity_gap:.3f}")
                    return "Unknown", best_similarity
            
            # 추가 검증: 1위와 3위의 차이도 확인 (더 엄격한 검증)
            # 단, 1위와 3위가 같은 사람이면 건너뛰기
            if k > 2:
                third_similarity = float(similarities[0][2])
                third_idx = int(indices[0][2])
                gap_1_3 = best_similarity - third_similarity
                
                # 1위와 3위가 같은 사람인지 확인
                same_person_1_3 = False
                if labels is not None and best_idx < len(labels) and third_idx < len(labels):
                    best_name_raw = labels[best_idx]
                    third_name_raw = labels[third_idx]
                    
                    # 이름 추출 (객체 또는 문자열)
                    def extract_name(name_obj):
                        if isinstance(name_obj, dict):
                            return name_obj.get('name') or name_obj.get('id') or 'Unknown'
                        elif isinstance(name_obj, str):
                            return name_obj.strip()
                        else:
                            return str(name_obj).strip()
                    
                    best_name = extract_name(best_name_raw)
                    third_name = extract_name(third_name_raw)
                    
                    if best_name == third_name and best_name != 'Unknown':
                        same_person_1_3 = True
                
                # 1위와 3위의 차이 검증: 1위 유사도가 임계값 이상이면 검증 건너뛰기
                # 단, 같은 사람이면 차이 검증 건너뛰기
                # 완화 조건: 1위 유사도가 임계값 이상이면 3위 검증 건너뛰기 (인식률 향상)
                if not same_person_1_3 and gap_1_3 < 0.15:
                    # 1위 유사도가 임계값 이상이면 3위 검증 건너뛰기
                    if best_similarity >= threshold:
                        logging.info(f"✅ FAISS 매칭 성공 (3위 차이 작지만 임계값 이상): 1위={best_similarity:.3f} >= {threshold}, 3위={third_similarity:.3f}, 차이={gap_1_3:.3f}")
                        # 3위 검증 건너뛰고 계속 진행
                    else:
                        logging.warning(f"⚠️ FAISS 매칭 불확실 (3위 검증): 1위={best_similarity:.3f} < {threshold}, 3위={third_similarity:.3f}, 차이={gap_1_3:.3f} < 0.15 (오인식 방지)")
                        return "Unknown", best_similarity

        # 디버깅: 매칭 결과 상세 로깅 (INFO 레벨로 변경하여 항상 표시)
        logging.info(f"🔍 FAISS 매칭 결과: 인덱스={best_idx}, 유사도={best_similarity:.3f}, 임계값={threshold}, 인덱스 크기={faiss_index.ntotal}, 레이블 크기={len(labels) if labels is not None else 0}")

        # 오인식 방지: 최소 유사도 검증 추가
        # 임계값을 넘었더라도 최소 임계값 이상이어야 함 (너무 낮은 유사도는 오인식 가능성 높음)
        # config의 SIMILARITY 임계값을 사용 (하드코딩된 0.35 대신)
        min_absolute_similarity = threshold  # config.Thresholds.SIMILARITY와 동일하게 사용
        if best_similarity < min_absolute_similarity:
            logging.warning(f"⚠️ FAISS 매칭 유사도 부족: {best_similarity:.3f} < {min_absolute_similarity} (임계값 미달)")
            return "Unknown", best_similarity

        if best_similarity >= threshold:
            if best_idx < len(labels):
                best_match_name = labels[best_idx]
                # 이름 형식 처리 (객체 또는 문자열)
                if isinstance(best_match_name, dict):
                    # 딕셔너리인 경우 'name' 또는 'id' 필드 추출
                    best_match_name = best_match_name.get('name') or best_match_name.get('id') or 'Unknown'
                elif isinstance(best_match_name, str):
                    # 문자열인 경우 그대로 사용 (앞뒤 공백 제거)
                    best_match_name = best_match_name.strip()
                else:
                    # numpy 배열이나 다른 타입인 경우 문자열로 변환 후 strip
                    best_match_name = str(best_match_name).strip()
                logging.info(f"✅ FAISS 매칭 성공: 인덱스={best_idx}, 이름={best_match_name}, 유사도={best_similarity:.3f}")
                return best_match_name, best_similarity
            else:
                logging.warning(f"⚠️ FAISS 인덱스 범위 초과: 인덱스={best_idx}, 레이블 배열 크기={len(labels)}")
                return "Unknown", best_similarity
        else:
            logging.warning(f"⚠️ FAISS 매칭 실패: 유사도={best_similarity:.3f} < 임계값={threshold} (차이: {threshold - best_similarity:.3f})")
            return "Unknown", best_similarity
    except Exception as e:
        logging.error(f"Faiss 검색 중 오류 발생: {e}", exc_info=True)
        return "Unknown", 0.0


def log_violation(frame: np.ndarray, person_name: str, event_type: str, cam_id: int) -> None:
    # Unknown 사용자는 로컬 저장 건너뜀
    if not person_name or person_name.lower() in ['unknown', '알수없음', '알 수 없음', '미확인']:
        logging.debug(f"[CAM-{cam_id}] Unknown 사용자 로컬 저장 건너뜀: person={person_name}")
        return
    
    try:
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        safe_event_type = "".join(c for c in event_type if c.isalnum() or c in ('-'))

        # ⭐️ 로그 저장 경로를 config.py에서 가져오도록 수정 ⭐️
        log_folder = config.Paths.LOG_FOLDER
        image_filename = os.path.join(log_folder, f"{timestamp_str}_CAM{cam_id}_{person_name}_{safe_event_type}.jpg")

        # ⭐️ 이미지 저장 경로가 유효한지 확인 ⭐️
        if not os.path.exists(log_folder):
             os.makedirs(log_folder, exist_ok=True)

        cv2.imwrite(image_filename, frame)

        # --- [수정] config에서 CSV 파일 경로를 가져오도록 변경 ---
        log_filename = config.Paths.LOG_CSV
        log_entry = f"{now.strftime('%Y-%m-%d %H:%M:%S')},{person_name},{event_type},CAM-{cam_id},{image_filename}\n"

        # ⭐️ CSV 파일 헤더 쓰기 로직 개선 ⭐️
        file_exists = os.path.exists(log_filename)
        with open(log_filename, 'a', encoding='utf-8-sig', newline='') as f:
            if not file_exists:
                f.write("Timestamp,Person,Event,CameraID,EvidenceFile\n")
            f.write(log_entry)

        logging.info(f"[CAM-{cam_id}] 이벤트 기록 저장: {person_name} - {event_type}")
    except Exception as e:
        logging.error(f"로그 파일/이미지 저장 실패: {e}", exc_info=True)


def is_person_horizontal(
    keypoints: Union[Keypoints, Dict[str, Any], None], 
    bbox_xyxy: Tuple[float, float, float, float],
    person_crop: Optional[np.ndarray] = None,
    fall_model: Optional[Any] = None,
    cam_id: int = 0
) -> bool:
    """
    개선된 넘어짐 감지 로직 (오탐지 방지 강화)
    
    판정 기준:
    1. Fall 모델 고신뢰도 (>=0.7): 즉시 넘어짐 판정
    2. 키포인트 + 바운딩 박스 + Fall 모델 융합 점수
    3. 추가 검증: 앉아있는 자세, 웅크린 자세 필터링
    4. 키포인트가 없어도 박스 비율이 매우 높으면 넘어짐 후보
    
    Args:
        keypoints: 키포인트 객체 (Keypoints 또는 dict{'xy': np.array, 'conf': np.array})
        bbox_xyxy: 바운딩 박스 좌표
        person_crop: 사람 영역 크롭 이미지 (FallSafe 모델용, 선택적)
        fall_model: FallSafe 모델 객체 (선택적)
        cam_id: 카메라 ID (로깅용)
    """
    try:
        # 1. 바운딩 박스 기본 검증
        x1, y1, x2, y2 = bbox_xyxy
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        
        # 너무 작은 박스는 무시 (오탐지 방지)
        if box_area < 3000:
            logging.debug(f"넘어짐 감지 스킵: 박스 면적 너무 작음 ({box_area:.0f} < 3000)")
            return False
        
        # 박스 비율 계산 (디버깅용)
        aspect_ratio = box_w / box_h if box_h > 0 else 0
        logging.debug(f"🔍 넘어짐 분석: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), "
                     f"비율={aspect_ratio:.2f}, 면적={box_area:.0f}, "
                     f"키포인트={'있음' if keypoints is not None else '없음'}")
        
        # 2. 키포인트 기반 자세 분석
        keypoint_score = _analyze_pose_with_keypoints(keypoints)
        
        # 3. 바운딩 박스 비율 분석 (가로가 세로보다 긴 경우)
        bbox_score = _analyze_bbox_ratio(bbox_xyxy)
        
        # 4. 키포인트 분산 분석
        spread_score = _analyze_keypoint_spread(keypoints)
        
        # ⭐ 키포인트가 없거나 부족해도 박스 비율이 매우 높으면 넘어짐 후보
        # 박스 비율이 2.0 이상이면 키포인트 없이도 높은 점수 부여
        if keypoints is None and aspect_ratio >= 2.0:
            bbox_score = min((aspect_ratio - 1.5) / 0.5, 1.0)  # 더 높은 점수
            keypoint_score = 0.3  # 기본 점수 부여
            spread_score = 0.3
            logging.info(f"🔻 키포인트 없이 박스 비율로 넘어짐 후보 감지: 비율={aspect_ratio:.2f}")
        
        # ⭐ 앉아있는 자세 필터링 (오탐지 방지)
        # 앉아있으면 가로가 길어 보이지만, 어깨가 엉덩이 위에 있음
        is_sitting = _is_sitting_pose(keypoints)
        if is_sitting:
            # 앉아있는 경우 점수 대폭 감소
            keypoint_score *= 0.3
            bbox_score *= 0.3
            spread_score *= 0.3
            logging.debug(f"앉은 자세 감지: 점수 감소 적용")
        
        # 5. FallSafe 모델 분석 (조건부 실행)
        fallsafe_score = 0.0
        if fall_model is not None and person_crop is not None:
            # 초기 점수가 일정 수준 이상일 때만 Fall 모델 실행
            preliminary_score = keypoint_score * 0.4 + bbox_score * 0.3 + spread_score * 0.3
            
            if preliminary_score >= 0.25:  # 0.2 -> 0.25 (더 엄격하게)
                try:
                    results = fall_model(
                        person_crop,
                        conf=0.6,  # 0.5 -> 0.6 (더 엄격하게)
                        verbose=False,
                        imgsz=640
                    )
                    
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                # 클래스 인덱스 1 = Fall
                                if cls == 1 and conf >= 0.6:
                                    fallsafe_score = conf
                                    break
                except Exception:
                    pass
        
        # 6. Fall 모델 고신뢰도 즉시 판정 (0.65 -> 0.7)
        if fall_model is not None and fallsafe_score >= 0.7:
            logging.info(f"⚠️ 넘어짐 감지 (Fall 모델 고신뢰도): conf={fallsafe_score:.2f}")
            return True
        
        # 7. 융합 점수 계산
        if fall_model is not None and fallsafe_score > 0:
            # FallSafe 모델이 있으면: 키포인트 30%, 바운딩 박스 15%, 분산 15%, FallSafe 40%
            total_score = (
                keypoint_score * 0.30 + 
                bbox_score * 0.15 + 
                spread_score * 0.15 + 
                fallsafe_score * 0.40
            )
        else:
            # FallSafe 모델이 없으면: 키포인트 50%, 바운딩 박스 30%, 분산 20%
            total_score = (
                keypoint_score * 0.50 + 
                bbox_score * 0.30 + 
                spread_score * 0.20
            )
        
        # 8. 임계값 기반 판정 (0.6 -> 0.65, 더 엄격하게)
        fall_threshold = 0.65  # config.Thresholds.FALL_SCORE_THRESHOLD
        
        # Fall 모델이 중간 신뢰도(0.6~0.7)일 때만 임계값 약간 완화
        if fall_model is not None and 0.6 <= fallsafe_score < 0.7:
            fall_threshold = 0.55
        
        if total_score >= fall_threshold:
            logging.info(f"⚠️ 넘어짐 감지: 점수={total_score:.2f} "
                            f"(자세={keypoint_score:.2f}, 박스={bbox_score:.2f}, "
                        f"분산={spread_score:.2f}, Fall={fallsafe_score:.2f})")
            return True
        
        return False
        
    except Exception as e:
        logging.warning(f"넘어짐 감지 함수 오류: {e}")
        return False


def _is_sitting_pose(keypoints: Union[Keypoints, Dict[str, Any], None]) -> bool:
    """
    앉아있는 자세인지 판단 (오탐지 방지용)
    어깨가 엉덩이 위에 있고, 무릎이 엉덩이 근처에 있으면 앉아있는 것으로 판단
    """
    try:
        if keypoints is None:
            return False
        
        # dict 형태 처리
        if isinstance(keypoints, dict):
            points = keypoints.get('xy')
            confidences = keypoints.get('conf')
            if points is None or confidences is None:
                return False
            if hasattr(points, 'cpu'):
                points = points.cpu().numpy()
            if hasattr(confidences, 'cpu'):
                confidences = confidences.cpu().numpy()
        else:
            if keypoints.data is None or len(keypoints.data) == 0:
                return False
            points = keypoints.xy[0].cpu().numpy()
            confidences = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else None
        
        if confidences is None:
            return False
        
        valid_mask = confidences > config.Thresholds.POSE_CONFIDENCE
        
        # 어깨, 엉덩이, 무릎 키포인트 확인
        # 5: left_shoulder, 6: right_shoulder
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee
        
        if not (valid_mask[5] and valid_mask[6] and valid_mask[11] and valid_mask[12]):
            return False
        
        shoulder_mid = (points[5] + points[6]) / 2
        hip_mid = (points[11] + points[12]) / 2
        
        # 어깨가 엉덩이 위에 있는지 확인 (Y 좌표가 작을수록 위쪽)
        shoulder_above_hip = shoulder_mid[1] < hip_mid[1]
        
        if not shoulder_above_hip:
            return False  # 어깨가 엉덩이 아래면 앉아있지 않음
        
        # 어깨-엉덩이 수직 거리 계산
        vertical_dist = hip_mid[1] - shoulder_mid[1]
        
        # 무릎이 있으면 무릎-엉덩이 거리 확인
        if valid_mask[13] and valid_mask[14]:
            knee_mid = (points[13] + points[14]) / 2
            knee_hip_dist = abs(knee_mid[1] - hip_mid[1])
            
            # 무릎이 엉덩이 근처에 있으면 앉아있는 것으로 판단
            # (서있으면 무릎이 엉덩이보다 훨씬 아래에 있음)
            if knee_hip_dist < vertical_dist * 0.5:
                return True
        
        # 어깨-엉덩이 거리가 짧으면 (상체가 웅크린 상태) 앉아있을 가능성
        # 일반적으로 서있을 때 어깨-엉덩이 거리가 더 김
        shoulder_hip_horizontal = abs(shoulder_mid[0] - hip_mid[0])
        if vertical_dist < shoulder_hip_horizontal * 0.8:
            return True
        
        return False
        
    except Exception:
        return False


def _analyze_pose_with_keypoints(keypoints: Union[Keypoints, Dict[str, Any], None]) -> float:
    """
    키포인트 기반 자세 분석 (0.0 ~ 1.0)
    어깨-엉덩이-무릎 각도와 위치 관계를 분석
    
    Args:
        keypoints: Keypoints 객체 또는 dict{'xy': np.array(17,2), 'conf': np.array(17,)}
    """
    try:
        if keypoints is None:
            return 0.0
        
        # dict 형태 처리 (frame_processor에서 전달)
        if isinstance(keypoints, dict):
            points = keypoints.get('xy')
            confidences = keypoints.get('conf')
            if points is None or confidences is None:
                return 0.0
            # numpy array로 변환
            if hasattr(points, 'cpu'):
                points = points.cpu().numpy()
            if hasattr(confidences, 'cpu'):
                confidences = confidences.cpu().numpy()
        else:
            # Keypoints 객체 처리
            if keypoints.data is None or len(keypoints.data) == 0:
                return 0.0
        points = keypoints.xy[0].cpu().numpy()  # (17, 2)
        confidences = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else None  # (17,)
        
        if confidences is None:
            return 0.0
        
        # 유효한 키포인트만 필터링
        valid_mask = confidences > config.Thresholds.POSE_CONFIDENCE
        if np.sum(valid_mask) < 5:  # 최소 5개 키포인트 필요
            return 0.0
        
        score = 0.0
        
        # COCO 포맷 키포인트 인덱스
        # 5: left_shoulder, 6: right_shoulder
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee
        # 15: left_ankle, 16: right_ankle
        
        # 1. 어깨-엉덩이 수직성 검사 (40%)
        if (valid_mask[5] and valid_mask[6] and 
            valid_mask[11] and valid_mask[12]):
            shoulder_mid = (points[5] + points[6]) / 2
            hip_mid = (points[11] + points[12]) / 2
            
            # 어깨-엉덩이 벡터의 수직 각도 계산
            shoulder_hip_vec = hip_mid - shoulder_mid
            if np.linalg.norm(shoulder_hip_vec) > 1e-5:
                # 수직 벡터 (0, 1)와의 각도 계산
                vertical_vec = np.array([0, 1])
                dot_product = np.dot(shoulder_hip_vec, vertical_vec)
                norm_product = np.linalg.norm(shoulder_hip_vec) * np.linalg.norm(vertical_vec)
                if norm_product > 1e-5:
                    cos_angle = dot_product / norm_product
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    
                    # 각도가 작을수록 (수직에 가까울수록) 넘어짐 가능성 높음
                    # 0도: 완전 수직 (넘어짐), 90도: 완전 수평 (서있음)
                    if angle < 30:  # 30도 이하면 넘어짐 가능성 높음
                        angle_score = 1.0 - (angle / 30.0)  # 0도면 1.0, 30도면 0.0
                        score += angle_score * 0.4
        
        # 2. 무릎-발목 위치 검증 (30%)
        if (valid_mask[5] and valid_mask[6] and 
            valid_mask[13] and valid_mask[14]):
            shoulder_mid = (points[5] + points[6]) / 2
            knee_mid = (points[13] + points[14]) / 2
            
            # 무릎이 어깨보다 위에 있으면 넘어짐 가능성 높음
            # (Y 좌표가 작을수록 위쪽)
            if knee_mid[1] < shoulder_mid[1]:
                # 높이 차이 계산
                height_diff = shoulder_mid[1] - knee_mid[1]
                # 어깨-엉덩이 거리로 정규화
                if valid_mask[11] and valid_mask[12]:
                    hip_mid = (points[11] + points[12]) / 2
                    shoulder_hip_dist = np.linalg.norm(shoulder_mid - hip_mid)
                    if shoulder_hip_dist > 1e-5:
                        normalized_diff = height_diff / shoulder_hip_dist
                        # 정규화된 차이가 클수록 넘어짐 가능성 높음
                        position_score = min(normalized_diff / 0.5, 1.0)  # 0.5 이상이면 1.0
                        score += position_score * 0.3
        
        # 3. 엉덩이-무릎-발목 각도 검증 (30%)
        if (valid_mask[11] and valid_mask[12] and 
            valid_mask[13] and valid_mask[14] and
            valid_mask[15] and valid_mask[16]):
            hip_mid = (points[11] + points[12]) / 2
            knee_mid = (points[13] + points[14]) / 2
            ankle_mid = (points[15] + points[16]) / 2
            
            # 엉덩이-무릎-발목 벡터
            hip_knee_vec = knee_mid - hip_mid
            knee_ankle_vec = ankle_mid - knee_mid
            
            if (np.linalg.norm(hip_knee_vec) > 1e-5 and 
                np.linalg.norm(knee_ankle_vec) > 1e-5):
                # 두 벡터의 각도 계산
                dot_product = np.dot(hip_knee_vec, knee_ankle_vec)
                norm_product = np.linalg.norm(hip_knee_vec) * np.linalg.norm(knee_ankle_vec)
                if norm_product > 1e-5:
                    cos_angle = dot_product / norm_product
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    
                    # 각도가 작을수록 (다리가 펴져있을수록) 넘어짐 가능성 높음
                    # 0도: 완전히 펴짐 (넘어짐), 90도: 구부러짐 (서있음)
                    if angle < 60:  # 60도 이하면 넘어짐 가능성
                        angle_score = 1.0 - (angle / 60.0)
                        score += angle_score * 0.3
        
        return min(score, 1.0)
        
    except Exception as e:
        logging.debug(f"키포인트 자세 분석 오류: {e}")
        return 0.0


def _analyze_bbox_ratio(bbox_xyxy: Tuple[float, float, float, float]) -> float:
    """
    바운딩 박스 비율 분석 (0.0 ~ 1.0)
    기존 로직을 점수로 변환
    """
    try:
        x1, y1, x2, y2 = bbox_xyxy
        box_w = x2 - x1
        box_h = y2 - y1
        
        if box_h < 1e-5 or box_w < 1e-5:
            return 0.0
        
        aspect_ratio = box_w / box_h if box_h > 0 else 0
        
        # 가로가 세로보다 1.5배 이상이면 넘어짐으로 판단
        fall_aspect_ratio_threshold = config.Thresholds.FALL_ASPECT_RATIO  # 1.5
        
        # 추가 검증: 박스가 너무 작지 않아야 함 (오탐 방지)
        box_area = box_w * box_h
        min_box_area = 2000  # 최소 면적 필터
        
        if box_area < min_box_area:
            return 0.0
        
        # 비율이 임계값 이상이면 점수 계산
        if aspect_ratio >= fall_aspect_ratio_threshold:
            # 1.5배면 0.5점, 2.0배면 1.0점 (선형 보간)
            score = min((aspect_ratio - fall_aspect_ratio_threshold) / 0.5, 1.0)
            return score
        
        return 0.0
        
    except Exception as e:
        logging.debug(f"바운딩 박스 비율 분석 오류: {e}")
        return 0.0


def _analyze_keypoint_spread(keypoints: Union[Keypoints, Dict[str, Any], None]) -> float:
    """
    키포인트 분산 분석 (0.0 ~ 1.0)
    수평 분산이 수직 분산보다 크면 넘어짐 가능성 높음
    """
    try:
        if keypoints is None:
            return 0.0
        
        # dict 형태 처리
        if isinstance(keypoints, dict):
            points = keypoints.get('xy')
            confidences = keypoints.get('conf')
            if points is None or confidences is None:
                return 0.0
            if hasattr(points, 'cpu'):
                points = points.cpu().numpy()
            if hasattr(confidences, 'cpu'):
                confidences = confidences.cpu().numpy()
        else:
            if keypoints.data is None or len(keypoints.data) == 0:
                return 0.0
        points = keypoints.xy[0].cpu().numpy()  # (17, 2)
        confidences = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else None
        
        if confidences is None:
            return 0.0
        
        # 유효한 키포인트만 필터링
        valid_mask = confidences > config.Thresholds.POSE_CONFIDENCE
        if np.sum(valid_mask) < 5:  # 최소 5개 키포인트 필요
            return 0.0
        
        valid_points = points[valid_mask]
        
        # 수평 분산과 수직 분산 계산
        x_std = np.std(valid_points[:, 0])
        y_std = np.std(valid_points[:, 1])
        
        if x_std < 1e-5 or y_std < 1e-5:
            return 0.0
        
        # 수평 분산이 수직 분산보다 클수록 넘어짐 가능성 높음
        spread_ratio = x_std / y_std
        fall_spread_threshold = config.Thresholds.FALL_HORIZONTAL_SPREAD_RATIO  # 1.5
        
        if spread_ratio >= fall_spread_threshold:
            # 1.5배면 0.5점, 2.0배면 1.0점 (선형 보간)
            score = min((spread_ratio - fall_spread_threshold) / 0.5, 1.0)
            return score
        
        return 0.0
        
    except Exception as e:
        logging.debug(f"키포인트 분산 분석 오류: {e}")
        return 0.0


def clip_bbox_xyxy(bbox_xyxy: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if (x2 - x1) > 0 and (y2 - y1) > 0:
        return x1, y1, x2, y2
    return None

def refine_box_from_keypoints(
    keypoints: Keypoints, 
    original_box: Tuple[float, float, float, float], 
    frame_w: int, 
    frame_h: int, 
    padding_ratio: float = 0.1
) -> Optional[Tuple[int, int, int, int]]:
    """
    키포인트를 기반으로 박스를 더 정확하게 조정합니다.
    여러 사람이 겹칠 때 박스를 키포인트 범위에 맞게 축소하여 분리합니다.
    
    Args:
        keypoints: Keypoints 객체
        original_box: 원본 박스 (x1, y1, x2, y2)
        frame_w: 프레임 너비
        frame_h: 프레임 높이
        padding_ratio: 키포인트 범위에 추가할 패딩 비율 (기본 10%)
    
    Returns:
        조정된 박스 (x1, y1, x2, y2) 또는 None
    """
    try:
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return None
        
        # 키포인트 좌표 추출
        points = keypoints.xy[0].cpu().numpy()  # shape: (17, 2)
        confidences = keypoints.conf[0].cpu().numpy()  # shape: (17,)
        
        # 유효한 키포인트만 사용
        valid_mask = confidences > config.Thresholds.POSE_CONFIDENCE
        if np.sum(valid_mask) < 3:  # 최소 3개 키포인트 필요
            return None
        
        valid_points = points[valid_mask]
        
        # 키포인트의 최소/최대 좌표 계산
        min_x = np.min(valid_points[:, 0])
        max_x = np.max(valid_points[:, 0])
        min_y = np.min(valid_points[:, 1])
        max_y = np.max(valid_points[:, 1])
        
        # 패딩 추가 (키포인트 범위의 일정 비율)
        width = max_x - min_x
        height = max_y - min_y
        padding_x = width * padding_ratio
        padding_y = height * padding_ratio
        
        # 새로운 박스 계산
        new_x1 = max(0, int(min_x - padding_x))
        new_y1 = max(0, int(min_y - padding_y))
        new_x2 = min(frame_w, int(max_x + padding_x))
        new_y2 = min(frame_h, int(max_y + padding_y))
        
        # 원본 박스보다 크지 않도록 제한 (키포인트 기반 박스는 더 작아야 함)
        orig_x1, orig_y1, orig_x2, orig_y2 = original_box
        new_x1 = int(max(new_x1, orig_x1))
        new_y1 = int(max(new_y1, orig_y1))
        new_x2 = int(min(new_x2, orig_x2))
        new_y2 = int(min(new_y2, orig_y2))
        
        # 유효성 검사
        if (new_x2 - new_x1) > 0 and (new_y2 - new_y1) > 0:
            return (new_x1, new_y1, new_x2, new_y2)
        
        return None
    except Exception as e:
        logging.debug(f"키포인트 기반 박스 조정 오류: {e}")
        return None