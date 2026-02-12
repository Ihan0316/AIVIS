# utils.py (최종 수정본)
import datetime
import logging
import os
import time
from typing import Tuple, Optional, List, Dict, Any, Union
import cv2
import faiss
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
        # ⭐️ 로그 파일 경로를 config에서 가져오도록 수정 ⭐️
        os.path.join(config.Paths.LOG_FOLDER, "system.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )

    # 콘솔 핸들러 설정 (터미널에도 출력)
    console_handler = logging.StreamHandler(sys.stdout)

    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # ⭐️ 포맷터에 파일명/줄번호 추가 ⭐️
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 루트 로거 설정 - 모든 로거의 기본 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers.clear()
    
    # 파일과 콘솔 핸들러 추가
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 모든 서브 로거가 루트 로거를 사용하도록 설정 (propagate는 기본적으로 True)
    # 주요 모듈 로거들도 명시적으로 설정하여 파일에 저장되도록 보장
    for logger_name in ['', '__main__', 'core', 'utils', 'frame_processor', 'camera_worker', 
                        'storage_manager', 'database', 'main', 'state']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True  # 루트 로거로 전파 (파일 핸들러 사용)
        # 서브 로거에 직접 핸들러를 추가하지 않음 (루트 로거의 핸들러 사용)

    # PIL 로거 레벨 조정
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # pymongo 디버그 로그 줄이기 (너무 많은 연결/하트비트 로그 방지)
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('pymongo.connection').setLevel(logging.WARNING)
    logging.getLogger('pymongo.topology').setLevel(logging.WARNING)
    logging.getLogger('pymongo.serverSelection').setLevel(logging.WARNING)

    # ultralytics 로거 레벨 조정 (너무 많은 로그 방지)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    
    # aiohttp 로거 레벨 조정
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('aiohttp.access').setLevel(logging.WARNING)

    logging.info("로깅 시스템 초기화 완료 - 모든 로그가 파일에 저장됩니다")


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


def find_best_match_faiss(
    embedding: np.ndarray, 
    faiss_index: Union[Any, Tuple[Any, Optional[np.ndarray]]],
    threshold: float, 
    labels: Optional[np.ndarray] = None
) -> Tuple[str, float]:
    """insightface 임베딩과 Faiss IndexFlatIP에 최적화된 검색 함수"""
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
        
        # 오인식 방지: top-5 검색 후 2차 검증 (더 많은 후보 확인)
        k = min(5, faiss_index.ntotal)  # 최대 5개 후보 검색 (3 -> 5, 더 정확한 매칭)
        
        # FAISS 검색 실행
        faiss_search_start = time.time()
        logging.debug(f"[FAISS] 검색 시작: k={k}, 임베딩 수={faiss_index.ntotal}개, "
                     f"쿼리 임베딩 shape={query_embedding.shape}, 임계값={threshold:.3f}")
        
        similarities, indices = faiss_index.search(query_embedding, k)
        faiss_search_time = (time.time() - faiss_search_start) * 1000  # ms
        
        best_similarity = float(similarities[0][0])
        best_idx = int(indices[0][0])
        
        # Top-K 결과 상세 로깅
        top_k_results = []
        for i in range(min(k, len(similarities[0]))):
            idx = int(indices[0][i])
            sim = float(similarities[0][i])
            name = "Unknown"
            if labels is not None and idx < len(labels):
                name_obj = labels[idx]
                if isinstance(name_obj, dict):
                    name = name_obj.get('name') or name_obj.get('id') or 'Unknown'
                elif isinstance(name_obj, str):
                    name = name_obj.strip()
                else:
                    name = str(name_obj).strip()
            top_k_results.append(f"{name}({sim:.3f})")
        
        # Top-K 결과 로그 출력
        logging.info(f"[FAISS] 검색 완료: {faiss_search_time:.1f}ms, Top-{k} 결과: {', '.join(top_k_results)}")

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
            diff_threshold = config.Thresholds.SIMILARITY_DIFF_THRESHOLD
            if not same_person and similarity_gap < diff_threshold:
                logging.warning(f"[FAISS] ⚠️ 매칭 불확실 (2위 검증): "
                              f"1위={best_name}({best_similarity:.3f}), "
                              f"2위={second_name}({second_similarity:.3f}), "
                              f"차이={similarity_gap:.3f} < {diff_threshold} (오인식 방지), "
                              f"검색 시간={faiss_search_time:.1f}ms")
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
                
                # 1위와 3위의 차이가 너무 작으면 (0.15 미만) 불확실한 매칭
                # 단, 같은 사람이면 차이 검증 건너뛰기
                # 오인식 방지 강화: 0.10 -> 0.15
                if not same_person_1_3 and gap_1_3 < 0.15:
                    logging.warning(f"[FAISS] ⚠️ 매칭 불확실 (3위 검증): "
                                  f"1위={best_similarity:.3f}, 3위={third_similarity:.3f}, "
                                  f"차이={gap_1_3:.3f} < 0.15 (오인식 방지), 검색 시간={faiss_search_time:.1f}ms")
                    return "Unknown", best_similarity

        # Top-K 결과는 이미 위에서 생성됨 (590줄), 중복 생성 제거
        # Top-K 결과를 INFO 레벨로 출력 (유사도 매칭 결과 확인용)
        logging.info(f"[FAISS] 검색 완료: {faiss_search_time:.1f}ms, 인덱스 크기={faiss_index.ntotal}개, "
                     f"Top-{k} 결과: {', '.join(top_k_results)}")

        # 오인식 방지: 최소 유사도 검증 추가
        # 임계값을 넘었더라도 최소 0.35 이상이어야 함 (너무 낮은 유사도는 오인식 가능성 높음)
        min_absolute_similarity = 0.35
        if best_similarity < min_absolute_similarity:
            logging.warning(f"[FAISS] ⚠️ 최소 유사도 미달: {best_similarity:.3f} < {min_absolute_similarity} "
                          f"(오인식 방지, 검색 시간={faiss_search_time:.1f}ms, 최적={top_k_results[0] if top_k_results else 'N/A'})")
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
                logging.info(f"[FAISS] ✅ 매칭 성공: 인덱스={best_idx}, 이름={best_match_name}, "
                           f"유사도={best_similarity:.3f} >= 임계값({threshold:.3f}), "
                           f"차이={best_similarity - threshold:.3f}, 검색 시간={faiss_search_time:.1f}ms")
                return best_match_name, best_similarity
            else:
                logging.warning(f"[FAISS] ⚠️ 인덱스 범위 초과: 인덱스={best_idx}, "
                              f"레이블 배열 크기={len(labels)}, 최적 유사도={best_similarity:.3f}")
                return "Unknown", best_similarity
        else:
            logging.warning(f"[FAISS] ❌ 매칭 실패: 유사도={best_similarity:.3f} < 임계값={threshold:.3f} "
                          f"(차이={threshold - best_similarity:.3f}), 검색 시간={faiss_search_time:.1f}ms, "
                          f"인덱스 크기={faiss_index.ntotal}개")
            # 상위 5개 결과 상세 로깅 (얼굴 인식 실패 디버깅용)
            if k > 0:
                top_matches = []
                for i in range(min(5, k, len(similarities[0]))):
                    idx = int(indices[0][i])
                    sim = float(similarities[0][i])
                    name = "Unknown"
                    if labels is not None and idx < len(labels):
                        name_obj = labels[idx]
                        if isinstance(name_obj, dict):
                            name = name_obj.get('name') or name_obj.get('id') or 'Unknown'
                        elif isinstance(name_obj, str):
                            name = name_obj.strip()
                        else:
                            name = str(name_obj).strip()
                    gap = threshold - sim
                    top_matches.append(f"{name}({sim:.3f}, -{gap:.3f})")
                logging.info(f"[FAISS] 🔍 상위 {len(top_matches)}개 매칭 후보: {', '.join(top_matches)}")
            return "Unknown", best_similarity
    except Exception as e:
        logging.error(f"Faiss 검색 중 오류 발생: {e}", exc_info=True)
        return "Unknown", 0.0


def log_violation(frame: np.ndarray, person_name: str, event_type: str, cam_id: int) -> None:
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


def is_person_horizontal(keypoints: Keypoints, bbox_xyxy: Tuple[float, float, float, float]) -> bool:
    """
    바운딩 박스 기반 넘어짐 감지 로직 (비활성화됨 - 항상 False 반환)
    넘어짐 감지 기능이 비활성화되어 있습니다.
    """
    # 넘어짐 감지 비활성화 - 항상 False 반환
    return False


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
        
        # ⭐ 박스 확장 모드: 키포인트 범위를 포함하도록 원본 박스보다 더 크게 허용
        # 원본 박스와 키포인트 박스 중 더 큰 박스 사용 (사람 전체 포함)
        orig_x1, orig_y1, orig_x2, orig_y2 = original_box
        # 더 큰 박스 사용 (사람 전체 포함)
        new_x1 = int(min(new_x1, orig_x1))  # 더 위로 확장
        new_y1 = int(min(new_y1, orig_y1))  # 더 위로 확장
        new_x2 = int(max(new_x2, orig_x2))  # 더 아래로 확장
        new_y2 = int(max(new_y2, orig_y2))  # 더 아래로 확장
        
        # 유효성 검사
        if (new_x2 - new_x1) > 0 and (new_y2 - new_y1) > 0:
            return (new_x1, new_y1, new_x2, new_y2)
        
        return None
    except Exception as e:
        logging.debug(f"키포인트 기반 박스 조정 오류: {e}")
        return None
