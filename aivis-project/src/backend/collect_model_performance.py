"""
모델 성능 지표 수집 스크립트
현재 사용 중인 모델들의 성능 지표를 수집하고 리포트를 생성합니다.
Precision, Recall, F1-Score를 포함한 정확도 지표를 계산합니다.
"""
import sys
import os
import json
import time
import logging
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import cv2

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core import SafetySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """두 바운딩 박스의 IoU (Intersection over Union) 계산"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 교집합 영역 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 각 박스의 넓이
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # 합집합 영역
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_detections(
    pred_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    예측 박스와 Ground Truth 박스를 매칭하여 TP, FP, FN 계산
    
    Returns:
        (TP, FP, FN) 튜플
    """
    if not gt_boxes:
        return (0, len(pred_boxes), 0)
    if not pred_boxes:
        return (0, 0, len(gt_boxes))
    
    # 매칭된 GT 박스 추적
    matched_gt = set()
    tp = 0
    
    # 각 예측 박스에 대해 가장 높은 IoU를 가진 GT 박스 찾기
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    
    return (tp, fp, fn)


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Precision, Recall, F1-Score 계산"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def get_model_info(safety_system: SafetySystem) -> Dict[str, Any]:
    """사용 중인 모델 정보 수집"""
    model_info = {
        "violation_model": {
            "name": "YOLO Violation (PPE 감지)",
            "path": config.Paths.YOLO_VIOLATION_MODEL,
            "exists": os.path.exists(config.Paths.YOLO_VIOLATION_MODEL),
            "device": str(safety_system.device) if safety_system.violation_model else None,
            "uses_trt": safety_system.violation_uses_trt if hasattr(safety_system, 'violation_uses_trt') else False
        },
        "pose_model": {
            "name": "YOLO Pose (사람 감지)",
            "path": config.Paths.YOLO_POSE_MODEL,
            "exists": os.path.exists(config.Paths.YOLO_POSE_MODEL),
            "device": str(safety_system.device) if safety_system.pose_model else None,
            "uses_trt": safety_system.pose_uses_trt if hasattr(safety_system, 'pose_uses_trt') else False
        },
        "face_model": {
            "name": "YOLO Face (얼굴 감지)",
            "path": config.Paths.YOLO_FACE_MODEL,
            "exists": os.path.exists(config.Paths.YOLO_FACE_MODEL),
            "device": str(safety_system.device) if safety_system.face_model else None,
            "uses_trt": safety_system.face_uses_trt if hasattr(safety_system, 'face_uses_trt') else False
        },
        "face_recognition": {
            "name": "AdaFace/InsightFace (얼굴 인식)",
            "adaface_path": config.Paths.ADAFACE_MODEL,
            "adaface_exists": os.path.exists(config.Paths.ADAFACE_MODEL),
            "device": str(safety_system.device) if safety_system.face_analyzer else None
        }
    }
    
    # 모델 파일 크기 추가
    for key, info in model_info.items():
        if key == "face_recognition":
            if info["adaface_exists"]:
                info["adaface_size_mb"] = round(os.path.getsize(info["adaface_path"]) / (1024 * 1024), 2)
        else:
            if info["exists"]:
                info["size_mb"] = round(os.path.getsize(info["path"]) / (1024 * 1024), 2)
    
    return model_info


def benchmark_model(
    model: Any,
    model_name: str,
    input_size: tuple = (640, 480),
    num_iterations: int = 50,
    warmup: int = 5
) -> Dict[str, Any]:
    """모델 성능 벤치마크"""
    if model is None:
        return {
            "error": "모델이 None입니다",
            "avg_inference_time_ms": 0.0,
            "fps": 0.0,
            "min_time_ms": 0.0,
            "max_time_ms": 0.0,
            "std_time_ms": 0.0
        }
    
    try:
        # 더미 입력 생성
        dummy_input = np.random.randint(0, 255, (input_size[1], input_size[0], 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(warmup):
            try:
                _ = model(dummy_input, verbose=False)
            except:
                pass
        
        # 실제 벤치마크
        inference_times = []
        for i in range(num_iterations):
            start_time = time.time()
            try:
                _ = model(dummy_input, verbose=False)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
            except Exception as e:
                logger.warning(f"{model_name} 추론 중 오류 (반복 {i+1}): {e}")
                continue
        
        if not inference_times:
            return {
                "error": "모든 추론이 실패했습니다",
                "avg_inference_time_ms": 0.0,
                "fps": 0.0,
                "min_time_ms": 0.0,
                "max_time_ms": 0.0,
                "std_time_ms": 0.0
            }
        
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_time = np.std(inference_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "avg_inference_time_ms": round(avg_time, 2),
            "fps": round(fps, 2),
            "min_time_ms": round(min_time, 2),
            "max_time_ms": round(max_time, 2),
            "std_time_ms": round(std_time, 2),
            "num_iterations": len(inference_times),
            "success_rate": round(len(inference_times) / num_iterations * 100, 1)
        }
    except Exception as e:
        logger.error(f"{model_name} 벤치마크 중 오류: {e}", exc_info=True)
        return {
            "error": str(e),
            "avg_inference_time_ms": 0.0,
            "fps": 0.0,
            "min_time_ms": 0.0,
            "max_time_ms": 0.0,
            "std_time_ms": 0.0
        }


def benchmark_face_recognition(
    face_analyzer: Any,
    model_name: str,
    input_size: tuple = (640, 480),
    num_iterations: int = 50,
    warmup: int = 5
) -> Dict[str, Any]:
    """얼굴 인식 모델 성능 벤치마크"""
    if face_analyzer is None:
        return {
            "error": "얼굴 인식 모델이 None입니다",
            "avg_inference_time_ms": 0.0,
            "fps": 0.0
        }
    
    try:
        # 더미 얼굴 이미지 생성 (112x112, InsightFace 입력 크기)
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # rec_model 접근
        rec_model = None
        if hasattr(face_analyzer, 'models') and 'recognition' in face_analyzer.models:
            rec_model = face_analyzer.models['recognition']
        elif hasattr(face_analyzer, 'rec_model'):
            rec_model = face_analyzer.rec_model
        
        if rec_model is None:
            return {
                "error": "recognition 모델을 찾을 수 없습니다",
                "avg_inference_time_ms": 0.0,
                "fps": 0.0
            }
        
        # Warmup
        for _ in range(warmup):
            try:
                _ = rec_model.get_feat(dummy_face)
            except:
                pass
        
        # 실제 벤치마크
        inference_times = []
        for i in range(num_iterations):
            start_time = time.time()
            try:
                _ = rec_model.get_feat(dummy_face)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
            except Exception as e:
                logger.warning(f"{model_name} 추론 중 오류 (반복 {i+1}): {e}")
                continue
        
        if not inference_times:
            return {
                "error": "모든 추론이 실패했습니다",
                "avg_inference_time_ms": 0.0,
                "fps": 0.0
            }
        
        avg_time = np.mean(inference_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "avg_inference_time_ms": round(avg_time, 2),
            "fps": round(fps, 2),
            "min_time_ms": round(np.min(inference_times), 2),
            "max_time_ms": round(np.max(inference_times), 2),
            "std_time_ms": round(np.std(inference_times), 2),
            "num_iterations": len(inference_times),
            "success_rate": round(len(inference_times) / num_iterations * 100, 1)
        }
    except Exception as e:
        logger.error(f"{model_name} 벤치마크 중 오류: {e}", exc_info=True)
        return {
            "error": str(e),
            "avg_inference_time_ms": 0.0,
            "fps": 0.0
        }


def evaluate_model_on_images(
    model: Any,
    model_name: str,
    image_paths: List[str],
    task_type: str = "detect",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    실제 이미지들에 대해 모델 성능 평가
    Ground truth가 없으므로, 모델의 일관성과 감지율을 측정합니다.
    """
    if model is None:
        return {
            "error": "모델이 None입니다",
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_detections": 0.0,
            "num_images": 0
        }
    
    if not image_paths:
        return {
            "error": "테스트 이미지가 없습니다",
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_detections": 0.0,
            "num_images": 0
        }
    
    total_detections = 0
    successful_detections = 0
    total_images = 0
    detection_counts = []
    
    logger.info(f"  {len(image_paths)}개 이미지로 {model_name} 평가 중...")
    
    for img_path in image_paths[:50]:  # 최대 50개 이미지만 사용
        try:
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 모델 추론
            results = model(img, conf=confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        if conf >= confidence_threshold:
                            detections.append((int(x1), int(y1), int(x2), int(y2)))
            
            if detections:
                successful_detections += 1
                total_detections += len(detections)
                detection_counts.append(len(detections))
            
            total_images += 1
            
        except Exception as e:
            logger.warning(f"  이미지 {img_path} 처리 중 오류: {e}")
            continue
    
    if total_images == 0:
        return {
            "error": "처리된 이미지가 없습니다",
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_detections": 0.0,
            "num_images": 0
        }
    
    # Ground truth가 없으므로, 감지율과 일관성을 측정
    detection_rate = successful_detections / total_images if total_images > 0 else 0.0
    avg_detections = total_detections / total_images if total_images > 0 else 0.0
    
    # 일관성 측정 (표준 편차)
    std_detections = np.std(detection_counts) if detection_counts else 0.0
    
    # Ground truth가 없으므로, 감지율을 recall로 근사
    # Precision은 감지된 객체의 평균 신뢰도로 근사
    # 실제로는 ground truth가 필요하지만, 여기서는 감지율 기반 지표 제공
    
    return {
        "detection_rate": round(detection_rate, 4),  # 감지율 (Recall 근사)
        "avg_detections": round(avg_detections, 2),
        "std_detections": round(std_detections, 2),
        "num_images": total_images,
        "successful_images": successful_detections,
        "total_detections": total_detections,
        "note": "Ground truth가 없어 정확한 Precision/Recall/F1 계산 불가. 감지율 기반 지표 제공."
    }


def get_test_images(log_folder: str, max_images: int = 50) -> List[str]:
    """로그 폴더에서 테스트 이미지 경로 수집"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    for ext in image_extensions:
        pattern = os.path.join(log_folder, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))
    
    # 최신 이미지 우선 선택
    image_paths.sort(key=os.path.getmtime, reverse=True)
    
    return image_paths[:max_images]


def collect_performance_metrics(safety_system: SafetySystem) -> Dict[str, Any]:
    """전체 성능 지표 수집"""
    logger.info("=" * 60)
    logger.info("모델 성능 지표 수집 시작")
    logger.info("=" * 60)
    
    # 모델 정보 수집
    model_info = get_model_info(safety_system)
    
    # 벤치마크 설정
    input_size = (640, 480)  # 실제 사용되는 입력 크기
    num_iterations = 50  # 벤치마크 반복 횟수
    warmup = 5  # 워밍업 횟수
    
    performance_results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "device": str(safety_system.device),
            "input_size": f"{input_size[0]}x{input_size[1]}"
        },
        "models": {}
    }
    
    # 테스트 이미지 수집
    test_images = get_test_images(config.Paths.LOG_FOLDER, max_images=50)
    logger.info(f"\n📸 테스트 이미지 {len(test_images)}개 발견")
    
    # 1. Violation 모델 벤치마크
    logger.info("\n📊 Violation 모델 벤치마크 중...")
    if safety_system.violation_model:
        violation_perf = benchmark_model(
            safety_system.violation_model,
            "Violation 모델",
            input_size,
            num_iterations,
            warmup
        )
        
        # 실제 이미지 평가
        if test_images:
            violation_eval = evaluate_model_on_images(
                safety_system.violation_model,
                "Violation 모델",
                test_images,
                task_type="detect",
                confidence_threshold=config.Thresholds.YOLO_CONFIDENCE
            )
            violation_perf.update(violation_eval)
        
        performance_results["models"]["violation"] = {
            **model_info["violation_model"],
            **violation_perf
        }
        logger.info(f"   평균 추론 시간: {violation_perf.get('avg_inference_time_ms', 0):.2f}ms")
        logger.info(f"   FPS: {violation_perf.get('fps', 0):.2f}")
        if "detection_rate" in violation_perf:
            logger.info(f"   감지율: {violation_perf.get('detection_rate', 0):.2%}")
    else:
        performance_results["models"]["violation"] = {
            **model_info["violation_model"],
            "error": "모델이 로드되지 않았습니다"
        }
    
    # 2. Pose 모델 벤치마크
    logger.info("\n📊 Pose 모델 벤치마크 중...")
    if safety_system.pose_model:
        pose_perf = benchmark_model(
            safety_system.pose_model,
            "Pose 모델",
            input_size,
            num_iterations,
            warmup
        )
        
        # 실제 이미지 평가
        if test_images:
            pose_eval = evaluate_model_on_images(
                safety_system.pose_model,
                "Pose 모델",
                test_images,
                task_type="pose",
                confidence_threshold=config.Thresholds.POSE_CONFIDENCE
            )
            pose_perf.update(pose_eval)
        
        performance_results["models"]["pose"] = {
            **model_info["pose_model"],
            **pose_perf
        }
        logger.info(f"   평균 추론 시간: {pose_perf.get('avg_inference_time_ms', 0):.2f}ms")
        logger.info(f"   FPS: {pose_perf.get('fps', 0):.2f}")
        if "detection_rate" in pose_perf:
            logger.info(f"   감지율: {pose_perf.get('detection_rate', 0):.2%}")
    else:
        performance_results["models"]["pose"] = {
            **model_info["pose_model"],
            "error": "모델이 로드되지 않았습니다"
        }
    
    # 3. Face 모델 벤치마크
    logger.info("\n📊 Face 모델 벤치마크 중...")
    if safety_system.face_model:
        face_perf = benchmark_model(
            safety_system.face_model,
            "Face 모델",
            input_size,
            num_iterations,
            warmup
        )
        
        # 실제 이미지 평가
        if test_images:
            face_eval = evaluate_model_on_images(
                safety_system.face_model,
                "Face 모델",
                test_images,
                task_type="detect",
                confidence_threshold=config.Thresholds.FACE_DETECTION_CONFIDENCE
            )
            face_perf.update(face_eval)
        
        performance_results["models"]["face_detection"] = {
            **model_info["face_model"],
            **face_perf
        }
        logger.info(f"   평균 추론 시간: {face_perf.get('avg_inference_time_ms', 0):.2f}ms")
        logger.info(f"   FPS: {face_perf.get('fps', 0):.2f}")
        if "detection_rate" in face_perf:
            logger.info(f"   감지율: {face_perf.get('detection_rate', 0):.2%}")
    else:
        performance_results["models"]["face_detection"] = {
            **model_info["face_model"],
            "error": "모델이 로드되지 않았습니다"
        }
    
    # 4. Face Recognition 모델 벤치마크
    logger.info("\n📊 Face Recognition 모델 벤치마크 중...")
    if safety_system.face_analyzer:
        face_rec_perf = benchmark_face_recognition(
            safety_system.face_analyzer,
            "Face Recognition 모델",
            input_size,
            num_iterations,
            warmup
        )
        performance_results["models"]["face_recognition"] = {
            **model_info["face_recognition"],
            **face_rec_perf
        }
        logger.info(f"   평균 추론 시간: {face_rec_perf.get('avg_inference_time_ms', 0):.2f}ms")
        logger.info(f"   FPS: {face_rec_perf.get('fps', 0):.2f}")
    else:
        performance_results["models"]["face_recognition"] = {
            **model_info["face_recognition"],
            "error": "모델이 로드되지 않았습니다"
        }
    
    return performance_results


def generate_report(performance_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """성능 리포트 생성"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("모델 성능 지표 리포트")
    report_lines.append("=" * 80)
    report_lines.append(f"생성 시간: {performance_results['timestamp']}")
    report_lines.append(f"디바이스: {performance_results['system_info']['device']}")
    report_lines.append(f"입력 크기: {performance_results['system_info']['input_size']}")
    report_lines.append("")
    
    for model_key, model_data in performance_results["models"].items():
        report_lines.append("-" * 80)
        report_lines.append(f"모델: {model_data.get('name', model_key)}")
        report_lines.append("-" * 80)
        
        if "error" in model_data:
            report_lines.append(f"  ❌ 오류: {model_data['error']}")
        else:
            if "path" in model_data:
                report_lines.append(f"  경로: {model_data['path']}")
                if model_data.get('exists'):
                    report_lines.append(f"  파일 크기: {model_data.get('size_mb', 'N/A')} MB")
            if "adaface_path" in model_data:
                report_lines.append(f"  AdaFace 경로: {model_data['adaface_path']}")
                if model_data.get('adaface_exists'):
                    report_lines.append(f"  AdaFace 파일 크기: {model_data.get('adaface_size_mb', 'N/A')} MB")
            
            report_lines.append(f"  디바이스: {model_data.get('device', 'N/A')}")
            report_lines.append(f"  TensorRT 사용: {model_data.get('uses_trt', False)}")
            report_lines.append("")
            report_lines.append("  성능 지표 (추론 속도):")
            report_lines.append(f"    - 평균 추론 시간: {model_data.get('avg_inference_time_ms', 0):.2f} ms")
            report_lines.append(f"    - FPS: {model_data.get('fps', 0):.2f}")
            report_lines.append(f"    - 최소 시간: {model_data.get('min_time_ms', 0):.2f} ms")
            report_lines.append(f"    - 최대 시간: {model_data.get('max_time_ms', 0):.2f} ms")
            report_lines.append(f"    - 표준 편차: {model_data.get('std_time_ms', 0):.2f} ms")
            report_lines.append(f"    - 성공률: {model_data.get('success_rate', 0):.1f}%")
            report_lines.append(f"    - 반복 횟수: {model_data.get('num_iterations', 0)}")
            
            # 정확도 지표 (Precision, Recall, F1-Score)
            if "precision" in model_data:
                report_lines.append("")
                report_lines.append("  정확도 지표:")
                report_lines.append(f"    - Precision: {model_data.get('precision', 0):.4f}")
                report_lines.append(f"    - Recall: {model_data.get('recall', 0):.4f}")
                report_lines.append(f"    - F1-Score: {model_data.get('f1_score', 0):.4f}")
                report_lines.append(f"    - TP: {model_data.get('tp', 0)}")
                report_lines.append(f"    - FP: {model_data.get('fp', 0)}")
                report_lines.append(f"    - FN: {model_data.get('fn', 0)}")
            elif "detection_rate" in model_data:
                report_lines.append("")
                report_lines.append("  정확도 지표 (Ground Truth 없음):")
                report_lines.append(f"    - 감지율 (Detection Rate): {model_data.get('detection_rate', 0):.2%}")
                report_lines.append(f"    - 평균 감지 수: {model_data.get('avg_detections', 0):.2f}")
                report_lines.append(f"    - 감지 표준 편차: {model_data.get('std_detections', 0):.2f}")
                report_lines.append(f"    - 평가 이미지 수: {model_data.get('num_images', 0)}")
                if "note" in model_data:
                    report_lines.append(f"    - 참고: {model_data.get('note', '')}")
        
        report_lines.append("")
    
    # 전체 요약
    report_lines.append("=" * 80)
    report_lines.append("전체 요약")
    report_lines.append("=" * 80)
    
    total_inference_time = 0.0
    for model_data in performance_results["models"].values():
        if "avg_inference_time_ms" in model_data:
            total_inference_time += model_data["avg_inference_time_ms"]
    
    report_lines.append(f"전체 추론 시간 (병렬 실행 시): {total_inference_time:.2f} ms")
    report_lines.append(f"예상 전체 FPS: {1000.0 / total_inference_time if total_inference_time > 0 else 0:.2f}")
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # 파일로 저장
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(config.Paths.LOG_FOLDER, f"model_performance_{timestamp}.txt")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # JSON 파일도 저장
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(performance_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ 리포트 저장 완료:")
    logger.info(f"   텍스트: {output_file}")
    logger.info(f"   JSON: {json_file}")
    
    return report_text


def estimate_rtx2080ti_performance(
    current_performance: Dict[str, Any],
    use_tensorrt: bool = False,
    multi_gpu: bool = True
) -> Dict[str, Any]:
    """
    RTX 2080Ti 2대 환경에서의 예상 성능 계산
    
    성능 향상 계수:
    - MPS -> CUDA: 약 1.5-2.0배 (CUDA가 더 최적화됨)
    - CUDA -> TensorRT: 약 2.0-3.0배 (TensorRT 최적화)
    - Multi-GPU: 병렬 처리로 지연 시간 감소 (약 1.3-1.5배)
    """
    if "error" in current_performance or "avg_inference_time_ms" not in current_performance:
        return current_performance
    
    base_time = current_performance["avg_inference_time_ms"]
    
    # MPS -> CUDA 변환 계수 (CUDA가 일반적으로 더 빠름)
    cuda_speedup = 1.7  # CUDA가 MPS보다 약 1.7배 빠름
    
    # TensorRT 사용 시 추가 속도 향상
    if use_tensorrt:
        tensorrt_speedup = 2.5  # TensorRT가 PyTorch보다 약 2.5배 빠름
        estimated_time = base_time / (cuda_speedup * tensorrt_speedup)
        speedup_factor = cuda_speedup * tensorrt_speedup
    else:
        estimated_time = base_time / cuda_speedup
        speedup_factor = cuda_speedup
    
    # Multi-GPU 분산 처리 (병렬 실행으로 지연 시간 감소)
    if multi_gpu:
        # GPU 0: Violation/Pose, GPU 1: Face/Face Recognition
        # 병렬 실행으로 전체 파이프라인 시간 감소
        multi_gpu_speedup = 1.4  # 약 1.4배 향상
        estimated_time = estimated_time / multi_gpu_speedup
        speedup_factor *= multi_gpu_speedup
    
    estimated_fps = 1000.0 / estimated_time if estimated_time > 0 else 0.0
    
    result = current_performance.copy()
    result.update({
        "estimated_rtx2080ti_time_ms": round(estimated_time, 2),
        "estimated_rtx2080ti_fps": round(estimated_fps, 2),
        "speedup_factor": round(speedup_factor, 2),
        "use_tensorrt": use_tensorrt,
        "multi_gpu": multi_gpu,
        "gpu_config": "RTX 2080Ti x2 (CUDA)" + (" + TensorRT" if use_tensorrt else "")
    })
    
    return result


def generate_rtx2080ti_report(
    performance_results: Dict[str, Any],
    use_tensorrt: bool = False,
    output_file: Optional[str] = None
) -> str:
    """RTX 2080Ti 2대 환경 예상 성능 리포트 생성"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RTX 2080Ti 2대 환경 예상 성능 지표 리포트")
    report_lines.append("=" * 80)
    report_lines.append(f"생성 시간: {datetime.now().isoformat()}")
    report_lines.append(f"GPU 설정: RTX 2080Ti x2 (CUDA)" + (" + TensorRT" if use_tensorrt else " (PyTorch)"))
    report_lines.append(f"Multi-GPU: 활성화 (GPU 0: Violation/Pose, GPU 1: Face/Face Recognition)")
    report_lines.append("")
    report_lines.append("⚠️  참고: 이는 현재 MPS 환경 성능을 기반으로 한 추정치입니다.")
    report_lines.append("⚠️  실제 성능은 환경, 모델 버전, 드라이버 등에 따라 달라질 수 있습니다.")
    report_lines.append("")
    
    total_estimated_time = 0.0
    
    for model_key, model_data in performance_results["models"].items():
        if "error" in model_data:
            continue
        
        estimated = estimate_rtx2080ti_performance(model_data, use_tensorrt, multi_gpu=True)
        
        report_lines.append("-" * 80)
        report_lines.append(f"모델: {model_data.get('name', model_key)}")
        report_lines.append("-" * 80)
        
        if "estimated_rtx2080ti_time_ms" in estimated:
            report_lines.append(f"  현재 환경 (MPS):")
            report_lines.append(f"    - 추론 시간: {model_data.get('avg_inference_time_ms', 0):.2f} ms")
            report_lines.append(f"    - FPS: {model_data.get('fps', 0):.2f}")
            report_lines.append("")
            report_lines.append(f"  예상 성능 (RTX 2080Ti x2):")
            report_lines.append(f"    - 추론 시간: {estimated['estimated_rtx2080ti_time_ms']:.2f} ms")
            report_lines.append(f"    - FPS: {estimated['estimated_rtx2080ti_fps']:.2f}")
            report_lines.append(f"    - 속도 향상: {estimated['speedup_factor']:.2f}x")
            report_lines.append(f"    - TensorRT: {'사용' if use_tensorrt else '미사용'}")
            
            if model_key != "face_recognition":  # Face Recognition은 별도 GPU에서 실행
                total_estimated_time += estimated['estimated_rtx2080ti_time_ms']
        
        report_lines.append("")
    
    # 전체 파이프라인 성능 (병렬 실행 고려)
    report_lines.append("=" * 80)
    report_lines.append("전체 파이프라인 예상 성능")
    report_lines.append("=" * 80)
    
    # GPU 0: Violation + Pose (순차 실행)
    violation_time = 0.0
    pose_time = 0.0
    face_time = 0.0
    
    for model_key, model_data in performance_results["models"].items():
        if "error" in model_data:
            continue
        estimated = estimate_rtx2080ti_performance(model_data, use_tensorrt, multi_gpu=False)
        if "estimated_rtx2080ti_time_ms" in estimated:
            if model_key == "violation":
                violation_time = estimated['estimated_rtx2080ti_time_ms']
            elif model_key == "pose":
                pose_time = estimated['estimated_rtx2080ti_time_ms']
            elif model_key == "face_detection":
                face_time = estimated['estimated_rtx2080ti_time_ms']
    
    # GPU 0에서 Violation과 Pose는 순차 실행
    gpu0_time = violation_time + pose_time
    
    # GPU 1에서 Face는 별도 실행 (병렬)
    # 전체 파이프라인 시간 = max(GPU0 시간, GPU1 시간) + 기타 처리 시간
    pipeline_time = max(gpu0_time, face_time) + 5.0  # 기타 처리 시간 5ms 추가
    pipeline_fps = 1000.0 / pipeline_time if pipeline_time > 0 else 0.0
    
    report_lines.append(f"GPU 0 (Violation + Pose): {gpu0_time:.2f} ms")
    report_lines.append(f"GPU 1 (Face Detection): {face_time:.2f} ms")
    report_lines.append(f"전체 파이프라인 시간: {pipeline_time:.2f} ms")
    report_lines.append(f"예상 전체 FPS: {pipeline_fps:.2f}")
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # 파일로 저장
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorrt_suffix = "_tensorrt" if use_tensorrt else ""
        output_file = os.path.join(
            config.Paths.LOG_FOLDER, 
            f"rtx2080ti_performance{tensorrt_suffix}_{timestamp}.txt"
        )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"\n✅ RTX 2080Ti 리포트 저장 완료: {output_file}")
    
    return report_text


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='모델 성능 지표 수집 스크립트')
    parser.add_argument('--test-dataset', type=str, default=None,
                        help='테스트 데이터셋 경로 (Ground Truth 포함)')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Ground Truth JSON 파일 경로')
    parser.add_argument('--rtx2080ti', action='store_true',
                        help='RTX 2080Ti 2대 환경 예상 성능 계산')
    parser.add_argument('--tensorrt', action='store_true',
                        help='TensorRT 사용 시 예상 성능 (--rtx2080ti와 함께 사용)')
    args = parser.parse_args()
    
    logger.info("모델 성능 지표 수집 스크립트")
    logger.info("=" * 60)
    
    if args.ground_truth:
        logger.info(f"⚠️  Ground Truth 파일 지정됨: {args.ground_truth}")
        logger.info("⚠️  Ground Truth 지원은 향후 구현 예정입니다.")
        logger.info("⚠️  현재는 감지율(Detection Rate) 기반 지표만 제공됩니다.")
    
    try:
        # SafetySystem 초기화
        logger.info("\n🔧 SafetySystem 초기화 중...")
        safety_system = SafetySystem()
        logger.info("✅ SafetySystem 초기화 완료")
        
        # 성능 지표 수집
        performance_results = collect_performance_metrics(safety_system)
        
        # RTX 2080Ti 예상 성능 계산
        if args.rtx2080ti:
            logger.info("\n🚀 RTX 2080Ti 2대 환경 예상 성능 계산 중...")
            rtx2080ti_report = generate_rtx2080ti_report(
                performance_results,
                use_tensorrt=args.tensorrt
            )
            print("\n" + rtx2080ti_report)
        
        # 리포트 생성
        logger.info("\n📝 리포트 생성 중...")
        report_text = generate_report(performance_results)
        
        # 콘솔에 출력
        print("\n" + report_text)
        
        logger.info("\n✅ 성능 지표 수집 완료!")
        logger.info("\n💡 참고: 정확한 Precision, Recall, F1-Score를 계산하려면 Ground Truth 데이터가 필요합니다.")
        logger.info("💡 현재는 저장된 로그 이미지를 사용하여 감지율(Detection Rate)을 측정합니다.")
        if not args.rtx2080ti:
            logger.info("💡 RTX 2080Ti 예상 성능을 보려면 --rtx2080ti 옵션을 사용하세요.")
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
