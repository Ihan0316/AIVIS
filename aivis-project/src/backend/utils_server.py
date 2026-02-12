# utils_server.py - 서버 관련 유틸리티 함수
"""
서버 관련 유틸리티 함수 모듈
GPU 모니터링, 응답 압축, 데이터 필터링 등
"""
import logging
import json
import gzip
from typing import Dict, Any

import torch
from aiohttp import web


def get_gpu_usage_stats() -> Dict[int, Dict[str, Any]]:
    """GPU 사용량 통계 가져오기"""
    if not torch.cuda.is_available():
        return {}
    
    stats: Dict[int, Dict[str, Any]] = {}
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            memory_total = props.total_memory / 1024**3  # GB
            memory_free = memory_total - memory_reserved
            
            stats[i] = {
                "name": props.name,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_total_gb": memory_total,
                "memory_free_gb": memory_free,
                "memory_util_percent": (memory_reserved / memory_total) * 100 if memory_total > 0 else 0
            }
    except Exception as e:
        logging.warning(f"GPU 사용량 체크 실패: {e}")
    
    return stats


def log_gpu_optimization_recommendations(
    stats_lock: Any,
    system_stats: Dict[str, Any],
    default_face_workers: int,
    default_yolo_workers: int
) -> None:
    """GPU 사용량 기반 최적화 권장사항 출력"""
    if not torch.cuda.is_available():
        return
    
    stats = get_gpu_usage_stats()
    if not stats:
        return
    
    logging.info("=" * 80)
    logging.info("📊 GPU 사용량 모니터링")
    logging.info("=" * 80)
    
    for gpu_id, gpu_stat in stats.items():
        mem_util = gpu_stat["memory_util_percent"]
        mem_free = gpu_stat["memory_free_gb"]
        
        logging.info(f"GPU {gpu_id}: {gpu_stat['name']}")
        logging.info(f"  메모리: {gpu_stat['memory_reserved_gb']:.2f}GB / {gpu_stat['memory_total_gb']:.2f}GB ({mem_util:.1f}%)")
        logging.info(f"  여유 메모리: {mem_free:.2f}GB")
        
        # 최적화 권장사항
        if mem_util > 90:
            logging.warning(f"  ⚠️ GPU {gpu_id} 메모리 부족! 배치 크기 감소 또는 모델 입력 크기 감소 고려")
        elif mem_util < 50 and mem_free > 2:
            logging.info(f"  💡 GPU {gpu_id} 메모리 여유: 배치 크기 증가 가능")
        
        if gpu_id == 0:
            # GPU 0 (YOLO): 워커 수 체크
            if mem_util > 85:
                logging.warning(f"  ⚠️ GPU 0 과부하: YOLO 워커 수 감소 고려 (현재: {default_yolo_workers})")
        elif gpu_id == 1:
            # GPU 1 (Face): 워커 수 체크
            if mem_util > 85:
                logging.warning(f"  ⚠️ GPU 1 과부하: Face 워커 수 감소 고려 (현재: {default_face_workers})")
    
    # 멀티 GPU 균형 체크
    if len(stats) >= 2:
        gpu0_util = stats[0]["memory_util_percent"]
        gpu1_util = stats[1]["memory_util_percent"]
        util_diff = abs(gpu0_util - gpu1_util)
        
        if util_diff > 30:
            logging.warning(f"  ⚠️ GPU 사용률 불균형: GPU0={gpu0_util:.1f}%, GPU1={gpu1_util:.1f}% (차이: {util_diff:.1f}%)")
            logging.warning(f"     → 워커 수 재조정 고려 (현재: YOLO={default_yolo_workers}, Face={default_face_workers})")
        else:
            logging.info(f"  ✅ GPU 사용률 균형: GPU0={gpu0_util:.1f}%, GPU1={gpu1_util:.1f}%")
        
        # GPU 1 사용률이 너무 낮으면 경고
        if gpu1_util < 15:
            logging.warning(f"  ⚠️ GPU 1 (얼굴 인식) 사용률이 매우 낮습니다: {gpu1_util:.1f}%")
            logging.warning(f"     → 얼굴 인식이 활성화되어 있는지 확인하세요")
            logging.warning(f"     → Face 워커 수 증가 고려 (현재: {default_face_workers})")
    
    logging.info("=" * 80)
    
    # 통계 저장
    with stats_lock:
        system_stats["gpu_stats"] = stats


def create_compressed_response(data: Dict[str, Any], content_type: str = 'application/json') -> web.Response:
    """gzip 압축된 JSON 응답 생성"""
    try:
        json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        
        compressed_data = gzip.compress(json_data)
        
        response = web.Response(
            body=compressed_data,
            content_type=content_type,
            headers={
                'Content-Encoding': 'gzip',
                'Content-Length': str(len(compressed_data)),
                'Cache-Control': 'public, max-age=60',  # 1분 캐시
                'Vary': 'Accept-Encoding'
            }
        )
        return response
    except Exception as e:
        logging.error(f"압축 응답 생성 실패: {e}")
        # 폴백: 일반 JSON 응답
        return web.json_response(data)


def filter_model_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """모델 결과 데이터 필터링 - 필요한 데이터만 반환"""
    try:
        filtered_data = {
            "alerts": data.get("alerts", [])[-10:],  # 최근 10개 알림만
            "violations": data.get("violations", {}),
            "heatmap_counts": data.get("heatmap_counts", {}),
            "profile": data.get("profile", {}),
            "logs": data.get("logs", [])[-20:],  # 최근 20개 로그만
            "kpi_data": data.get("kpi_data", {}),
            "detected_workers": data.get("detected_workers", {})
        }
        
        # 빈 데이터 제거
        filtered_data = {k: v for k, v in filtered_data.items() if v}
        
        return filtered_data
    except Exception as e:
        logging.error(f"모델 결과 필터링 실패: {e}")
        return data

