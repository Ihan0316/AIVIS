"""
백그라운드 카메라 AI 처리 워커
2대 카메라를 독립적으로 처리하며, 프론트엔드 연결과 무관하게 24/7 실행
"""
import asyncio
import cv2
import time
import logging
import threading
import os
from typing import Dict, Optional
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config

# 카메라별 공유 버퍼 (전역)
camera_buffers: Dict[int, Dict] = {
    0: {
        "latest_frame": None,
        "latest_result": None,
        "thumbnail": None,
        "processing": False,
        "last_update": 0,
        "fps": 0.0
    },
    1: {
        "latest_frame": None,
        "latest_result": None,
        "thumbnail": None,
        "processing": False,
        "last_update": 0,
        "fps": 0.0
    }
}
buffer_lock = threading.Lock()


async def camera_worker(
    cam_id: int,
    camera_source: str,
    safety_system,
    storage_manager=None,
    db_service=None,
    fps: float = 30.0  # 30 FPS (실제 처리 속도에 맞춤, 프레임 드롭 방지)
):
    """
    백그라운드에서 항상 실행되는 AI 처리 워커
    
    Args:
        cam_id: 카메라 ID (0 또는 1)
        camera_source: 카메라 소스 (RTSP URL 또는 카메라 인덱스)
        safety_system: SafetySystem 인스턴스
        storage_manager: StorageManager 인스턴스 (선택)
        db_service: DatabaseService 인스턴스 (선택)
        fps: 목표 FPS (기본 30)
    """
    logging.info(f"🎥 Camera {cam_id} worker 시작 (소스: {camera_source})")
    
    # SafetySystem을 전역 변수에 할당 (process_single_frame이 읽을 수 있도록)
    # 순환 import를 피하기 위해 동적 import 사용
    if safety_system is not None:
        from state import safety_system_instance, safety_system_lock
        import state
        with safety_system_lock:
            # 이미 할당된 인스턴스가 있으면 덮어쓰지 않음 (main.py에서 할당한 값 유지)
            if state.safety_system_instance is None:
                state.safety_system_instance = safety_system
                logging.info(f"[CAM-{cam_id}] ✅ SafetySystem을 전역 변수에 할당 완료 (인스턴스 존재: {safety_system is not None})")
            else:
                logging.info(f"[CAM-{cam_id}] ℹ️ SafetySystem이 이미 할당되어 있습니다. 기존 인스턴스 유지.")
    else:
        logging.warning(f"[CAM-{cam_id}] ⚠️ safety_system이 None입니다! AI 처리가 불가능합니다.")
    
    # 카메라 초기화
    camera = None
    try:
        import platform
        # RTSP URL 또는 비디오 파일 경로인지 확인
        if camera_source.startswith('rtsp://') or camera_source.startswith('http://'):
            camera = cv2.VideoCapture(camera_source)
        elif camera_source.endswith('.mp4') or camera_source.endswith('.avi') or camera_source.endswith('.mov') or '\\' in camera_source or '/' in camera_source:
            # 비디오 파일 경로
            camera = cv2.VideoCapture(camera_source)
        else:
            # 카메라 인덱스 (숫자) - 플랫폼별 백엔드 사용
            cam_index = int(camera_source)
            if platform.system() == 'Darwin':  # macOS
                # AVFoundation 백엔드 사용 (Mac 기본 카메라 백엔드)
                camera = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
                logging.info(f"📹 Camera {cam_id}: Mac AVFoundation 백엔드 사용 (인덱스: {cam_index})")
            elif platform.system() == 'Windows':
                # DirectShow 백엔드 사용 (Windows)
                camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                logging.info(f"📹 Camera {cam_id}: Windows DirectShow 백엔드 사용 (인덱스: {cam_index})")
            else:
                # Linux 또는 기타 - 기본 백엔드
                camera = cv2.VideoCapture(cam_index)
                logging.info(f"📹 Camera {cam_id}: 기본 백엔드 사용 (인덱스: {cam_index})")
        
        if not camera.isOpened():
            logging.error(f"❌ Camera {cam_id} 열기 실패: {camera_source}")
            logging.error(f"💡 카메라가 연결되어 있는지 확인하거나 다른 인덱스를 시도해보세요 (0, 1, 2...)")
            return
        
        # 카메라 설정
        camera.set(cv2.CAP_PROP_FPS, fps)
        # 프레임 유지율 최대화: 버퍼 크기 증가 (MPS 환경 최적화: 3 -> 5, 프레임 드롭 방지)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # 버퍼 증가 (MPS 환경 프레임 유지율 향상)
        
        # 카메라 해상도 설정 (성능 최적화: 640x480)
        target_width = int(os.getenv('CAMERA_WIDTH', '640'))
        target_height = int(os.getenv('CAMERA_HEIGHT', '480'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        
        # 카메라 해상도 확인 및 로깅
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"📹 Camera {cam_id} 해상도: {actual_width}x{actual_height} (요청: {target_width}x{target_height})")
        
        # 카메라 해상도를 버퍼에 저장 (프론트엔드에서 바운딩 박스 계산 시 사용)
        with buffer_lock:
            camera_buffers[cam_id]["frame_width"] = actual_width
            camera_buffers[cam_id]["frame_height"] = actual_height
        
        logging.info(f"✅ Camera {cam_id} 초기화 완료")
        
    except Exception as e:
        logging.error(f"❌ Camera {cam_id} 초기화 오류: {e}", exc_info=True)
        return
    
    frame_count = 0
    fps_start_time = time.time()
    last_frame_time = 0
    min_frame_interval = 1.0 / fps
    last_thumbnail_time = 0  # 썸네일 생성 주기 제어
    
    # 메모리 누수 방지: 주기적 캐시 정리 (MPS 환경 최적화)
    last_cache_cleanup = time.time()
    CACHE_CLEANUP_INTERVAL = 30.0  # 60 -> 30초 (MPS 환경 메모리 관리 개선)
    
    while True:
        try:
            current_time = time.time()
            
            # 주기적 캐시 정리 (메모리 누수 방지 - 프레임 드롭 방지)
            if current_time - last_cache_cleanup > CACHE_CLEANUP_INTERVAL:
                try:
                    from state import recent_identity_cache, embedding_buffers, MAX_EMBEDDING_BUFFERS_PER_CAM
                    with buffer_lock:
                        # IdentityCache는 자동으로 크기 제한 및 만료 처리하므로
                        # 주기적 정리가 필요 없음 (get_recent 호출 시 자동 정리됨)
                        # 여기서는 embedding_buffers만 정리
                        if cam_id in embedding_buffers:
                            buffers = embedding_buffers[cam_id]
                            if len(buffers) > MAX_EMBEDDING_BUFFERS_PER_CAM:
                                # 가장 오래된 항목 제거
                                keys_to_remove = list(buffers.keys())[:-MAX_EMBEDDING_BUFFERS_PER_CAM]
                                for key in keys_to_remove:
                                    del buffers[key]
                    last_cache_cleanup = current_time
                except Exception as cleanup_error:
                    pass  # 캐시 정리 실패는 무시
            
            # 프레임 간격 제어 완화 (처리 속도에 맞춰 자동 조절)
            time_since_last = current_time - last_frame_time
            # 처리 중이 아니고 최소 간격보다 짧을 때만 대기 (처리 시간이 길면 대기하지 않음)
            with buffer_lock:
                processing = camera_buffers[cam_id]["processing"]
            if time_since_last < min_frame_interval and not processing:
                await asyncio.sleep(min_frame_interval - time_since_last)
            
            # 프레임 읽기
            ret, frame = camera.read()
            if not ret:
                # 비디오 파일인 경우 처음으로 되감기 (루프 재생)
                if isinstance(camera_source, str) and (camera_source.endswith('.mp4') or camera_source.endswith('.avi') or camera_source.endswith('.mov')):
                    logging.info(f"🔄 Camera {cam_id} 비디오 루프: 처음으로 되감기")
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = camera.read()
                    if not ret:
                        logging.error(f"❌ Camera {cam_id} 비디오 되감기 실패")
                        await asyncio.sleep(0.1)
                        continue
                else:
                    logging.warning(f"⚠️ Camera {cam_id} 프레임 읽기 실패")
                    await asyncio.sleep(0.1)
                    continue
            
            # 프레임 유지율 최대화: 처리 중이어도 프레임 큐에 추가 (프레임 드롭 방지)
            with buffer_lock:
                processing = camera_buffers[cam_id]["processing"]
                # 프레임 유지율 최대화: 큐가 가득 차지 않으면 프레임 추가
                from state import frame_queues, MAX_QUEUE_SIZE, queue_lock
                with queue_lock:
                    queue_full = cam_id in frame_queues and frame_queues[cam_id].qsize() >= MAX_QUEUE_SIZE
                # 큐가 가득 찬 경우 오래된 프레임 하나 제거하고 최신 프레임 추가 (프레임 드롭 방지)
                if processing and queue_full:
                    # 큐에서 오래된 프레임 하나 제거하고 최신 프레임 추가
                    with queue_lock:
                        try:
                            if cam_id in frame_queues:
                                frame_queues[cam_id].get_nowait()  # 오래된 프레임 제거
                        except queue.Empty:
                            pass
                    # continue 제거 - 프레임을 버리지 않고 계속 처리
                
                # 처리 상태 설정
                camera_buffers[cam_id]["processing"] = True
            
            # AI 처리 (비동기, MPS와 CPU 최대 활용)
            try:
                # 프레임 인코딩과 AI 처리를 병렬로 실행 (CPU와 MPS 동시 활용)
                loop = asyncio.get_event_loop()
                from frame_processor import process_single_frame
                from state import frame_processing_executor
                
                # 프레임 인코딩 함수 (CPU 활용, 빠른 처리)
                def encode_frame():
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        raise Exception("프레임 인코딩 실패")
                    return buffer.tobytes()
                
                # 프레임 인코딩을 병렬로 실행 (CPU 활용)
                encode_future = frame_processing_executor.submit(encode_frame)
                
                # 프레임 인코딩 완료 대기 (비동기, CPU 활용)
                frame_bytes = await loop.run_in_executor(None, encode_future.result)
                
                # AI 처리를 별도 스레드에서 실행 (MPS/GPU 활용, CPU와 병렬)
                # frame_processing_executor는 MPS와 CPU를 최대한 활용하도록 최적화됨
                processed_frame_bytes, result_data = await loop.run_in_executor(
                    frame_processing_executor,  # 전용 프레임 처리 Executor 사용 (MPS 최대 활용)
                    process_single_frame,
                    frame_bytes,
                    cam_id
                )
                
                
                # 타임스탬프 추가 (없으면 추가)
                if "ts_ms" not in result_data and "timestamp" not in result_data:
                    result_data["ts_ms"] = int(time.time() * 1000)
                    result_data["timestamp"] = result_data["ts_ms"]
                
                # cam_id 확인
                if "cam_id" not in result_data:
                    result_data["cam_id"] = cam_id
                
                result = result_data

                # state의 latest_frames, latest_result_data 업데이트 (MJPEG 스트림용)
                from state import latest_frames, latest_result_data, frame_lock as main_frame_lock
                with main_frame_lock:
                    latest_frames[cam_id] = processed_frame_bytes
                    latest_result_data[cam_id] = result

                # 결과 저장 (메모리 최적화: 큰 프레임은 저장하지 않음)
                with buffer_lock:
                    # latest_frame 저장 (MJPEG 스트림의 원본 영상용)
                    camera_buffers[cam_id]["latest_frame"] = frame
                    camera_buffers[cam_id]["latest_result"] = result
                    camera_buffers[cam_id]["last_update"] = current_time

                    # 썸네일 생성 (CPU 부하 감소를 위해 0.5초마다 갱신)
                    if current_time - last_thumbnail_time > 0.5:
                        camera_buffers[cam_id]["thumbnail"] = create_thumbnail(frame, result)
                        last_thumbnail_time = current_time
                    elif camera_buffers[cam_id]["thumbnail"] is None:
                        # 초기 썸네일 생성
                        camera_buffers[cam_id]["thumbnail"] = create_thumbnail(frame, result)
                        last_thumbnail_time = current_time

                    # FPS 계산 및 적응형 프레임 간격 조정 (MPS/CPU 최대 활용)
                    frame_count += 1
                    processing_time = current_time - last_frame_time
                    
                    # 적응형 프레임 간격 조정: 처리 시간이 길면 간격 완화
                    if processing_time > min_frame_interval * 1.5:
                        # 처리 시간이 목표 간격보다 50% 이상 길면 간격 완화
                        adaptive_frame_interval = min(processing_time * 0.8, min_frame_interval * 2.0)
                    else:
                        # 정상 처리 시 목표 간격 유지
                        adaptive_frame_interval = min_frame_interval
                    
                    elapsed = current_time - fps_start_time
                    if elapsed >= 1.0:
                        camera_buffers[cam_id]["fps"] = frame_count / elapsed
                        frame_count = 0
                        fps_start_time = current_time

                    camera_buffers[cam_id]["processing"] = False
                
                # 위반 사항이 있으면 이미지 저장 후 MongoDB에 저장
                if result.get("violations") and len(result.get("violations", [])) > 0:
                    # 위반 이미지 저장 (바운딩 박스 포함)
                    import config
                    from state import image_last_saved, IMAGE_SAVE_MIN_INTERVAL
                    # cv2, os, datetime은 이미 파일 상단에서 import됨 (6, 10, 12줄)
                    
                    violations_with_images = []
                    current_time = time.time()
                    
                    for violation in result["violations"]:
                        # recognized_name 우선 사용 (얼굴 인식 결과), 없으면 worker 사용
                        recognized_name = violation.get("recognized_name", "Unknown")
                        worker_name = violation.get("worker", "Unknown")
                        # recognized_name이 "Unknown"이 아니면 사용, 아니면 worker 사용
                        final_worker_name = recognized_name if recognized_name != "Unknown" else (worker_name if worker_name != "Unknown" else "알 수 없음")
                        violation_types = violation.get("violations", [])
                        safe_worker_name = "".join(c for c in final_worker_name if c.isalnum() or c in ('-', '_'))[:20]
                        
                        # 이미지 저장 중복 방지: 1초에 1건만 저장 (worker_name + cam_id 기준)
                        image_cache_key = f"{worker_name}_{cam_id}"
                        last_image_saved_time = image_last_saved.get(image_cache_key, 0)
                        
                        if current_time - last_image_saved_time < IMAGE_SAVE_MIN_INTERVAL:
                            # 1초 이내에 이미 저장했으면 건너뜀
                            # 이미지 경로 없이 violation 데이터만 저장
                            violation_copy = violation.copy()
                            violations_with_images.append(violation_copy)
                            continue
                        
                        # 이미지 저장 시간 업데이트
                        image_last_saved[image_cache_key] = current_time
                        
                        # 이미지 저장 경로 생성
                        now = datetime.now()
                        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                        
                        # 로그 폴더 확인 및 생성
                        log_folder = config.Paths.LOG_FOLDER
                        if not os.path.exists(log_folder):
                            os.makedirs(log_folder, exist_ok=True)
                        
                        # 이미지 파일명 생성 (첫 번째 위반 유형 사용)
                        violation_type = violation_types[0] if violation_types else "violation"
                        safe_event_type = "".join(c for c in violation_type if c.isalnum() or c in ('-', '_'))[:20]
                        image_filename = f"{timestamp_str}_CAM{cam_id}_{safe_worker_name}_{safe_event_type}.jpg"
                        image_path = os.path.join(log_folder, image_filename)
                        
                        # 이미지 저장 (바운딩 박스 포함)
                        try:
                            # 프레임 복사 (원본 프레임 보존)
                            frame_with_bbox = frame.copy()
                            
                            # PIL Image로 변환 (한글 텍스트 그리기용)
                            pil_image = Image.fromarray(cv2.cvtColor(frame_with_bbox, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            
                            # 폰트 크기 설정
                            font_size = 20
                            try:
                                korean_font = ImageFont.truetype(KOREAN_FONT.path if hasattr(KOREAN_FONT, 'path') else config.Paths.FONT_PATH, font_size)
                            except:
                                try:
                                    import config
                                    korean_font = ImageFont.truetype(config.Paths.FONT_PATH, font_size)
                                except:
                                    korean_font = ImageFont.load_default()
                            
                            # ⭐ 얼굴 박스는 저장 이미지에서 제거 (Person 박스만 표시)
                            # 위반 사항 바운딩 박스 그리기 (Person 박스만)
                            person_box = violation.get("person_box") or violation.get("bbox")
                            if person_box and len(person_box) == 4:
                                x1, y1, x2, y2 = map(int, person_box)
                                # OpenCV로 박스 그리기
                                frame_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                                cv2.rectangle(frame_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                pil_image = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(pil_image)
                                
                                # 작업자 이름과 위반 유형 텍스트 표시 (한글 지원)
                                text_parts = []
                                
                                # 작업자 이름 추가 (recognized_name 우선 사용)
                                display_worker_name = final_worker_name if final_worker_name != "알 수 없음" else worker_name
                                if display_worker_name and display_worker_name != "Unknown" and display_worker_name != "알 수 없음":
                                    text_parts.append(display_worker_name)
                                
                                # 위반 유형 추가
                                if violation_types:
                                    violation_text = ", ".join(violation_types)
                                    text_parts.append(violation_text)
                                
                                # 텍스트 표시
                                if text_parts:
                                    display_text = " | ".join(text_parts)
                                    # 배경 사각형 그리기
                                    text_bbox = draw.textbbox((x1, y1 - 25), display_text, font=korean_font)
                                    bg_coords = [text_bbox[0] - 3, text_bbox[1] - 2, text_bbox[2] + 3, text_bbox[3] + 2]
                                    draw.rectangle(bg_coords, fill=(0, 0, 0, 180))
                                    # 텍스트 그리기
                                    draw.text((x1, y1 - 25), display_text, font=korean_font, fill=(255, 0, 0))
                            
                            # PIL Image를 다시 OpenCV 형식으로 변환
                            frame_with_bbox = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                            
                            # 바운딩 박스가 그려진 프레임 저장
                            success = cv2.imwrite(image_path, frame_with_bbox)
                            
                            # 이미지 저장 성공 여부 확인
                            if success and os.path.exists(image_path):
                                file_size = os.path.getsize(image_path)
                                # 이미지 경로를 violation 데이터에 추가
                                violation_copy = violation.copy()
                                violation_copy["image_path"] = image_path
                                violations_with_images.append(violation_copy)
                                
                                # 유사도 매칭 결과 로깅 (오인식 디버깅용)
                                similarity = violation.get("similarity", None)
                                recognized_name = violation.get("recognized_name", "Unknown")
                                worker_name = violation.get("worker", "Unknown")
                                if similarity is not None:
                                    logging.info(f"[CAM-{cam_id}] ✅ 위반 이미지 저장 성공: {image_path} (크기: {file_size} bytes) | "
                                               f"인식결과={recognized_name}, worker={worker_name}, 유사도={similarity:.3f}")
                                else:
                                    logging.info(f"[CAM-{cam_id}] ✅ 위반 이미지 저장 성공: {image_path} (크기: {file_size} bytes) | "
                                               f"인식결과={recognized_name}, worker={worker_name}, 유사도=없음")
                            else:
                                logging.error(f"[CAM-{cam_id}] ❌ 위반 이미지 저장 실패: {image_path} (cv2.imwrite={success}, 파일존재={os.path.exists(image_path) if image_path else False})")
                                # 이미지 저장 실패해도 violation 데이터는 저장
                                violation_copy = violation.copy()
                                violations_with_images.append(violation_copy)
                        except Exception as e:
                            logging.error(f"[CAM-{cam_id}] 위반 이미지 저장 실패: {e}", exc_info=True)
                            # 이미지 저장 실패해도 violation 데이터는 저장
                            violation_copy = violation.copy()
                            violations_with_images.append(violation_copy)
                    
                    # 이미지 경로가 포함된 violation 데이터로 저장
                    await save_violations_to_db(
                        violations_with_images,
                        cam_id,
                        result.get("recognized_faces", []),
                        db_service
                    )
                
                # WebSocket으로 브로드캐스트 (가이드 스키마 호환)
                await broadcast_to_websockets(cam_id, result)
                
                # 로깅 (주기적으로)
                if frame_count % 30 == 0:
                    logging.info(
                        f"[CAM-{cam_id}] 처리 완료: "
                        f"얼굴={len(result.get('recognized_faces', []))}개, "
                        f"위반={len(result.get('violations', []))}개, "
                        f"FPS={camera_buffers[cam_id]['fps']:.1f}"
                    )
                
            except Exception as e:
                logging.error(f"❌ Camera {cam_id} AI 처리 오류: {e}", exc_info=True)
                with buffer_lock:
                    camera_buffers[cam_id]["processing"] = False
            
            last_frame_time = time.time()
            
        except asyncio.CancelledError:
            logging.info(f"🛑 Camera {cam_id} worker 취소됨")
            break
        except Exception as e:
            logging.error(f"❌ Camera {cam_id} worker 오류: {e}", exc_info=True)
            await asyncio.sleep(1)
    
    # 정리
    if camera:
        camera.release()
    logging.info(f"🛑 Camera {cam_id} worker 종료")


def create_thumbnail(frame: np.ndarray, result: Dict) -> Optional[np.ndarray]:
    """
    대시보드용 썸네일 생성 (바운딩 박스 포함)
    
    Args:
        frame: 원본 프레임
        result: AI 처리 결과
    
    Returns:
        썸네일 이미지 (320x240) 또는 None
    """
    try:
        thumb = frame.copy()
        
        # 얼굴 바운딩 박스 그리기
        for face in result.get("recognized_faces", []):
            bbox = face.get("box") or face.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(thumb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 이름 표시
                name = face.get("name", "Unknown")
                if name and name != "Unknown" and name != "알 수 없음":
                    cv2.putText(
                        thumb, name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
        
        # 위반 바운딩 박스 그리기
        for violation in result.get("violations", []):
            person_box = violation.get("person_box") or violation.get("bbox")
            if person_box and len(person_box) == 4:
                x1, y1, x2, y2 = map(int, person_box)
                cv2.rectangle(thumb, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # 위반 텍스트
                violation_types = violation.get("violations", [])
                if violation_types:
                    text = "⚠️ " + ", ".join(violation_types[:2])
                    cv2.putText(
                        thumb, text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )
        
        # 위반이 있으면 전체 프레임에 경고 표시
        if result.get("violation_count", 0) > 0:
            cv2.putText(
                thumb, "⚠️ VIOLATION",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        
        # 리사이즈 (320x240)
        thumb = cv2.resize(thumb, (320, 240))
        
        return thumb
        
    except Exception as e:
        logging.error(f"썸네일 생성 오류: {e}", exc_info=True)
        return None


async def save_violations_to_db(
    violations: list,
    cam_id: int,
    recognized_faces: list,
    db_service=None
):
    """
    위반 사항을 MongoDB에 저장 (배치 처리로 DB 부하 감소)
    
    Args:
        violations: 위반 사항 리스트
        cam_id: 카메라 ID
        recognized_faces: 인식된 얼굴 리스트
        db_service: DatabaseService 인스턴스
    """
    if not db_service or not db_service.is_connected():
        return
    
    # 배치 큐에 추가 (즉시 저장하지 않음)
    from state import violation_batch_queue, violation_batch_lock, VIOLATION_MIN_INTERVAL, violation_last_saved
    import time
    
    current_time = time.time()
    
    for violation in violations:
        worker_name = violation.get("worker", "Unknown")
        violation_types = violation.get("violations", [])
        
        for violation_type in violation_types:
            if not violation_type:
                continue
            
            # 중복 저장 방지: 같은 위반에 대해 최소 간격 내 저장 방지
            cache_key = f"{worker_name}_{violation_type}_{cam_id}"
            with violation_batch_lock:
                last_saved_time = violation_last_saved.get(cache_key, 0)
                if current_time - last_saved_time < VIOLATION_MIN_INTERVAL:
                    # 최소 간격 내에는 배치 큐에 추가하지 않음
                    continue
                violation_last_saved[cache_key] = current_time
            
            # 배치 큐에 추가
            batch_item = {
                'violations': [violation],
                'cam_id': cam_id,
                'recognized_faces': recognized_faces,
                'db_service': db_service,
                'timestamp': current_time
            }
            violation_batch_queue.put(batch_item)
    
    # 배치 처리로 대체됨 (위 코드는 제거)


def get_camera_buffer(cam_id: int) -> Optional[Dict]:
    """
    카메라 버퍼 가져오기 (스레드 안전)
    
    Args:
        cam_id: 카메라 ID
    
    Returns:
        카메라 버퍼 딕셔너리 또는 None
    """
    with buffer_lock:
        return camera_buffers.get(cam_id)


async def broadcast_to_websockets(cam_id: int, result: Dict):
    """
    WebSocket으로 AI 결과 브로드캐스트 (가이드 스키마 호환)
    
    Args:
        cam_id: 카메라 ID
        result: AI 처리 결과
    """
    try:
        # main.py의 connected_websockets 가져오기
        # 순환 import 방지를 위해 동적 import
        try:
            import importlib
            main_module = importlib.import_module('main')
            if hasattr(main_module, 'connected_websockets'):
                active_ws = main_module.connected_websockets
            else:
                return
        except (ImportError, AttributeError):
            # main 모듈이 아직 로드되지 않았거나 속성이 없으면 무시
            return
        
        if not active_ws:
            return
        
        # 가이드 스키마 메시지 형식
        message = {
            "type": "ai_result",
            "timestamp": result.get("timestamp") or result.get("ts_ms", int(time.time() * 1000)),
            "cam_id": cam_id,
            "faces": result.get("faces", []) or result.get("recognized_faces", []),
            "violations": result.get("violations", []),
            "ppe_detections": result.get("ppe_detections", []),
            "processing_time_ms": result.get("processing_time_ms", 0)
        }
        
        # 연결된 모든 WebSocket에 전송
        disconnected = set()
        for ws in active_ws.copy():  # copy로 반복 중 수정 방지
            try:
                # cam_id 필터링 (선택적 - 클라이언트가 특정 카메라만 원할 수 있음)
                # 여기서는 모든 연결에 전송하고, 클라이언트에서 필터링
                import json
                await ws.send_str(json.dumps(message))
            except Exception:
                disconnected.add(ws)
        
        # 끊어진 연결 제거
        for ws in disconnected:
            active_ws.discard(ws)
            
    except Exception:
        pass


def get_camera_thumbnail(cam_id: int) -> Optional[bytes]:
    """
    카메라 썸네일을 JPEG 바이트로 반환
    
    Args:
        cam_id: 카메라 ID
    
    Returns:
        JPEG 바이트 또는 None
    """
    with buffer_lock:
        buffer = camera_buffers.get(cam_id)
        if not buffer or buffer.get("thumbnail") is None:
            return None
        
        thumbnail = buffer["thumbnail"]
        ret, jpeg = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            return jpeg.tobytes()
        return None

