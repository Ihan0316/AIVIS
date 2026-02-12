"""
PPE 합성 및 임베딩 통합 스크립트
1. image 폴더의 모든 사진을 nanobanana로 보호구 착용 합성
2. 합성된 사진을 data/images 폴더로 복사
3. FAISS 임베딩 데이터베이스 구축
"""
import os
import sys
import time
import shutil
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
# from tqdm import tqdm  # 상세 로그를 위해 주석 처리

# Windows 콘솔 인코딩 설정
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    from google import genai
    NANOBANANA_AVAILABLE = True
except ImportError:
    NANOBANANA_AVAILABLE = False
    print("[WARN] nanobanana를 사용하려면 google-genai 패키지가 필요합니다: pip install google-genai")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# 경로 설정 - 현재 파일 기준으로 자동 계산
_current_file = Path(__file__).resolve()
script_dir = _current_file.parent  # scripts/
final_dir = script_dir.parent  # face/
IMAGE_DIR = final_dir / "image"  # 원본 이미지 폴더
DATA_IMAGES_DIR = final_dir / "data" / "images"  # 임베딩용 이미지 폴더
PROMPT_FILE = final_dir / "nanobanana" / "prompts" / "ppe_ko.txt"
ENV_FILE = final_dir / ".env"  # .env 파일 경로
MODEL = "gemini-2.5-flash-image"
RETRIES = 3  # 기본 재시도
SLEEP = 2.0  # 빠른 재시도 간격

# 성능 최적화 설정
MAX_WORKERS = 6  # 고속 모드 병렬 처리
MAX_IMAGE_SIZE = (1280, 1280)  # 처리 속도 최적화
ENABLE_PARALLEL = True  # 병렬 처리 활성화
print_lock = Lock()  # 출력 동기화용

# 사람당 최대 PPE 합성 개수 (비PPE:PPE ≈ 1:3 권장)
# 전략적 배분: 실제 현장 상황을 반영한 다양한 PPE 조합
# ⭐ 개선: 안전모+마스크 동시 착용, 안전조끼 추가로 더 현실적인 인식
MAX_PPE_PER_PERSON = 12  # 총 12장 합성 (다양한 조합)

# PPE 종류별 합성 비율 (실제 현장 상황 반영)
PPE_STRATEGY = {
    'helmet_only': 3,      # 안전모만 (얼굴 보임 ✅) - 가장 기본
    'mask_only': 2,        # 마스크만 (얼굴 일부 가림 ⚠️)
    'helmet_mask': 3,      # 안전모+마스크 (실제 현장에서 가장 흔함 ⭐)
    'helmet_vest': 2,      # 안전모+조끼 (상체 특징 변화)
    'vest_only': 2,        # 조끼만 (상체 특징 학습용)
}
# 총: 3+2+3+2+2 = 12장
# 얼굴 완전 노출: 5장 (helmet_only 3 + vest_only 2)
# 얼굴 부분 가림: 5장 (mask_only 2 + helmet_mask 3)
# 상체 변화: 4장 (helmet_vest 2 + vest_only 2)

# 작업 디렉토리를 스크립트 디렉토리로 변경
os.chdir(str(script_dir))

# 이미지 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
if HEIC_SUPPORT:
    IMAGE_EXTENSIONS.add(".heic")


def load_image(image_path: Path, optimize_size: bool = True) -> Image.Image:
    """이미지를 로드하고 RGB로 변환, 필요시 크기 최적화"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # 큰 이미지 리사이즈 (API 처리 속도 향상)
    if optimize_size and MAX_IMAGE_SIZE:
        width, height = img.size
        max_width, max_height = MAX_IMAGE_SIZE
        if width > max_width or height > max_height:
            # 비율 유지하며 리사이즈
            ratio = min(max_width / width, max_height / height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    return img


def synthesize_ppe(image_path: Path, output_path: Path, prompt: str, client, model_name: str) -> bool:
    """
    단일 이미지에 PPE를 합성합니다.
    
    Args:
        image_path: 원본 이미지 경로
        output_path: 출력 이미지 경로
        prompt: PPE 합성 프롬프트
        client: Gemini API 클라이언트
        model_name: 사용할 모델 이름
    
    Returns:
        성공 여부
    """
    try:
        img = load_image(image_path)
        
        # 재시도 로직
        for attempt in range(RETRIES + 1):
            try:
                # API 호출
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, img],
                )
                
                # 응답에서 이미지 추출
                out_img = None
                for cand in resp.candidates:
                    if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                        for part in cand.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                out_img = Image.open(BytesIO(part.inline_data.data))
                                break
                    if out_img:
                        break
                
                if not out_img:
                    # 후보에서 직접 이미지 찾기 시도
                    for cand in resp.candidates:
                        if hasattr(cand, 'content'):
                            if hasattr(cand.content, 'parts'):
                                for part in cand.content.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        out_img = Image.open(BytesIO(part.inline_data.data))
                                        break
                            elif hasattr(cand.content, 'inline_data') and cand.content.inline_data:
                                out_img = Image.open(BytesIO(cand.content.inline_data.data))
                                break
                    
                    if not out_img:
                        raise RuntimeError("응답에서 이미지를 찾을 수 없음")
                
                # RGB 변환 및 저장
                if out_img.mode != "RGB":
                    out_img = out_img.convert("RGB")
                
                # 출력 디렉토리 생성
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out_img.save(output_path, quality=90)
                return True
                
            except Exception as e:
                if attempt < RETRIES:
                    # 429 등 레이트리밋 대응: 가변 백오프
                    time.sleep(SLEEP + attempt * 5.0)
                    continue
                else:
                    # 실패 메시지는 상위에서 출력
                    raise e
        
        return False
        
    except Exception as e:
        # 상세 오류 정보 출력
        import traceback
        error_msg = str(e)
        print(f"\n      오류 상세: {error_msg}")
        if "API" in error_msg or "key" in error_msg.lower() or "auth" in error_msg.lower():
            print(f"      ⚠️ API 키 또는 인증 문제일 수 있습니다.")
        return False


def process_single_image(args):
    """단일 이미지 처리 함수 (병렬 처리용)"""
    img_file, output_original, output_ppe, prompt, client, model_name, idx, total = args
    
    original_name = img_file.name
    result = {"success": False, "skipped": False, "failed": False, "name": original_name}
    
    # 원본 이미지 복사 (아직 없으면)
    if not output_original.exists():
        shutil.copy2(img_file, output_original)
        with print_lock:
            print(f"    [{idx}/{total}] 원본 복사: {original_name}")
    
    # PPE 합성본 생성 (이미 있으면 건너뜀)
    if output_ppe.exists():
        result["skipped"] = True
        with print_lock:
            print(f"    [{idx}/{total}] 건너뜀 (이미 존재): {original_name}")
        return result
    
    # PPE 합성 시도
    with print_lock:
        print(f"    [{idx}/{total}] PPE 합성 중: {original_name}...", end=" ", flush=True)
    
    if synthesize_ppe(img_file, output_ppe, prompt, client, model_name):
        result["success"] = True
        with print_lock:
            print("✅")
    else:
        result["failed"] = True
        with print_lock:
            print("❌")
    
    return result


def process_person_folder(person_dir: Path, data_person_dir: Path, prompt: str, client, model_name: str):
    """
    한 사람의 폴더 내 모든 이미지를 처리합니다 (병렬 처리 지원).
    
    Args:
        person_dir: 원본 이미지 폴더 (image/[이름]/)
        data_person_dir: 데이터 이미지 폴더 (data/images/[이름]/)
        prompt: PPE 합성 프롬프트
        client: Gemini API 클라이언트
        model_name: 사용할 모델 이름
    """
    person_name = person_dir.name
    print(f"\n▶ '{person_name}' 폴더 처리 중...")
    
    # 출력 폴더 생성
    data_person_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 찾기
    image_files = [f for f in person_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    
    # 이미 PPE 합성된 파일 제외
    image_files = [f for f in image_files 
                   if not (f.name.startswith("ppe_") or "_ppe." in f.name.lower())]
    
    if not image_files:
        print(f"  ℹ️ 처리할 이미지 파일이 없습니다.")
        return
    
    print(f"  📸 총 {len(image_files)}개 이미지 발견")
    
    # 모든 원본 이미지는 우선 복사 (비PPE 데이터 확보)
    copied = 0
    for img_file in image_files:
        dst = data_person_dir / img_file.name
        if not dst.exists():
            try:
                shutil.copy2(img_file, dst)
                copied += 1
            except Exception:
                pass
    if copied:
        print(f"  📥 원본 복사 완료: {copied}개")

    # PPE 합성 대상 선정: 파일 크기(바이트) 기준 상위 N개
    ranked = sorted(image_files, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    selected_for_ppe = ranked[:MAX_PPE_PER_PERSON]

    # PPE 종류별 프롬프트 파일 로드 (PPE_STRATEGY에 정의된 타입만)
    prompt_dir = final_dir / "nanobanana" / "prompts"
    prompts_by_type = {}
    print(f"  📝 PPE 프롬프트 파일 로드 중...")
    for ppe_type in PPE_STRATEGY.keys():  # PPE_STRATEGY에 정의된 타입만 로드
        prompt_file = prompt_dir / f"{ppe_type}.txt"
        if prompt_file.exists():
            prompts_by_type[ppe_type] = prompt_file.read_text(encoding='utf-8').strip()
            print(f"    ✅ {ppe_type}.txt 로드 완료 ({len(prompts_by_type[ppe_type])}자)")
        else:
            # 폴백: 기본 프롬프트 사용
            prompts_by_type[ppe_type] = prompt
            print(f"    ⚠️ {ppe_type}.txt 없음, 기본 프롬프트 사용")
    
    # 전략적 PPE 합성 작업 목록 준비 (얼굴 인식 최적화)
    tasks = []
    task_idx = 0
    
    print(f"  🎯 PPE 합성 전략:")
    for ppe_type, count in PPE_STRATEGY.items():
        if ppe_type in prompts_by_type:
            print(f"    - {ppe_type}: {count}장 합성 예정")
    
    # PPE 전략에 따라 작업 분배
    for ppe_type, count in PPE_STRATEGY.items():
        if ppe_type not in prompts_by_type:
            continue
        
        ppe_prompt = prompts_by_type[ppe_type]
        
        for i in range(count):
            if task_idx >= len(selected_for_ppe):
                # 이미지가 부족하면 순환 사용
                img_idx = task_idx % len(selected_for_ppe)
            else:
                img_idx = task_idx
            
            img_file = selected_for_ppe[img_idx]
            original_name = img_file.name
            output_original = data_person_dir / original_name
            # PPE 타입을 파일명에 포함
            output_ppe = data_person_dir / f"ppe_{ppe_type}_{img_idx}_{original_name}"
            tasks.append((img_file, output_original, output_ppe, ppe_prompt, client, model_name, task_idx + 1, MAX_PPE_PER_PERSON))
            task_idx += 1
    
    # 병렬 처리 또는 순차 처리
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    if ENABLE_PARALLEL and len(selected_for_ppe) > 1:
        # 병렬 처리
        print(f"  🔄 병렬 처리 모드 (최대 {MAX_WORKERS}개 동시 처리, PPE 대상 {len(selected_for_ppe)}개)")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_single_image, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    processed_count += 1
                elif result["skipped"]:
                    skipped_count += 1
                elif result["failed"]:
                    failed_count += 1
    else:
        # 순차 처리
        for task in tasks:
            result = process_single_image(task)
            if result["success"]:
                processed_count += 1
            elif result["skipped"]:
                skipped_count += 1
            elif result["failed"]:
                failed_count += 1
    
    print(f"  ✅ 완료: {processed_count}개 합성, {skipped_count}개 건너뜀, {failed_count}개 실패")
    print(f"  📁 저장 위치: {data_person_dir}")


def main():
    """메인 함수"""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # Windows 출력 인코딩 설정
    
    print("=" * 70)
    print("🛠️ PPE 합성 및 임베딩 통합 작업 시작")
    print("=" * 70)
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"스크립트 파일 경로: {Path(__file__).resolve()}")
    print()
    
    # 1. 환경 변수 확인
    print("1단계: 환경 변수 확인 중...")
    # .env 파일이 있으면 로드
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)  # override=True로 기존 환경 변수 덮어쓰기
        print(f"✅ .env 파일 발견: {ENV_FILE}")
        # .env 파일 내용 확인 (디버깅용)
        try:
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                env_content = f.read().strip()
                if env_content:
                    # API 키 부분만 마스킹하여 표시
                    if 'GEMINI_API_KEY=' in env_content:
                        masked_key = env_content.split('GEMINI_API_KEY=')[1].split('\n')[0]
                        if len(masked_key) > 10:
                            masked_key = masked_key[:10] + '...'
                        print(f"   API 키 확인됨: {masked_key}")
                    else:
                        print(f"   경고: .env 파일에 GEMINI_API_KEY가 없습니다.")
        except Exception as e:
            print(f"   경고: .env 파일 읽기 오류: {e}")
    else:
        load_dotenv()  # 현재 디렉토리나 상위 디렉토리에서 찾기
        if not ENV_FILE.exists():
            print(f"ℹ️ .env 파일이 없습니다: {ENV_FILE}")
            print(f"   .env.example 파일을 참고하여 .env 파일을 생성하세요.")
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    # API 키가 없으면 오류 메시지 표시
    if not api_key:
        print("❌ 오류: GEMINI_API_KEY 환경 변수를 찾을 수 없습니다.")
        print("   해결 방법:")
        print("   1. face/.env 파일을 생성하고 다음을 추가하세요:")
        print("      GEMINI_API_KEY=your_api_key_here")
        print("   2. 또는 환경 변수로 설정하세요:")
        print("      set GEMINI_API_KEY=your_api_key_here")
        print("   3. API 키 발급: https://aistudio.google.com/app/apikey")
        return
    else:
        print("✅ 환경 변수에서 API 키를 찾았습니다.")
    
    # 2. nanobanana 사용 가능 여부 확인
    print("\n2단계: nanobanana 사용 가능 여부 확인 중...")
    if not NANOBANANA_AVAILABLE:
        print("❌ 오류: nanobanana를 사용할 수 없습니다.")
        print("   다음 명령어로 설치해주세요: pip install google-genai")
        return
    print("✅ nanobanana 사용 가능")
    
    # 3. 프롬프트 파일 확인
    print("\n3단계: 프롬프트 파일 확인 중...")
    print(f"프롬프트 파일 경로: {PROMPT_FILE}")
    print(f"프롬프트 파일 존재: {PROMPT_FILE.exists()}")
    if not PROMPT_FILE.exists():
        print(f"❌ 오류: 프롬프트 파일을 찾을 수 없습니다: {PROMPT_FILE}")
        return
    
    try:
        prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
        print(f"✅ 프롬프트 파일 로드: {PROMPT_FILE.name} ({len(prompt)}자)")
    except Exception as e:
        print(f"❌ 오류: 프롬프트 파일 읽기 실패: {e}")
        return
    
    # 4. Gemini API 클라이언트 초기화
    print("\n4단계: Gemini API 클라이언트 초기화 중...")
    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini API 클라이언트 초기화 완료")
    except Exception as e:
        print(f"❌ 오류: Gemini API 클라이언트 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 이미지 폴더 확인
    print("\n5단계: 이미지 폴더 확인 중...")
    print(f"이미지 폴더 경로: {IMAGE_DIR}")
    print(f"이미지 폴더 존재: {IMAGE_DIR.exists()}")
    if not IMAGE_DIR.exists():
        print(f"❌ 오류: 이미지 폴더를 찾을 수 없습니다: {IMAGE_DIR}")
        return
    
    # 6. 사람 폴더 목록 가져오기
    print("\n6단계: 사람 폴더 목록 가져오는 중...")
    try:
        person_folders = [d for d in IMAGE_DIR.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')]
        print(f"발견된 폴더: {[f.name for f in person_folders]}")
    except Exception as e:
        print(f"❌ 오류: 폴더 목록 가져오기 실패: {e}")
        import traceback
        traceback.print_exc()
        input("\n계속하려면 Enter를 누르세요...")
        return
    
    if not person_folders:
        print(f"⚠️ '{IMAGE_DIR}' 폴더에 사람 폴더가 없습니다.")
        print(f"   예: {IMAGE_DIR}/홍길동/ 폴더를 만들고 사진을 넣어주세요.")
        input("\n계속하려면 Enter를 누르세요...")
        return
    
    print(f"\n📋 총 {len(person_folders)}명의 사람 폴더 발견")
    
    # 7. 각 사람 폴더 처리
    print("\n7단계: 각 사람 폴더 처리 시작...")
    total_start_time = time.time()
    for person_dir in person_folders:
        data_person_dir = DATA_IMAGES_DIR / person_dir.name
        process_person_folder(person_dir, data_person_dir, prompt, client, MODEL)
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️ PPE 합성 작업 완료 (총 소요 시간: {total_time:.1f}초)")
    
    # 8. 임베딩 데이터베이스 구축
    print("\n" + "=" * 70)
    print("📊 FAISS 임베딩 데이터베이스 구축 시작")
    print("=" * 70)
    
    # build_database.py 실행
    build_database_script = script_dir / "build_database.py"
    print(f"빌드 스크립트 경로: {build_database_script}")
    print(f"빌드 스크립트 존재: {build_database_script.exists()}")
    
    if not build_database_script.exists():
        print(f"❌ 오류: build_database.py를 찾을 수 없습니다: {build_database_script}")
        input("\n계속하려면 Enter를 누르세요...")
        return
    
    import subprocess
    
    # 환경 변수 전달
    env = os.environ.copy()
    pythonpath_list = [
        str(final_dir / "src" / "backend"),
        str(final_dir / "src"),
        str(final_dir),
    ]
    if 'PYTHONPATH' in env:
        pythonpath_list.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(pythonpath_list)
    
    print(f"🔄 build_database.py 실행 중...")
    try:
        result = subprocess.run(
            [sys.executable, str(build_database_script)],
            cwd=str(script_dir),
            env=env,
            check=False
        )
    except Exception as e:
        print(f"❌ 오류: build_database.py 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        result = type('obj', (object,), {'returncode': 1})()
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("🎉 모든 작업이 완료되었습니다!")
        print("=" * 70)
        print(f"✅ PPE 합성된 이미지: {DATA_IMAGES_DIR}")
        print(f"✅ 임베딩 데이터베이스: {final_dir / 'data' / 'embeddings'}")
    else:
        print(f"\n⚠️ build_database.py 실행 중 오류 발생 (코드: {result.returncode})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

