"""
얼굴 임베딩 데이터베이스 전체 재구축 스크립트
1. face/data 폴더 내용 삭제
2. face/image에서 face/data/images로 복사
3. (선택) PPE 합성
4. 증강 및 임베딩 생성
"""
import os
import sys
import shutil
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 경로 설정
_current_file = Path(__file__).resolve()
script_dir = _current_file.parent  # scripts/
face_dir = script_dir.parent  # face/
project_root = face_dir.parent  # aivis-project/

# 경로 정의
IMAGE_DIR = face_dir / "image"  # 원본 이미지
DATA_DIR = face_dir / "data"  # 데이터 폴더
DATA_IMAGES_DIR = DATA_DIR / "images"  # 복사된 이미지
DATA_AUGMENTED_DIR = DATA_DIR / "augmented"  # 증강 이미지
DATA_EMBEDDINGS_DIR = DATA_DIR / "embeddings"  # 임베딩
FAISS_INDEX_FILE = DATA_DIR / "face_index.faiss"
FAISS_LABELS_FILE = DATA_DIR / "face_index.faiss.labels.npy"

def clear_data_folder():
    """face/data 폴더의 생성된 파일들 삭제"""
    print("\n" + "="*70)
    print("1단계: face/data 폴더 정리 중...")
    print("="*70)
    
    # 삭제할 항목들
    items_to_remove = [
        DATA_AUGMENTED_DIR,
        DATA_EMBEDDINGS_DIR,
        DATA_IMAGES_DIR,
        FAISS_INDEX_FILE,
        FAISS_LABELS_FILE,
    ]
    
    for item in items_to_remove:
        if item.exists():
            if item.is_dir():
                print(f"  삭제 중: {item} (폴더)")
                try:
                    # Windows에서 파일이 사용 중일 수 있으므로 강제 삭제
                    shutil.rmtree(item, ignore_errors=True)
                    # 삭제 확인
                    if item.exists():
                        # 재시도 (파일 잠금 해제 대기)
                        import time
                        time.sleep(0.5)
                        shutil.rmtree(item, ignore_errors=True)
                except Exception as e:
                    print(f"    경고: {item} 삭제 중 오류 (무시하고 계속): {e}")
            else:
                print(f"  삭제 중: {item} (파일)")
                try:
                    item.unlink()
                except Exception as e:
                    print(f"    경고: {item} 삭제 중 오류 (무시하고 계속): {e}")
    
    # __init__.py는 유지
    print("[OK] face/data 폴더 정리 완료")
    print(f"   (__init__.py는 유지됨)")

def copy_images():
    """face/image에서 face/data/images로 원본 이미지 복사"""
    print("\n" + "="*70)
    print("2단계: 원본 이미지 복사 중...")
    print("="*70)
    
    if not IMAGE_DIR.exists():
        print(f"[ERROR] 원본 이미지 폴더를 찾을 수 없습니다: {IMAGE_DIR}")
        return False
    
    # data/images 폴더 생성
    DATA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 각 사람 폴더 복사
    person_folders = [d for d in IMAGE_DIR.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
    
    if not person_folders:
        print(f"[WARN] '{IMAGE_DIR}' 폴더에 사람 폴더가 없습니다.")
        return False
    
    print(f"발견된 사람 폴더: {len(person_folders)}개")
    
    total_images = 0
    for person_dir in person_folders:
        dest_dir = DATA_IMAGES_DIR / person_dir.name
        dest_dir.mkdir(exist_ok=True)
        
        # 이미지 파일 복사
        image_files = [f for f in person_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']]
        
        for img_file in image_files:
            dest_file = dest_dir / img_file.name
            shutil.copy2(img_file, dest_file)
            total_images += 1
        
        print(f"  [OK] {person_dir.name}: {len(image_files)}개 이미지 복사 완료")
    
    print(f"\n[OK] 총 {total_images}개 이미지 복사 완료")
    return True

def run_ppe_synthesis(auto_mode=False, force_run=False):
    """PPE 합성 실행 (선택적)"""
    print("\n" + "="*70)
    print("3단계: PPE 합성 (나노바나나)...")
    print("="*70)
    
    if force_run:
        print("[INFO] 강제 실행 모드: PPE 합성 실행")
        auto_mode = False
    
    if auto_mode and not force_run:
        print("[SKIP] 자동 모드: PPE 합성 건너뜀 (필요시 수동 실행: python ppe_synthesis_and_embedding.py)")
        return True
    
    try:
        if not force_run:
            try:
                response = input("PPE 합성을 실행하시겠습니까? (y/n, 기본값: y): ").strip().lower()
                if response == 'n':
                    print("[SKIP] PPE 합성 건너뜀")
                    return True
            except EOFError:
                print("[INFO] 비대화형 모드: PPE 합성 실행")
    except:
        pass
    
    try:
        # ppe_synthesis_and_embedding.py 실행
        ppe_script = script_dir / "ppe_synthesis_and_embedding.py"
        if not ppe_script.exists():
            print(f"[ERROR] PPE 합성 스크립트를 찾을 수 없습니다: {ppe_script}")
            return False
        
        print(f"[INFO] PPE 합성 스크립트 실행: {ppe_script}")
        print("[INFO] 나노바나나 API를 사용하여 PPE 합성 중...")
        
        # 작업 디렉토리를 scripts로 변경
        os.chdir(str(script_dir))
        
        # 스크립트 실행
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ppe_script)],
            cwd=str(script_dir),
            capture_output=False
        )
        
        if result.returncode == 0:
            print("\n[OK] PPE 합성 완료!")
            return True
        else:
            print(f"\n[ERROR] PPE 합성 실패 (종료 코드: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"[ERROR] 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_build_database():
    """임베딩 데이터베이스 구축"""
    print("\n" + "="*70)
    print("4단계: 증강 및 임베딩 생성 중...")
    print("="*70)
    
    try:
        # build_database.py 실행
        build_script = script_dir / "build_database.py"
        if not build_script.exists():
            print(f"[ERROR] 임베딩 생성 스크립트를 찾을 수 없습니다: {build_script}")
            return False
        
        print(f"[INFO] 임베딩 생성 스크립트 실행: {build_script}")
        
        # 작업 디렉토리를 scripts로 변경
        os.chdir(str(script_dir))
        
        # OpenMP 환경 변수 설정
        env = os.environ.copy()
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # 스크립트 실행
        import subprocess
        result = subprocess.run(
            [sys.executable, str(build_script)],
            cwd=str(script_dir),
            env=env,
            capture_output=False
        )
        
        if result.returncode == 0:
            print("\n[OK] 임베딩 데이터베이스 구축 완료!")
            return True
        else:
            print(f"\n[ERROR] 오류: 임베딩 생성 실패 (종료 코드: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"[ERROR] 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("얼굴 임베딩 데이터베이스 전체 재구축")
    print("="*70)
    print(f"프로젝트 루트: {project_root}")
    print(f"원본 이미지: {IMAGE_DIR}")
    print(f"데이터 폴더: {DATA_DIR}")
    
    # 자동 실행 모드 (명령줄 인자로 --auto 또는 --yes 전달 시)
    auto_mode = '--auto' in sys.argv or '--yes' in sys.argv or '-y' in sys.argv
    # PPE 합성 강제 실행 모드
    force_ppe = '--ppe' in sys.argv or '--with-ppe' in sys.argv
    
    if not auto_mode:
        # 확인
        try:
            response = input("\nface/data 폴더의 모든 내용을 삭제하고 재구축하시겠습니까? (y/n): ").strip().lower()
            if response != 'y':
                print("취소되었습니다.")
                return
        except EOFError:
            # 비대화형 모드에서는 자동으로 진행
            print("\n비대화형 모드: 자동으로 진행합니다...")
            auto_mode = True
    
    # 1. 데이터 폴더 정리
    clear_data_folder()
    
    # 2. 이미지 복사
    if not copy_images():
        print("\n[ERROR] 이미지 복사 실패. 프로세스를 중단합니다.")
        return
    
    # 3. PPE 합성 (나노바나나)
    if not run_ppe_synthesis(auto_mode=auto_mode, force_run=force_ppe):
        print("\n[WARN] PPE 합성 실패했지만 계속 진행합니다...")
        if not auto_mode:
            try:
                response = input("계속 진행하시겠습니까? (y/n, 기본값: y): ").strip().lower()
                if response == 'n':
                    print("프로세스를 중단합니다.")
                    return
            except EOFError:
                print("[INFO] 비대화형 모드: 자동으로 계속 진행합니다...")
        else:
            print("[INFO] 자동 모드: 계속 진행합니다...")
    
    # 4. 임베딩 생성
    if not run_build_database():
        print("\n[ERROR] 임베딩 생성 실패.")
        return
    
    # 완료
    print("\n" + "="*70)
    print("[OK] 전체 프로세스 완료!")
    print("="*70)
    print(f"생성된 파일:")
    if FAISS_INDEX_FILE.exists():
        print(f"  [OK] {FAISS_INDEX_FILE}")
    if FAISS_LABELS_FILE.exists():
        print(f"  [OK] {FAISS_LABELS_FILE}")
    if DATA_EMBEDDINGS_DIR.exists():
        print(f"  [OK] {DATA_EMBEDDINGS_DIR}")
    if DATA_AUGMENTED_DIR.exists():
        print(f"  [OK] {DATA_AUGMENTED_DIR} (증강 이미지)")

if __name__ == "__main__":
    main()

