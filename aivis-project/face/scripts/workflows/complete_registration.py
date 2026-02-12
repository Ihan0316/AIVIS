# workflows/complete_registration.py
"""
완전한 얼굴 등록 워크플로
1. 원본 이미지 이동
2. PPE 합성 (선택적)
3. FAISS DB 구축/업데이트
"""
import os
import sys
import argparse
from pathlib import Path

# 현재 파일 위치 기준 경로 수정
script_dir = Path(__file__).parent  # workflows/
final_dir = script_dir.parent.parent  # final/

# utils 경로 추가
utils_dir = final_dir / "src" / "utils"
sys.path.insert(0, str(utils_dir))
sys.path.insert(0, str(final_dir / "src"))

# scripts 경로 추가 (build_database 임포트용)
scripts_dir = final_dir / "scripts"
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(final_dir))

try:
    from ppe_synthesizer import process_with_ppe, NANOBANANA_AVAILABLE
except ImportError:
    # 절대 경로로 시도
    sys.path.insert(0, str(final_dir))
    from src.utils.ppe_synthesizer import process_with_ppe, NANOBANANA_AVAILABLE

from dotenv import load_dotenv


def complete_face_registration(
    enable_ppe=True,
    prompt_file="../nanobanana/prompts/ppe_ko.txt",  # scripts 디렉토리 기준
    model_name="gemini-2.5-flash-image"
):
    """
    완전한 얼굴 등록 프로세스를 실행합니다.
    
    Args:
        enable_ppe: PPE 합성 활성화 여부
        prompt_file: PPE 프롬프트 파일 경로
        model_name: Gemini 모델 이름
    """
    print("=" * 60)
    print("🎯 완전한 얼굴 등록 워크플로 시작")
    print("=" * 60)
    
    INPUT_DIR = str(final_dir / "data" / "new_faces")
    DB_PATH = str(final_dir / "image")
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)
    
    # 1단계: 처리할 폴더 확인
    person_folders = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    if not person_folders:
        print(f"\n'{INPUT_DIR}' 폴더에 처리할 이름 폴더가 없습니다.")
        print(f"예: '{os.path.join(INPUT_DIR, '홍길동')}' 폴더를 만들고 그 안에 사진을 넣어주세요.")
        return
    
    print(f"\n📋 총 {len(person_folders)}명의 인물 폴더를 처리합니다.")
    
    # 2단계: PPE 합성 설정
    ppe_api_key = None
    if enable_ppe and NANOBANANA_AVAILABLE:
        load_dotenv()
        ppe_api_key = os.getenv("GEMINI_API_KEY")
        if not ppe_api_key:
            print("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
            print("   PPE 합성을 건너뜁니다.")
            enable_ppe = False
    
    if enable_ppe:
        print("🛠️ PPE 합성 모드: 활성화")
    else:
        print("ℹ️ PPE 합성 모드: 비활성화")
    
    # 3단계: 각 사람별 처리
    import shutil
    for person_name in person_folders:
        source_dir = os.path.join(INPUT_DIR, person_name)
        destination_dir = os.path.join(DB_PATH, person_name)
        
        print("-" * 30)
        print(f"▶ '{person_name}' 폴더를 처리 중...")
        
        # 이미지 이동
        if os.path.exists(destination_dir):
            print(f"  '{person_name}' DB 폴더가 이미 존재합니다. 파일들을 통합합니다.")
            for filename in os.listdir(source_dir):
                shutil.move(os.path.join(source_dir, filename), os.path.join(destination_dir, filename))
            os.rmdir(source_dir)
        else:
            shutil.move(source_dir, destination_dir)
        
        # PPE 합성 수행
        if enable_ppe and ppe_api_key:
            prompt_path = Path(prompt_file)
            if prompt_path.exists():
                print(f"  🛠️ PPE 합성 시작...")
                ppe_count, failed_count = process_with_ppe(
                    destination_dir, 
                    destination_dir, 
                    prompt_path, 
                    ppe_api_key,
                    model_name
                )
                print(f"  ✅ PPE 합성 완료: {ppe_count}개 성공, {failed_count}개 실패")
            else:
                print(f"  ⚠️ 프롬프트 파일을 찾을 수 없습니다: {prompt_file}")
        
        print(f"  ✅ '{person_name}' 폴더를 '{DB_PATH}'(으)로 성공적으로 이동/통합했습니다.")
    
    # 4단계: FAISS DB 구축/업데이트
    print("\n" + "=" * 60)
    print("📊 FAISS 데이터베이스 구축/업데이트 시작")
    print("=" * 60)
    
    # build_database.py 실행 (워킹 디렉토리 변경)
    import subprocess
    
    # 환경 변수 전달 (특히 PYTHONPATH)
    env = os.environ.copy()
    env['PYTHONPATH'] = ':'.join([
        str(final_dir / "src" / "backend"),
        str(final_dir / "src"),
        str(final_dir),
        env.get('PYTHONPATH', '')
    ])
    
    result = subprocess.run(
        [sys.executable, str(scripts_dir / "build_database.py")],
        cwd=str(scripts_dir),
        capture_output=False,
        env=env
    )
    if result.returncode != 0:
        print(f"⚠️ build_database.py 실행 중 오류 발생 (코드: {result.returncode})")
    
    print("\n" + "=" * 60)
    print("🎉 완전한 얼굴 등록 워크플로 완료!")
    print("=" * 60)


def main():
    """커맨드라인 인터페이스"""
    parser = argparse.ArgumentParser(description='완전한 얼굴 등록 워크플로')
    parser.add_argument('--no-ppe', action='store_true', help='PPE 합성을 건너뜁니다')
    prompt_file_path = str(final_dir / "nanobanana" / "prompts" / "ppe_ko.txt")
    parser.add_argument('--prompt-file', default=prompt_file_path, help='PPE 프롬프트 파일 경로')
    parser.add_argument('--model', default='gemini-2.5-flash-image', help='Gemini 모델 이름')
    
    args = parser.parse_args()
    
    complete_face_registration(
        enable_ppe=not args.no_ppe,
        prompt_file=args.prompt_file,
        model_name=args.model
    )


if __name__ == "__main__":
    main()