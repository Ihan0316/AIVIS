"""
AdaFace 모델 다운로드 스크립트
GitHub 저장소에서 ONNX 모델을 다운로드합니다.
"""
import os
import sys
import urllib.request
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 프로젝트 루트 경로
_current_file = Path(__file__).resolve()
script_dir = _current_file.parent  # scripts/
face_dir = script_dir.parent  # face/
project_root = face_dir.parent  # aivis-project/

# 모델 저장 경로
model_dir = project_root / "model"
model_dir.mkdir(exist_ok=True)

# 다운로드할 모델 정보
# 참고: GitHub Releases에서 직접 다운로드가 안 될 수 있으므로,
# HuggingFace나 수동 다운로드를 권장합니다.
MODELS = {
    "r50_ms1mv2": {
        "name": "AdaFace R50 MS1MV2",
        "url": None,  # GitHub Releases URL이 작동하지 않음
        "huggingface_repo": "mk-minchul/AdaFace",  # HuggingFace 저장소 (확인 필요)
        "filename": "adaface_ir50_ms1mv2.onnx",
        "description": "ResNet50 + MS1MV2 (추천: 속도와 정확도 균형)",
        "recommended": True,
        "manual_download": "https://github.com/mk-minchul/AdaFace (저장소 클론 후 모델 파일 찾기)"
    },
    "r100_ms1mv2": {
        "name": "AdaFace R100 MS1MV2",
        "url": None,
        "huggingface_repo": "mk-minchul/AdaFace",
        "filename": "adaface_ir100_ms1mv2.onnx",
        "description": "ResNet100 + MS1MV2 (최고 정확도, 느림)",
        "recommended": False,
        "manual_download": "https://github.com/mk-minchul/AdaFace"
    },
    "r50_webface4m": {
        "name": "AdaFace R50 WebFace4M",
        "url": None,
        "huggingface_repo": "mk-minchul/AdaFace",
        "filename": "adaface_ir50_webface4m.onnx",
        "description": "ResNet50 + WebFace4M (대규모 데이터셋)",
        "recommended": False,
        "manual_download": "https://github.com/mk-minchul/AdaFace"
    }
}

def download_from_huggingface(repo_id: str, filename: str, filepath: Path) -> bool:
    """HuggingFace에서 모델을 다운로드합니다."""
    try:
        try:
            from huggingface_hub import hf_hub_download
            print(f"  HuggingFace에서 다운로드 중...")
            print(f"  저장소: {repo_id}")
            print(f"  파일: {filename}")
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(filepath.parent),
                local_dir_use_symlinks=False
            )
            
            # 다운로드된 파일을 목적지로 이동
            if downloaded_path != str(filepath):
                import shutil
                shutil.move(downloaded_path, filepath)
            
            print(f"  ✅ 다운로드 완료!")
            return True
        except ImportError:
            print(f"  ⚠️ huggingface_hub가 설치되지 않았습니다.")
            print(f"  설치 방법: pip install huggingface_hub")
            return False
    except Exception as e:
        print(f"\n  ❌ HuggingFace 다운로드 실패: {e}")
        return False

def download_file(url: str, filepath: Path, description: str) -> bool:
    """파일을 다운로드합니다."""
    if url is None:
        return False
    
    try:
        print(f"  다운로드 중: {description}")
        print(f"  URL: {url}")
        print(f"  저장 경로: {filepath}")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            size_mb = total_size / (1024 * 1024)
            downloaded_mb = downloaded / (1024 * 1024)
            print(f"\r  진행률: {percent:.1f}% ({downloaded_mb:.1f}MB / {size_mb:.1f}MB)", end='', flush=True)
        
        urllib.request.urlretrieve(url, filepath, show_progress)
        print("\n  ✅ 다운로드 완료!")
        return True
    except Exception as e:
        print(f"\n  ❌ 다운로드 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("=" * 70)
    print("📥 AdaFace 모델 다운로드")
    print("=" * 70)
    print()
    
    print("사용 가능한 모델:")
    print()
    for key, model in MODELS.items():
        recommended = "⭐ 추천" if model["recommended"] else ""
        print(f"  {key}: {model['name']} {recommended}")
        print(f"    {model['description']}")
        print()
    
    print("=" * 70)
    print("💡 추천: r50_ms1mv2 (속도와 정확도 균형, CCTV 환경 최적화)")
    print("=" * 70)
    print()
    
    # 명령줄 인자 확인
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        # 기본값: 추천 모델
        try:
            choice = input("다운로드할 모델을 선택하세요 (r50_ms1mv2/r100_ms1mv2/r50_webface4m) [기본: r50_ms1mv2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            # 비대화형 환경에서는 기본값 사용
            choice = "r50_ms1mv2"
            print("비대화형 환경 감지, 기본 모델 다운로드: r50_ms1mv2")
    
    if not choice:
        choice = "r50_ms1mv2"
    
    if choice not in MODELS:
        print(f"❌ 잘못된 선택: {choice}")
        print("사용 가능한 옵션: r50_ms1mv2, r100_ms1mv2, r50_webface4m")
        return
    
    model_info = MODELS[choice]
    model_path = model_dir / model_info["filename"]
    
    # 이미 존재하는지 확인
    if model_path.exists():
        print(f"⚠️ 모델 파일이 이미 존재합니다: {model_path}")
        overwrite = input("덮어쓰시겠습니까? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("다운로드를 취소했습니다.")
            return
        model_path.unlink()
    
    print()
    print(f"📥 {model_info['name']} 다운로드 시작...")
    print()
    
    # HuggingFace에서 다운로드 시도
    success = False
    if "huggingface_repo" in model_info and model_info["huggingface_repo"]:
        success = download_from_huggingface(
            model_info["huggingface_repo"],
            model_info["filename"],
            model_path
        )
    
    # HuggingFace 실패 시 URL 다운로드 시도
    if not success and model_info.get("url"):
        success = download_file(
            model_info["url"],
            model_path,
            model_info["description"]
        )
    
    if success:
        print()
        print("=" * 70)
        print("✅ 다운로드 완료!")
        print("=" * 70)
        print(f"모델 경로: {model_path}")
        print()
        print("💡 다음 단계:")
        print("   1. 환경 변수 설정: set USE_ADA_FACE=true")
        print("   2. 임베딩 데이터베이스 구축: python face\\scripts\\build_database.py")
        print()
    else:
        print()
        print("=" * 70)
        print("❌ 다운로드 실패")
        print("=" * 70)
        print()
        print("💡 수동 다운로드 방법:")
        print(f"   1. GitHub 저장소 클론:")
        print(f"      git clone https://github.com/mk-minchul/AdaFace.git")
        print(f"      cd AdaFace")
        print(f"      (모델 파일 찾기)")
        print(f"   2. 또는 HuggingFace에서 다운로드:")
        print(f"      pip install huggingface_hub")
        print(f"      python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='mk-minchul/AdaFace', filename='{model_info['filename']}', local_dir='model')\"")
        print(f"   3. 다운로드한 파일을 다음 경로에 저장:")
        print(f"      {model_path}")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n\n❌ 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)



