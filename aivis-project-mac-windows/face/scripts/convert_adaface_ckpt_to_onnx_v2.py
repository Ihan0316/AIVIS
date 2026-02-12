"""
AdaFace .ckpt 파일을 .onnx로 변환하는 스크립트 (AdaFace 저장소 사용)
"""
import os
import sys
import torch
import torch.onnx
import numpy as np
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

# AdaFace 저장소 경로 추가
adaface_repo = project_root / "AdaFace"
if adaface_repo.exists():
    sys.path.insert(0, str(adaface_repo))
    print(f"✅ AdaFace 저장소 경로 추가: {adaface_repo}")
else:
    print(f"❌ AdaFace 저장소를 찾을 수 없습니다: {adaface_repo}")
    sys.exit(1)

# 모델 경로
model_dir = project_root / "model"
ckpt_path = model_dir / "adaface_ir50_ms1mv2.ckpt"
onnx_path = model_dir / "adaface_ir50_ms1mv2.onnx"

def load_adaface_model(ckpt_path: Path):
    """
    AdaFace .ckpt 파일에서 모델을 로드합니다.
    """
    try:
        from net import build_model
        
        # 모델 아키텍처 결정
        if 'ir50' in ckpt_path.name.lower() or 'r50' in ckpt_path.name.lower():
            model_name = 'ir_50'
        elif 'ir100' in ckpt_path.name.lower() or 'r100' in ckpt_path.name.lower():
            model_name = 'ir_101'
        else:
            # 기본값: ir50
            model_name = 'ir_50'
            print(f"⚠️ 모델 아키텍처를 확인할 수 없어 기본값 사용: {model_name}")
        
        print(f"📦 모델 아키텍처: {model_name}")
        model = build_model(model_name=model_name)
        
        # 체크포인트 로드
        print(f"📥 체크포인트 로드 중: {ckpt_path}")
        # PyTorch Lightning 체크포인트일 수 있으므로 weights_only=False 사용
        # (신뢰할 수 있는 소스에서 다운로드한 파일이므로 안전)
        try:
            checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"⚠️ torch.load 실패: {e}")
            print("💡 pytorch_lightning 설치 시도 중...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch_lightning", "-q"])
                print("✅ pytorch_lightning 설치 완료, 재시도 중...")
                checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
            except Exception as e2:
                print(f"❌ pytorch_lightning 설치 실패: {e2}")
                print("💡 수동 설치: pip install pytorch_lightning")
                raise
        
        # state_dict 추출 (AdaFace 저장소 방식)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and any('state_dict' in str(k).lower() for k in checkpoint.keys()):
            # PyTorch Lightning 형식일 수 있음
            for key in ['state_dict', 'model_state_dict', 'model']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # AdaFace 저장소 방식: 'model.' 접두사 제거
        # validate_IJB_BC.py의 load_pretrained_model 참고
        model_statedict = {}
        for key, val in state_dict.items():
            if key.startswith('model.'):
                # 'model.' 접두사 제거 (6자)
                new_key = key[6:]
            elif key.startswith('module.'):
                # 'module.' 접두사 제거
                new_key = key[7:]
            elif key.startswith('backbone.'):
                # 'backbone.' 접두사 제거
                new_key = key[9:]
            else:
                new_key = key
            model_statedict[new_key] = val
        
        # 가중치 로드
        model.load_state_dict(model_statedict, strict=False)
        model.eval()
        
        print(f"✅ 모델 로드 완료")
        return model
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_to_onnx(model, onnx_path: Path, input_size=(1, 3, 112, 112)):
    """
    PyTorch 모델을 ONNX로 변환합니다.
    """
    try:
        print(f"🔄 ONNX 변환 중...")
        print(f"   입력 크기: {input_size}")
        
        # 더미 입력 생성
        dummy_input = torch.randn(*input_size)
        
        # ONNX로 변환
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX 변환 완료: {onnx_path}")
        
        # 파일 크기 확인
        file_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   파일 크기: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("=" * 70)
    print("🔄 AdaFace .ckpt → .onnx 변환")
    print("=" * 70)
    print()
    
    # 파일 확인
    if not ckpt_path.exists():
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        print()
        print("💡 해결 방법:")
        print(f"   1. 체크포인트 파일을 다음 경로에 저장하세요:")
        print(f"      {ckpt_path}")
        print(f"   2. 또는 다른 경로에 있다면 경로를 수정하세요.")
        return
    
    if onnx_path.exists():
        print(f"⚠️ ONNX 파일이 이미 존재합니다: {onnx_path}")
        try:
            overwrite = input("덮어쓰시겠습니까? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            overwrite = 'n'
        
        if overwrite != 'y':
            print("변환을 취소했습니다.")
            return
        onnx_path.unlink()
    
    print(f"📁 체크포인트 파일: {ckpt_path}")
    print(f"📁 출력 ONNX 파일: {onnx_path}")
    print()
    
    # 모델 로드
    model = load_adaface_model(ckpt_path)
    if model is None:
        print()
        print("❌ 모델 로드 실패")
        return
    
    # ONNX 변환
    success = convert_to_onnx(model, onnx_path)
    
    if success:
        print()
        print("=" * 70)
        print("✅ 변환 완료!")
        print("=" * 70)
        print(f"ONNX 파일: {onnx_path}")
        print()
        print("💡 다음 단계:")
        print("   1. 환경 변수 설정: set USE_ADA_FACE=true")
        print("   2. 임베딩 데이터베이스 구축: python face\\scripts\\build_database.py")
        print()
    else:
        print()
        print("=" * 70)
        print("❌ 변환 실패")
        print("=" * 70)
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



