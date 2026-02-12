#!/usr/bin/env python3
"""
Ultralytics YOLO → TensorRT 변환 스크립트

사용 예시:
    python scripts/export_tensorrt.py \
        --weights model/Yolo11n_PPE1.pt \
        --imgsz 512 \
        --device cuda \
        --half \
        --output model/Yolo11n_PPE1.engine

※ INT8 변환 시에는 --int8 와 함께 --calib-dir 로 캘리브레이션 이미지 디렉터리를 지정하세요.
"""

import argparse
import os
from typing import Optional

from ultralytics import YOLO


def export_tensorrt(weights: str,
                    imgsz: int,
                    device: str,
                    half: bool,
                    int8: bool,
                    dynamic: bool,
                    workspace: Optional[int],
                    simplify: bool,
                    output: Optional[str],
                    calib_dir: Optional[str]) -> None:
    model = YOLO(weights)

    export_kwargs = {
        "format": "engine",
        "device": device,
        "imgsz": imgsz,
        "half": half,
        "dynamic": dynamic,
        "simplify": simplify,
    }

    if workspace is not None:
        export_kwargs["workspace"] = workspace

    if output:
        export_kwargs["save_dir"] = os.path.dirname(output) or "."
        export_kwargs["name"] = os.path.splitext(os.path.basename(output))[0]

    if int8:
        if not calib_dir:
            raise ValueError("INT8 변환 시 --calib-dir 옵션으로 캘리브레이션 이미지를 지정해야 합니다.")
        export_kwargs["int8"] = True
        export_kwargs["calib"] = calib_dir

    engine_path = model.export(**export_kwargs)
    print(f"[TensorRT] 엔진 생성 완료: {engine_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 모델을 TensorRT 엔진으로 변환합니다.")
    parser.add_argument("--weights", required=True, help="학습된 YOLO 가중치(.pt) 경로")
    parser.add_argument("--imgsz", type=int, default=512, help="입력 해상도 (정사각형)")
    parser.add_argument("--device", default="cuda", help="변환에 사용할 디바이스 (cuda 또는 cpu)")
    parser.add_argument("--half", action="store_true", help="FP16 최적화 사용")
    parser.add_argument("--int8", action="store_true", help="INT8 최적화 사용 (캘리브레이션 필요)")
    parser.add_argument("--dynamic", action="store_true", help="동적 배치/해상도 지원")
    parser.add_argument("--workspace", type=int, default=None, help="TensorRT 워크스페이스 크기(MB)")
    parser.add_argument("--simplify", action="store_true", help="ONNX 그래프 간소화 수행")
    parser.add_argument("--output", default=None, help="출력 엔진 파일 경로 (생략 시 Ultralytics 기본 규칙 사용)")
    parser.add_argument("--calib-dir", default=None, help="INT8 캘리브레이션 이미지 디렉터리 경로")
    return parser.parse_args()


def main():
    args = parse_args()
    export_tensorrt(
        weights=args.weights,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        workspace=args.workspace,
        simplify=args.simplify,
        output=args.output,
        calib_dir=args.calib_dir,
    )


if __name__ == "__main__":
    main()

