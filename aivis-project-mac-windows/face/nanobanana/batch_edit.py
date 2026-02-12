import os, time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps
from tqdm import tqdm
from google import genai

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

MODEL = "gemini-2.5-flash-image"
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("edited_images")
PROMPT_FILE = Path("prompts/ppe_ko.txt")
RETRIES = 2
SLEEP = 2.0

def load_image(p: Path) -> Image.Image:
    im = Image.open(p)
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key, "GEMINI_API_KEY가 .env에 없습니다."

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if HEIC_SUPPORT:
        extensions.add(".heic")

    files = sorted([p for p in INPUT_DIR.glob("*.*")
                    if p.suffix.lower() in extensions])
    
    has_heic_files = any(p.suffix.lower() == '.heic' for p in INPUT_DIR.glob("*.*"))

    if not files:
        print("⚠️ input_images 폴더에 이미지가 없습니다.")
        if has_heic_files and not HEIC_SUPPORT:
            print("HEIC 파일을 처리하려면 'pip install pillow-heif'를 설치하세요.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    client = genai.Client(api_key=api_key)

    for p in tqdm(files, desc="PPE 합성"):
        out_p = OUTPUT_DIR / f"{p.stem}_ppe.jpg"
        if out_p.exists():  # 이미 처리됨
            continue

        img = load_image(p)

        ok, last_err = False, None
        for _ in range(RETRIES + 1):
            try:
                # ✅ Part 안 쓰고: [프롬프트 문자열, PIL.Image]
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=[prompt, img],
                )
                out_img = None
                for cand in resp.candidates:
                    for part in cand.content.parts:
                        if getattr(part, "inline_data", None):
                            out_img = Image.open(BytesIO(part.inline_data.data))
                            break
                    if out_img: break

                if not out_img:
                    raise RuntimeError("이미지 파트 없음")

                if out_img.mode != "RGB": out_img = out_img.convert("RGB")
                out_img.save(out_p, quality=95)
                ok = True
                break
            except Exception as e:
                last_err = e
                time.sleep(SLEEP)

        if not ok:
            print(f"❌ 실패: {p.name} → {last_err}")

    print("✅ 완료. 결과는 edited_images/에서 확인하세요.")
    if has_heic_files and not HEIC_SUPPORT:
        print("HEIC 파일은 건너뛰었습니다. 처리하려면 'pip install pillow-heif'를 설치하세요.")


if __name__ == "__main__":
    main()