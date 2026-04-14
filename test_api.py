"""
Quick test — sends sample images to the OCR API.
Usage:
    python test_api.py image1.jpg image2.png ...
    python test_api.py          ← uses any images found in current folder
"""
import sys
import glob
import json
import time
import httpx

API_URL = "http://localhost:8000/ocr"

def run_test(image_paths: list[str]):
    if not image_paths:
        print("❌ No images provided or found.")
        return

    print(f"📤 Sending {len(image_paths)} image(s) to {API_URL} ...\n")

    files = []
    for path in image_paths:
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp",
            "bmp": "image/bmp", "gif": "image/gif",
        }.get(ext, "image/jpeg")
        files.append(("files", (path, open(path, "rb"), mime)))

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(API_URL, files=files)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return
    finally:
        for _, (_, f, _) in files:
            f.close()

    elapsed = round(time.perf_counter() - t0, 3)
    data = resp.json()

    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n⏱  Total round-trip time: {elapsed}s")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        # Auto-find images in current dir
        paths = (
            glob.glob("*.jpg") + glob.glob("*.jpeg") +
            glob.glob("*.png") + glob.glob("*.webp")
        )
    run_test(paths)
