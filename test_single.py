"""Test the single-image OCR API."""
from fastapi.testclient import TestClient
from main import app
import base64
import json
import time

# Use as context manager to trigger lifespan (startup/shutdown)
client = TestClient(app, raise_server_exceptions=False)
client.__enter__()


def test_health():
    print("=== TEST 1: Health check ===")
    r = client.get("/health")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    print("PASS\n")


def test_root_serves_ui():
    print("=== TEST 2: Root serves UI ===")
    r = client.get("/")
    print(f"Status: {r.status_code}")
    ct = r.headers.get("content-type", "?")
    print(f"Content-Type: {ct}")
    assert r.status_code == 200
    assert "html" in ct.lower() or "<!DOCTYPE" in r.text[:50]
    print("PASS\n")


def test_no_file():
    print("=== TEST 3: No file uploaded ===")
    r = client.post("/ocr")
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
    assert r.status_code == 422  # FastAPI validation error
    print("PASS\n")


def test_wrong_type():
    print("=== TEST 4: Wrong content type ===")
    r = client.post("/ocr", files={"file": ("test.txt", b"hello", "text/plain")})
    print(f"Status: {r.status_code}")
    print(f"Detail: {r.json().get('detail', '?')}")
    assert r.status_code == 415
    print("PASS\n")


def test_empty_file():
    print("=== TEST 5: Empty file ===")
    r = client.post("/ocr", files={"file": ("empty.jpg", b"", "image/jpeg")})
    print(f"Status: {r.status_code}")
    print(f"Detail: {r.json().get('detail', '?')}")
    assert r.status_code == 400
    print("PASS\n")


def test_real_image():
    print("=== TEST 6: Real image (1x1 PNG) → Ollama ===")
    # Minimal valid 1x1 red PNG
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    png_bytes = base64.b64decode(png_b64)
    t0 = time.perf_counter()
    r = client.post("/ocr", files={"file": ("pixel.png", png_bytes, "image/png")})
    elapsed = time.perf_counter() - t0
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"OCR status: {data.get('status')}")
    print(f"Filename: {data.get('filename')}")
    if data.get("status") == "error":
        print(f"Error: {data.get('error')}")
    else:
        print(f"Text: {data.get('extracted_text', '')[:100]}")
    print(f"Server time: {data.get('time_seconds')}s")
    print(f"Round-trip:  {elapsed:.2f}s")
    assert r.status_code == 200
    assert data["filename"] == "pixel.png"
    assert data["status"] in ("success", "error")
    print("PASS\n")


if __name__ == "__main__":
    test_health()
    test_root_serves_ui()
    test_no_file()
    test_wrong_type()
    test_empty_file()
    test_real_image()
    print("=== ALL 6 TESTS PASSED ===")
