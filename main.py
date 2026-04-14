"""
Groq Vision Invoice OCR — v9.0
================================
Single-image, maximum accuracy architecture.

Orientation pipeline (no YOLO needed — wrong tool for documents):
  Step 1 — EXIF transpose (handles phone metadata rotation)
  Step 2 — OpenCV document orientation:
            • Grayscale + adaptive threshold
            • Find largest contour (document boundary)
            • Compute minimum area rect angle → rotate to upright
  Step 3 — Text-line skew correction via Hough line voting
  Step 4 — Aspect ratio sanity: if still landscape after above, rotate 90°
  Step 5 — Contrast + sharpness boost for OCR

Extraction pipeline (4 passes, all concurrent):
  Pass 0 — Zone scan   : full image → detect header/names/numbers/totals zones
  Pass 1 — Header      : cropped header zone → all supplier/buyer fields
  Pass 2 — Names strip : left side of items table → sno + product name + HSN
  Pass 3 — Numbers strip: right side of items table → all numeric columns
  Pass 4 — Totals      : footer zone → subtotal/tax/grand total

Each pass runs crop variants internally and votes for maximum accuracy.
"""
import base64
import asyncio
import io
import json
import logging
import os
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL            = "meta-llama/llama-4-scout-17b-16e-instruct"
TIMEOUT          = 120.0
MAX_RETRIES      = 3
MAX_FILE_SIZE    = 20 * 1024 * 1024
ZONE_MARGIN      = 0.02
GROQ_CONCURRENCY = 8

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("groq-ocr")

http_client: Optional[httpx.AsyncClient] = None
groq_semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app):
    global http_client, groq_semaphore
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY not set — add it to .env or export it")
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    groq_semaphore = asyncio.Semaphore(GROQ_CONCURRENCY)
    log.info("Groq OCR v9.0 ready — model: %s", MODEL)
    yield
    await http_client.aclose()


app = FastAPI(
    title="Groq Vision OCR API",
    description="Single-image invoice OCR with smart orientation correction",
    version="9.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Groq OCR v9.0 — POST /ocr with an invoice image."}


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# User always uploads the image the correct way up.
# We just handle EXIF metadata rotation (phone sometimes stores rotation
# in metadata instead of rotating pixels), then enhance for OCR.
# ══════════════════════════════════════════════════════════════════════════════

def smart_orient_and_preprocess(image_bytes: bytes) -> tuple:
    """
    Preprocessing pipeline:
    Step 1 — EXIF transpose: fixes silent metadata rotation from phones
    Step 2 — Contrast + sharpness + brightness boost for OCR accuracy
    Step 3 — Upscale if too small (LLM reads text better at higher res)

    Returns (processed_png_bytes, info_dict)
    """
    info = {}

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    info["original_size"] = f"{img.width}x{img.height}"
    log.info("Image loaded: %dx%d", img.width, img.height)

    # Enhancement for OCR
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    img = ImageEnhance.Brightness(img).enhance(1.05)

    # Upscale short side to at least 1200px so text is crisp for LLM
    min_short = 1200
    short = min(img.width, img.height)
    if short < min_short:
        scale = min_short / short
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS
        )
        info["upscaled"] = f"→ {img.width}x{img.height}"
        log.info("Upscaled to %dx%d", img.width, img.height)

    info["final_size"] = f"{img.width}x{img.height}"

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), info


# ── Crop zone ──────────────────────────────────────────────────────────────────
def crop_zone(image_bytes: bytes, top_pct: float, bottom_pct: float,
              left_pct: float = 0.0, right_pct: float = 1.0,
              min_width: int = 1200) -> tuple:
    img = Image.open(io.BytesIO(image_bytes))
    W, H = img.width, img.height
    cropped = img.crop((
        int(W * left_pct), int(H * top_pct),
        int(W * right_pct), int(H * bottom_pct),
    ))
    if cropped.width < min_width:
        scale = min_width / cropped.width
        cropped = cropped.resize(
            (int(cropped.width * scale), int(cropped.height * scale)),
            Image.LANCZOS
        )
    buf = io.BytesIO()
    cropped.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


def img_to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# GROQ API
# ══════════════════════════════════════════════════════════════════════════════

async def call_groq(prompt: str, image_b64: str, media_type: str,
                    label: str, max_tokens: int = 8192) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{image_b64}"
                }},
            ],
        }],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("[%s] attempt %d/%d", label, attempt, MAX_RETRIES)
            async with groq_semaphore:
                resp = await http_client.post(GROQ_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            log.info("[%s] OK — %d chars", label, len(content))
            return content
        except httpx.TimeoutException:
            last_error = f"Timeout attempt {attempt}"
            log.warning("[%s] timeout attempt %d", label, attempt)
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            log.error("[%s] %s", label, last_error)
            if e.response.status_code in (401, 403):
                raise RuntimeError("Auth error — check GROQ_API_KEY")
            if e.response.status_code == 429:
                await asyncio.sleep(2 * attempt)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            log.warning("[%s] error attempt %d: %s", label, attempt, last_error)
        if attempt < MAX_RETRIES:
            await asyncio.sleep(1)
    raise RuntimeError(f"[{label}] all {MAX_RETRIES} attempts failed: {last_error}")


def parse_json(text: str) -> Union[dict, list]:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in [('{', '}'), ('[', ']')]:
        s = text.find(opener)
        e = text.rfind(closer)
        if s != -1 and e != -1 and s < e:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Cannot parse JSON: {text[:300]}")


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_value(v):
    if v in (None, "", "null", "NULL", "N/A", "n/a", "NA", "none", "None"):
        return None
    return v


def clean_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        text = str(v).strip().replace(",", "")
        text = re.sub(r"[^0-9.\-]", "", text)
        if text in ("", ".", "-", "-.", ".-"):
            return None
        return float(text)
    except Exception:
        return None


def clean_int(v):
    n = clean_number(v)
    if n is None:
        return None
    try:
        return int(round(n))
    except Exception:
        return None


def strip_trailing_number(name: str) -> str:
    if not name:
        return name
    return re.sub(r'\s+\d+$', '', name.strip())


def clean_bill_no(v: str) -> str:
    v = re.sub(r"(?i)(invoice\s*no\s*[:\-]?\s*|bill\s*no\s*[:\-]?\s*|no\s*[:\-]\s*)", "", v)
    return v.strip()


def clean_gstin(v: str) -> str:
    if not v:
        return v
    match = re.search(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', v.upper())
    return match.group(0) if match else v.strip()


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalize_zone(zone: dict, fallback: tuple) -> tuple:
    try:
        t = clamp01(float(zone.get("top_pct", 0)))
        b = clamp01(float(zone.get("bottom_pct", 1)))
        l = clamp01(float(zone.get("left_pct", 0.0)))
        r = clamp01(float(zone.get("right_pct", 1.0)))
    except Exception:
        return fallback
    if b <= t or r <= l:
        return fallback
    if (b - t) < 0.05 or (r - l) < 0.10:
        return fallback
    return (t, b, l, r)


def build_crop_variants(zone: tuple, margin: float = ZONE_MARGIN) -> list:
    """3 crops per zone: tight, expanded, and exact."""
    t, b, l, r = zone
    variants = {
        (t, b, l, r),                                          # exact
        (clamp01(t - margin), clamp01(b + margin),             # expanded
         clamp01(l - margin), clamp01(r + margin)),
        (clamp01(t - margin/2), clamp01(b + margin/2),         # slightly expanded
         clamp01(l), clamp01(r)),
    }
    valid = [v for v in variants if v[1] > v[0] and v[3] > v[2]]
    return sorted(valid)


def normalize_item(item: dict) -> dict:
    out = dict(item or {})
    out["sno"] = clean_int(out.get("sno"))
    if out.get("name") is not None:
        out["name"] = strip_trailing_number(str(out.get("name"))) or None
    if out.get("hsn") is not None:
        hsn = re.sub(r"\D", "", str(out.get("hsn")))
        out["hsn"] = hsn if hsn else None
    for field in ("mrp", "qty", "rate", "discount", "cgst_pct", "sgst_pct",
                  "taxable", "cgst_amt", "sgst_amt", "amount"):
        out[field] = clean_number(out.get(field))
    return out


def normalize_totals(totals: dict) -> dict:
    out = dict(totals or {})
    for field in ("subtotal", "total_discount", "total_taxable",
                  "total_cgst", "total_sgst", "grand_total"):
        if field in out:
            out[field] = clean_number(out.get(field))
    if "total_items_count" in out:
        out["total_items_count"] = clean_int(out.get("total_items_count"))
    return out


def reconcile_totals_from_items(items: list, totals: dict) -> dict:
    """
    Fill missing tax subtotals from line-item math.
    Does NOT fill grand_total — if it's not in the image (multi-page invoice)
    it should stay null, not be calculated from partial items.
    """
    out = dict(totals or {})
    item_taxable = sum((i.get("taxable") or 0.0) for i in items if i.get("taxable") is not None)
    item_cgst    = sum((i.get("cgst_amt") or 0.0) for i in items if i.get("cgst_amt") is not None)
    item_sgst    = sum((i.get("sgst_amt") or 0.0) for i in items if i.get("sgst_amt") is not None)

    if out.get("total_taxable") is None and item_taxable > 0:
        out["total_taxable"] = round(item_taxable, 2)
    if out.get("total_cgst") is None and item_cgst > 0:
        out["total_cgst"] = round(item_cgst, 2)
    if out.get("total_sgst") is None and item_sgst > 0:
        out["total_sgst"] = round(item_sgst, 2)
    # Do NOT fill grand_total or total_items_count from items —
    # on page 1 of a multi-page invoice these are not visible
    return out


def vote_scalar(values: list):
    non_null = [v for v in values if v is not None and str(v).strip() not in ("", "null", "NULL")]
    if not non_null:
        return None
    if len(non_null) == 1:
        return non_null[0]
    try:
        counts = Counter(str(v).strip().upper() for v in non_null)
        best, count = counts.most_common(1)[0]
        if count > 1:
            for v in non_null:
                if str(v).strip().upper() == best:
                    return v
    except Exception:
        pass
    if isinstance(non_null[0], str):
        return max(non_null, key=lambda x: len(str(x)))
    try:
        s = sorted(non_null)
        return s[len(s) // 2]
    except Exception:
        return non_null[0]


def vote_dict(dicts: list, fields: list) -> dict:
    result = {}
    for field in fields:
        values = [d.get(field) for d in dicts if d]
        result[field] = vote_scalar(values)
    return result


def stitch_items(names: list, numbers: list) -> list:
    numbers = [normalize_item(x) for x in (numbers or [])]
    names   = [normalize_item(x) for x in (names or [])]
    num_by_sno = {item.get("sno"): item for item in numbers if item.get("sno") is not None}
    result = []
    seen = set()

    for name_item in names:
        sno = name_item.get("sno")
        if sno is None or sno in seen:
            continue
        seen.add(sno)
        name = strip_trailing_number(name_item.get("name") or "")
        merged = {
            "sno": sno, "name": name or None,
            "hsn": name_item.get("hsn"),
            "mrp": None, "qty": None, "rate": None, "discount": None,
            "cgst_pct": None, "sgst_pct": None, "taxable": None,
            "cgst_amt": None, "sgst_amt": None, "amount": None,
            "confidence": "high",
        }
        if sno in num_by_sno:
            for f in ("mrp", "qty", "rate", "discount", "cgst_pct", "sgst_pct",
                      "taxable", "cgst_amt", "sgst_amt", "amount"):
                merged[f] = num_by_sno[sno].get(f)
        else:
            merged["confidence"] = "low"
        result.append(merged)

    for sno, num in num_by_sno.items():
        if sno not in seen:
            num["confidence"] = "low"
            result.append(num)

    result.sort(key=lambda x: x.get("sno") or 9999)
    return result


def extract_header_fields(header: dict) -> tuple:
    fields = ["store_name", "address", "gstin", "phone", "bill_no", "date",
              "salesperson", "retailer_name", "retailer_address",
              "retailer_gstin", "fssal_no", "dl_no"]
    result = {}
    confidences = {}
    for field in fields:
        raw = header.get(field)
        if isinstance(raw, dict):
            result[field] = clean_value(raw.get("value"))
            confidences[field] = raw.get("confidence", "high")
        else:
            result[field] = clean_value(raw)
            confidences[field] = "high"
    return result, confidences


def extract_totals_fields(totals: dict) -> tuple:
    fields = ["subtotal", "total_discount", "total_taxable", "total_cgst",
              "total_sgst", "grand_total", "total_items_count", "amount_in_words"]
    result = {}
    confidences = {}
    for field in fields:
        raw = totals.get(field)
        if isinstance(raw, dict):
            result[field] = clean_value(raw.get("value"))
            confidences[field] = raw.get("confidence", "high")
        else:
            result[field] = clean_value(raw)
            confidences[field] = "high"
    return result, confidences


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

ZONES = {
    "portrait": {
        "header":  (0.00, 0.38, 0.00, 1.00),
        "names":   (0.38, 0.90, 0.00, 0.45),
        "numbers": (0.38, 0.90, 0.28, 1.00),
        "totals":  (0.85, 1.00, 0.00, 1.00),
    },
    "landscape": {
        "header":  (0.00, 0.32, 0.00, 1.00),
        "names":   (0.18, 0.90, 0.00, 0.38),
        "numbers": (0.18, 0.90, 0.22, 1.00),
        "totals":  (0.82, 1.00, 0.00, 1.00),
    },
}

ZONE_SCAN_PROMPT = """You are a layout analyzer for Indian GST invoices.

Analyze this invoice image and return the exact bounding zones as percentages (0.0 to 1.0):
- header: the top section with supplier name, address, GSTIN, buyer info, bill no, date
- names: the left portion of the line items table — Sr.No + Product Name/Description + HSN columns
- numbers: the right portion of the line items table — all numeric columns (Qty, Rate, MRP, Disc, Tax, Amount)
- totals: the bottom section with subtotal, tax totals, grand total, amount in words

Return ONLY this JSON (no markdown, no explanation):
{
  "layout": "portrait",
  "header":  {"top_pct": 0.0, "bottom_pct": 0.35, "left_pct": 0.0, "right_pct": 1.0},
  "names":   {"top_pct": 0.35, "bottom_pct": 0.88, "left_pct": 0.0, "right_pct": 0.45},
  "numbers": {"top_pct": 0.35, "bottom_pct": 0.88, "left_pct": 0.28, "right_pct": 1.0},
  "totals":  {"top_pct": 0.85, "bottom_pct": 1.0, "left_pct": 0.0, "right_pct": 1.0}
}

Rules:
- names and numbers zones should OVERLAP horizontally (both include the Sr.No area)
- totals must include ALL summary lines including grand total and amount in words
- if the image is landscape (wider than tall), adjust accordingly"""


HEADER_PROMPT = """You are a precise Indian GST invoice OCR engine.

Extract ALL header information from this invoice image crop (top portion of invoice).

Return ONLY this JSON:
{
  "store_name": {"value": "supplier company name", "confidence": "high"},
  "address": {"value": "full supplier address including city/state/pin", "confidence": "high"},
  "gstin": {"value": "supplier GSTIN exactly 15 chars e.g. 27AAAAA0000A1Z5", "confidence": "high"},
  "phone": {"value": "phone/mobile number", "confidence": "high"},
  "bill_no": {"value": "invoice/bill/SO number", "confidence": "high"},
  "date": {"value": "DD/MM/YYYY", "confidence": "high"},
  "salesperson": {"value": "salesperson or beat name or null", "confidence": "high"},
  "retailer_name": {"value": "buyer/customer company name", "confidence": "high"},
  "retailer_address": {"value": "buyer full address", "confidence": "high"},
  "retailer_gstin": {"value": "buyer GSTIN 15 chars or null", "confidence": "high"},
  "fssal_no": {"value": "FSSAI number or null", "confidence": "high"},
  "dl_no": {"value": "Drug License number or null", "confidence": "high"}
}

CRITICAL RULES:
- GSTIN is exactly 15 characters: 2 digits + 5 letters + 4 digits + 1 letter + 1 alphanumeric + Z + 1 alphanumeric
- Be extra careful: O vs 0, I vs 1, B vs 8, S vs 5
- Set confidence "low" if text is blurry, cut off, or you are uncertain
- null for fields not present in this invoice
- Return ONLY raw JSON, no markdown"""


ITEMS_NAMES_PROMPT = """You are a precise Indian GST invoice OCR engine.

This image shows the LEFT SIDE of an invoice's line items table — containing:
- Sr.No / S.No (serial number column)
- Product name / Item description (may span multiple lines per item)
- HSN/SAC code (if present in this section)

Extract EVERY row. Each item may have 2-4 lines of text (item code, product name, batch no, expiry date — they all belong to ONE row).

Return ONLY this JSON:
{
  "items": [
    {
      "sno": 1,
      "name": "complete item name including item code and description e.g. 3277930 - DETTOL ANTISEPTIC 550ML x30 (Batch:VK806, Exp:01/28)",
      "hsn": "30049099"
    }
  ]
}

CRITICAL:
- Merge ALL text lines belonging to the same row into one "name" string
- Include item codes, batch numbers, expiry dates in the name
- Do NOT include numbers from the numeric columns (qty, rate, amount)
- Return ONLY raw JSON"""


ITEMS_NUMBERS_PROMPT = """You are a precise Indian GST invoice OCR engine.

This image shows the NUMERIC COLUMNS of an invoice items table.

IMPORTANT — Read columns LEFT TO RIGHT strictly in this order:
Sr.No | HSN | U/O/M | MRP | Qty | Rate | Sch Amt | DB Disc | CGST% | CGST Amt | UT/SGST% | SGST Amt | IGST% | Taxable Amt | Total Amt

Column definitions:
- MRP = Maximum Retail Price (the printed price on the product package)
- Qty = quantity ordered (may show as "3EA", "12EA", "1CA" — extract just the number)
- Rate = the actual selling/invoice rate per unit (always less than or equal to MRP)
- Sch Amt = scheme/discount amount
- DB Disc = additional distributor discount
- CGST% = CGST percentage (e.g. 2.5, 9.0, 12.0)
- CGST Amt = CGST rupee amount
- SGST% = SGST percentage
- SGST Amt = SGST rupee amount
- Taxable Amt = taxable base amount
- Total Amt = final amount for this row

Extract EVERY row. Return ONLY this JSON:
{
  "items": [
    {
      "sno": 1,
      "mrp": 267.09,
      "qty": 3,
      "rate": 223.13,
      "discount": 0.00,
      "cgst_pct": 2.5,
      "sgst_pct": 2.5,
      "taxable": 669.39,
      "cgst_amt": 16.74,
      "sgst_amt": 16.74,
      "amount": 702.86
    }
  ]
}

CRITICAL:
- Read each row STRICTLY within its own horizontal band — NEVER mix rows
- sno is always the leftmost integer sequence number
- MRP is always >= Rate. If you see Rate > MRP you have the columns swapped — fix it
- Qty: if printed as "3EA" or "3 EA" extract 3. If "1CA 0EA" extract 25 (1 carton)
- CGST% and SGST% are percentages NOT rupee amounts (2.5, 9.0, 12.0, 18.0)
- null for genuinely blank/missing cells
- Return ONLY raw JSON"""


TOTALS_PROMPT = """You are a precise Indian GST invoice OCR engine.

This image shows the FOOTER / TOTALS section of an invoice.

Extract ONLY what is PHYSICALLY PRINTED and VISIBLE in this image:
{
  "subtotal": {"value": 12500.00, "confidence": "high"},
  "total_discount": {"value": 500.00, "confidence": "high"},
  "total_taxable": {"value": 12000.00, "confidence": "high"},
  "total_cgst": {"value": 1080.00, "confidence": "high"},
  "total_sgst": {"value": 1080.00, "confidence": "high"},
  "grand_total": {"value": 14160.00, "confidence": "high"},
  "total_items_count": {"value": 15, "confidence": "high"},
  "amount_in_words": {"value": "Fourteen Thousand One Hundred Sixty Only", "confidence": "high"}
}

CRITICAL RULES — READ CAREFULLY:
- ONLY extract values you can literally SEE printed in this image
- If this is page 1 of a multi-page invoice and the totals section is NOT shown — return null for ALL fields
- If the totals area is cut off or not in this crop — return null for ALL fields
- Do NOT calculate totals from line items
- Do NOT guess or estimate any value
- Do NOT invent round numbers
- total_cgst and total_sgst are RUPEE AMOUNTS not percentages
- grand_total is the FINAL amount payable
- Set confidence "low" if the number is partially visible or hard to read
- Return ONLY raw JSON"""


# ── Round 2 Judge Prompts ──────────────────────────────────────────────────────

JUDGE_HEADER_PROMPT = """You are a strict invoice auditor. Your job is to VERIFY and CORRECT extracted data.

You have the invoice image crop (header section) AND the extracted values below.
Some fields were flagged LOW CONFIDENCE — meaning the extraction may be wrong.

Extracted values to verify:
{extracted_json}

For EACH field listed above:
1. Find that exact field in the image — look carefully with fresh eyes
2. Compare what you see vs what was extracted
3. If CORRECT → keep the value, set confidence "high"
4. If WRONG → replace with exactly what the image shows, set confidence "corrected"
5. If NOT VISIBLE → set value null, confidence "low"

STRICT RULES:
- You are a judge. Trust ONLY the image. Do not trust the extracted values.
- Read every character carefully — O vs 0, I vs 1, B vs 8, S vs 5
- Return ALL fields (both corrected and unchanged)
- Return ONLY raw JSON, same structure as input"""


JUDGE_ITEMS_PROMPT = """You are a strict invoice auditor. Your job is to VERIFY and CORRECT extracted data.

You have the invoice items table image AND the extracted rows below.
Some rows were flagged LOW CONFIDENCE — meaning values may be wrong.
values.
- MRP is always >= Rate. If Rate > MRP in your reading, you have columns swapped — fix it.
- Read numbers digit by digit — 568 is not 568, 16.74 is not 16.74 — read carefully
- Return ALL rows (corrected and unchanged)
- Return ONLY raw JSON: {{"items": [...]}}"""


JUDGE_TOTALS_PROMPT = """You are a strict invoice auditor. Your job is to VERIFY and CORRECT extracted data.

You have the invoice footer/totals section image AND the extracted totals below.
Some fields were flagged LOW CONFIDENCE — meaning values may be wrong.

Extracted totals to verify:
{extracted_json}

For EACH field:
1. Find it in the image — look carefully
2. If CORRECT → keep it
3. If WRONG → replace with exactly what is printed
4. If NOT VISIBLE in this image → null

STRICT RULES:
- Trust ONLY the image. Do not trust the extracted values.
- If totals section is not visible in this image, return null for ALL fields
- Do NOT calculate or guess any value
- Return ONLY raw JSON, same structure as input"""


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def get_default_zones(image_bytes: bytes) -> tuple:
    img = Image.open(io.BytesIO(image_bytes))
    ratio = img.height / img.width
    layout = "portrait" if ratio >= 1.0 else "landscape"
    log.info("Layout: %s (ratio=%.2f, %dx%d)", layout, ratio, img.width, img.height)
    return layout, ZONES[layout]


async def extract_from_image(image_bytes: bytes) -> dict:
    """
    4-pass extraction with concurrent execution for speed.
    """
    # ── Zone detection ──────────────────────────────────────────────────────
    layout, zones = get_default_zones(image_bytes)
    try:
        raw = await call_groq(
            ZONE_SCAN_PROMPT,
            img_to_b64(image_bytes),
            "image/png",
            "ZONESCAN",
            max_tokens=600,
        )
        detected = parse_json(raw)
        zones = {
            "header":  normalize_zone(detected.get("header", {}),  zones["header"]),
            "names":   normalize_zone(detected.get("names", {}),   zones["names"]),
            "numbers": normalize_zone(detected.get("numbers", {}), zones["numbers"]),
            "totals":  normalize_zone(detected.get("totals", {}),  zones["totals"]),
        }
        log.info("Zone scan OK: %s", {k: v for k, v in zones.items()})
    except Exception as e:
        log.warning("Zone scan failed, using defaults: %s", e)

    # ── Run all 4 passes concurrently ───────────────────────────────────────
    header_task  = asyncio.create_task(_pass_header(image_bytes, zones["header"]))
    names_task   = asyncio.create_task(_pass_names(image_bytes, zones["names"]))
    numbers_task = asyncio.create_task(_pass_numbers(image_bytes, zones["numbers"]))
    totals_task  = asyncio.create_task(_pass_totals(image_bytes, zones["totals"]))

    header, names_items, numbers_items, totals = await asyncio.gather(
        header_task, names_task, numbers_task, totals_task,
        return_exceptions=False
    )

    items = stitch_items(names_items, numbers_items)
    log.info("Stitched %d items", len(items))

    return {"header": header, "items": items, "totals": totals, "zones": zones}


async def _pass_header(image_bytes: bytes, zone: tuple) -> dict:
    candidates = []
    for idx, z in enumerate(build_crop_variants(zone)):
        try:
            t, b, l, r = z
            crop_bytes, mime = crop_zone(image_bytes, t, b, l, r)
            raw = await call_groq(HEADER_PROMPT, img_to_b64(crop_bytes), mime,
                                  f"HEADER_{idx}")
            candidates.append(parse_json(raw))
        except Exception as e:
            log.warning("Header variant %d failed: %s", idx, e)
    if not candidates:
        log.error("All header variants failed")
        return {}
    fields_list = [extract_header_fields(c)[0] for c in candidates]
    merged = vote_dict(fields_list, [
        "store_name", "address", "gstin", "phone", "bill_no", "date",
        "salesperson", "retailer_name", "retailer_address",
        "retailer_gstin", "fssal_no", "dl_no",
    ])
    return {k: {"value": v, "confidence": "high" if v is not None else "low"}
            for k, v in merged.items()}


async def _pass_names(image_bytes: bytes, zone: tuple) -> list:
    all_lists = []
    for idx, z in enumerate(build_crop_variants(zone)):
        try:
            t, b, l, r = z
            crop_bytes, mime = crop_zone(image_bytes, t, b, l, r)
            raw  = await call_groq(ITEMS_NAMES_PROMPT, img_to_b64(crop_bytes), mime,
                                   f"NAMES_{idx}")
            data = parse_json(raw)
            items = data.get("items", []) if isinstance(data, dict) else (data or [])
            all_lists.append(items)
        except Exception as e:
            log.warning("Names variant %d failed: %s", idx, e)

    by_sno = {}
    for item_list in all_lists:
        for item in item_list:
            n = normalize_item(item)
            sno = n.get("sno")
            if sno is None:
                continue
            by_sno.setdefault(sno, []).append(n)

    result = []
    for sno in sorted(by_sno.keys()):
        variants = by_sno[sno]
        names = [x.get("name") for x in variants if x.get("name")]
        result.append({
            "sno":  sno,
            "name": max(names, key=len) if names else None,
            "hsn":  vote_scalar([x.get("hsn") for x in variants]),
        })
    log.info("Names pass: %d rows", len(result))
    return result


async def _pass_numbers(image_bytes: bytes, zone: tuple) -> list:
    all_lists = []
    for idx, z in enumerate(build_crop_variants(zone)):
        try:
            t, b, l, r = z
            crop_bytes, mime = crop_zone(image_bytes, t, b, l, r)
            raw  = await call_groq(ITEMS_NUMBERS_PROMPT, img_to_b64(crop_bytes), mime,
                                   f"NUMBERS_{idx}")
            data = parse_json(raw)
            items = data.get("items", []) if isinstance(data, dict) else (data or [])
            all_lists.append(items)
        except Exception as e:
            log.warning("Numbers variant %d failed: %s", idx, e)

    # Vote across crop variants
    by_sno = {}
    for item_list in all_lists:
        for item in item_list:
            n = normalize_item(item)
            sno = n.get("sno")
            if sno is None:
                continue
            by_sno.setdefault(sno, []).append(n)

    result = []
    for sno in sorted(by_sno.keys()):
        versions = by_sno[sno]
        merged = {"sno": sno}
        for field in ("mrp", "qty", "rate", "discount", "cgst_pct", "sgst_pct",
                      "taxable", "cgst_amt", "sgst_amt", "amount"):
            merged[field] = vote_scalar([v.get(field) for v in versions])
        result.append(merged)

    log.info("Numbers pass: %d rows", len(result))
    return result


async def _pass_totals(image_bytes: bytes, zone: tuple) -> dict:
    candidates = []
    for idx, z in enumerate(build_crop_variants(zone)):
        try:
            t, b, l, r = z
            crop_bytes, mime = crop_zone(image_bytes, t, b, l, r)
            raw = await call_groq(TOTALS_PROMPT, img_to_b64(crop_bytes), mime,
                                  f"TOTALS_{idx}")
            candidates.append(parse_json(raw))
        except Exception as e:
            log.warning("Totals variant %d failed: %s", idx, e)
    if not candidates:
        log.error("All totals variants failed")
        return {}
    fields_list = [extract_totals_fields(c)[0] for c in candidates]
    merged = vote_dict(fields_list, [
        "subtotal", "total_discount", "total_taxable", "total_cgst",
        "total_sgst", "grand_total", "total_items_count", "amount_in_words",
    ])
    return {k: {"value": v, "confidence": "high" if v is not None else "low"}
            for k, v in merged.items()}


# ══════════════════════════════════════════════════════════════════════════════
# ROUND 2 — JUDGE (self-correction on low confidence fields only)
# ══════════════════════════════════════════════════════════════════════════════

def _get_low_conf_header_fields(header_vals: dict, header_confs: dict) -> dict:
    """Return only the low confidence header fields for re-checking."""
    low = {}
    for field, conf in header_confs.items():
        if conf == "low":
            low[field] = {"value": header_vals.get(field), "confidence": "low"}
    return low


def _get_low_conf_items(items: list) -> list:
    """Return only items rows that are low confidence."""
    return [item for item in items if item.get("confidence") == "low"]


def _get_low_conf_totals(totals_vals: dict, totals_raw: dict) -> dict:
    """Return only low confidence totals fields."""
    low = {}
    for field, raw in totals_raw.items():
        conf = raw.get("confidence", "high") if isinstance(raw, dict) else "high"
        if conf == "low":
            low[field] = {"value": totals_vals.get(field), "confidence": "low"}
    return low


async def judge_header(
    image_bytes: bytes,
    zone: tuple,
    low_conf_fields: dict,
) -> dict:
    """
    Round 2: send header crop + low confidence fields to judge.
    Returns corrected field values.
    """
    if not low_conf_fields:
        return {}
    try:
        t, b, l, r = zone
        crop_bytes, mime = crop_zone(image_bytes, t, b, l, r)
        prompt = JUDGE_HEADER_PROMPT.format(
            extracted_json=json.dumps(low_conf_fields, indent=2)
        )
        raw = await call_groq(prompt, img_to_b64(crop_bytes), mime,
                              "JUDGE_HEADER", max_tokens=2048)
        corrected = parse_json(raw)
        log.info("Judge header corrected %d fields", len(corrected))
        return corrected
    except Exception as e:
        log.warning("Judge header failed: %s", e)
        return {}


async def judge_items(
    image_bytes: bytes,
    names_zone: tuple,
    numbers_zone: tuple,
    low_conf_items: list,
) -> list:
    """
    Round 2: send items crop + low confidence rows to judge.
    Uses a combined crop covering both names and numbers zones.
    Returns corrected rows.
    """
    if not low_conf_items:
        return []
    try:
        # Build a crop that covers both names and numbers zones combined
        t = min(names_zone[0], numbers_zone[0])
        b = max(names_zone[1], numbers_zone[1])
        l = min(names_zone[2], numbers_zone[2])
        r = max(names_zone[3], numbers_zone[3])
        crop_bytes, mime = crop_zone(image_bytes, t, b, l, r)
        prompt = JUDGE_ITEMS_PROMPT.format(
            extracted_json=json.dumps({"items": low_conf_items}, indent=2)
        )
        raw = await call_groq(prompt, img_to_b64(crop_bytes), mime,
                              "JUDGE_ITEMS", max_tokens=4096)
        data = parse_json(raw)
        corrected = data.get("items", []) if isinstance(data, dict) else (data or [])
        log.info("Judge items corrected %d rows", len(corrected))
        return corrected
    except Exception as e:
        log.warning("Judge items failed: %s", e)
        return []


async def judge_totals(
    image_bytes: bytes,
    zone: tuple,
    low_conf_totals: dict,
) -> dict:
    """
    Round 2: send totals crop + low confidence totals to judge.
    Returns corrected totals.
    """
    if not low_conf_totals:
        return {}
    try:
        t, b, l, r = zone
        crop_bytes, mime = crop_zone(image_bytes, t, b, l, r)
        prompt = JUDGE_TOTALS_PROMPT.format(
            extracted_json=json.dumps(low_conf_totals, indent=2)
        )
        raw = await call_groq(prompt, img_to_b64(crop_bytes), mime,
                              "JUDGE_TOTALS", max_tokens=1024)
        corrected = parse_json(raw)
        log.info("Judge totals corrected %d fields", len(corrected))
        return corrected
    except Exception as e:
        log.warning("Judge totals failed: %s", e)
        return {}


def _merge_judge_header(header_vals: dict, header_confs: dict,
                        corrected: dict) -> tuple:
    """Merge judge corrections back into header, tracking what changed."""
    corrected_fields = []
    for field, data in corrected.items():
        if not isinstance(data, dict):
            continue
        new_val = clean_value(data.get("value"))
        new_conf = data.get("confidence", "high")
        old_val = header_vals.get(field)
        if new_val != old_val:
            corrected_fields.append(field)
            log.info("Judge corrected header.%s: %r → %r", field, old_val, new_val)
        header_vals[field] = new_val
        header_confs[field] = new_conf
    return header_vals, header_confs, corrected_fields


def _merge_judge_items(items: list, corrected_rows: list) -> tuple:
    """Merge judge corrections back into items list by sno."""
    corrected_snos = []
    corrected_by_sno = {}
    for row in corrected_rows:
        n = normalize_item(row)
        sno = n.get("sno")
        if sno is not None:
            corrected_by_sno[sno] = n

    result = []
    for item in items:
        sno = item.get("sno")
        if sno in corrected_by_sno:
            merged = dict(item)
            for field in ("mrp", "qty", "rate", "discount", "cgst_pct", "sgst_pct",
                          "taxable", "cgst_amt", "sgst_amt", "amount", "name", "hsn"):
                new_val = corrected_by_sno[sno].get(field)
                if new_val is not None and new_val != item.get(field):
                    log.info("Judge corrected item %d .%s: %r → %r",
                             sno, field, item.get(field), new_val)
                    corrected_snos.append(sno)
                if new_val is not None:
                    merged[field] = new_val
            merged["confidence"] = "corrected"
            result.append(merged)
        else:
            result.append(item)

    return result, list(set(corrected_snos))


def _merge_judge_totals(totals_vals: dict, corrected: dict) -> tuple:
    """Merge judge corrections back into totals."""
    corrected_fields = []
    for field, data in corrected.items():
        if not isinstance(data, dict):
            continue
        new_val = clean_value(data.get("value"))
        if field in ("subtotal", "total_discount", "total_taxable",
                     "total_cgst", "total_sgst", "grand_total"):
            new_val = clean_number(new_val)
        elif field == "total_items_count":
            new_val = clean_int(new_val)
        old_val = totals_vals.get(field)
        if new_val != old_val:
            corrected_fields.append(field)
            log.info("Judge corrected totals.%s: %r → %r", field, old_val, new_val)
        totals_vals[field] = new_val
    return totals_vals, corrected_fields

async def ocr_invoice(image_bytes: bytes, filename: str) -> dict:
    t0 = time.perf_counter()

    # ── Round 1: Extract ────────────────────────────────────────────────────
    extraction = await extract_from_image(image_bytes)
    zones      = extraction["zones"]

    raw_header   = extraction["header"]
    header_vals, header_confs = extract_header_fields(raw_header)

    if header_vals.get("bill_no"):
        header_vals["bill_no"] = clean_bill_no(str(header_vals["bill_no"]))
    if header_vals.get("gstin"):
        header_vals["gstin"] = clean_gstin(str(header_vals["gstin"]))
    if header_vals.get("retailer_gstin"):
        header_vals["retailer_gstin"] = clean_gstin(str(header_vals["retailer_gstin"]))

    items      = extraction["items"]
    raw_totals = extraction["totals"]
    totals_vals, _ = extract_totals_fields(raw_totals)
    totals_vals = normalize_totals(totals_vals)
    totals_vals = reconcile_totals_from_items(items, totals_vals)

    # ── Round 2: Judge low confidence fields ────────────────────────────────
    low_header = _get_low_conf_header_fields(header_vals, header_confs)
    low_items  = _get_low_conf_items(items)
    low_totals = _get_low_conf_totals(totals_vals, raw_totals)

    judge_audit = {
        "low_conf_header_fields":  list(low_header.keys()),
        "low_conf_item_rows":      [i.get("sno") for i in low_items],
        "low_conf_total_fields":   list(low_totals.keys()),
        "corrected_header_fields": [],
        "corrected_item_rows":     [],
        "corrected_total_fields":  [],
    }

    if low_header or low_items or low_totals:
        log.info("Round 2 judge — header:%d items:%d totals:%d",
                 len(low_header), len(low_items), len(low_totals))

        async def _noop_dict(): return {}
        async def _noop_list(): return []

        j_header_coro = judge_header(image_bytes, zones["header"], low_header) \
                        if low_header else _noop_dict()
        j_items_coro  = judge_items(image_bytes, zones["names"],
                                    zones["numbers"], low_items) \
                        if low_items else _noop_list()
        j_totals_coro = judge_totals(image_bytes, zones["totals"], low_totals) \
                        if low_totals else _noop_dict()

        j_header, j_items, j_totals = await asyncio.gather(
            j_header_coro, j_items_coro, j_totals_coro
        )

        if j_header:
            header_vals, header_confs, fixed = _merge_judge_header(
                header_vals, header_confs, j_header)
            judge_audit["corrected_header_fields"] = fixed

        if j_items:
            items, fixed_snos = _merge_judge_items(items, j_items)
            judge_audit["corrected_item_rows"] = fixed_snos

        if j_totals:
            totals_vals, fixed = _merge_judge_totals(totals_vals, j_totals)
            judge_audit["corrected_total_fields"] = fixed

        log.info("Round 2 done: %s", judge_audit)
    else:
        log.info("Round 2 skipped — all fields high confidence")

    # ── Validation ───────────────────────────────────────────────────────────
    validation = {}
    try:
        item_sum = sum(float(i.get("amount") or 0)
                       for i in items if i.get("amount") is not None)
        gt = float(totals_vals.get("grand_total") or 0)
        if gt > 0:
            diff_pct = abs(item_sum - gt) / gt * 100
            validation["items_sum"]         = round(item_sum, 2)
            validation["grand_total_match"] = diff_pct < 5
            validation["diff_pct"]          = round(diff_pct, 1)
    except Exception:
        pass

    return {
        "filename":          filename,
        "status":            "success",
        "store_name":        header_vals.get("store_name"),
        "address":           header_vals.get("address"),
        "gstin":             header_vals.get("gstin"),
        "phone":             header_vals.get("phone"),
        "bill_no":           header_vals.get("bill_no"),
        "date":              header_vals.get("date"),
        "salesperson":       header_vals.get("salesperson"),
        "retailer_name":     header_vals.get("retailer_name"),
        "retailer_address":  header_vals.get("retailer_address"),
        "retailer_gstin":    header_vals.get("retailer_gstin"),
        "fssal_no":          header_vals.get("fssal_no"),
        "dl_no":             header_vals.get("dl_no"),
        "items":             items,
        "subtotal":          totals_vals.get("subtotal"),
        "total_discount":    totals_vals.get("total_discount"),
        "total_taxable":     totals_vals.get("total_taxable"),
        "total_cgst":        totals_vals.get("total_cgst"),
        "total_sgst":        totals_vals.get("total_sgst"),
        "grand_total":       totals_vals.get("grand_total"),
        "total_items_count": totals_vals.get("total_items_count"),
        "amount_in_words":   totals_vals.get("amount_in_words"),
        "confidence":        header_confs,
        "validation":        validation,
        "judge_audit":       judge_audit,
        "time_seconds":      round(time.perf_counter() - t0, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ocr", response_class=JSONResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 20MB.")

    try:
        processed, orient_info = smart_orient_and_preprocess(image_bytes)
        log.info("Orientation: %s", orient_info)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {e}")

    try:
        result = await ocr_invoice(processed, file.filename)
        result["orientation"] = orient_info
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
async def health():
    if not GROQ_API_KEY:
        return {"status": "degraded", "error": "GROQ_API_KEY not set"}
    try:
        resp = await http_client.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        return {
            "status": "ok",
            "provider": "groq",
            "model": MODEL,
            "model_available": MODEL in models,
        }
    except Exception as e:
        return {"status": "degraded", "provider": "groq", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)