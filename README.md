# 🧾 Groq Vision Invoice OCR

> *Because manually typing invoice data is a crime , and you deserve better.*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq%20%E2%9A%A1-f55036.svg)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Invoices Feared](https://img.shields.io/badge/invoices-feared-purple.svg)](#)

---

**Groq Vision Invoice OCR v9.0** is a blazing-fast, hilariously over-engineered invoice extraction API that reads your messy, blurry, sideways-scanned Indian GST invoices and returns clean structured JSON — like a caffeinated accountant who never sleeps, never complains, and never asks for a raise.

It uses **Llama 4 Scout** via Groq to run 4 concurrent extraction passes on every invoice, then has the audacity to *second-guess itself* in a Round 2 judge pass. Accuracy-obsessed doesn't begin to cover it.

---

## ✨ Features

- **4-pass concurrent extraction** — Header, item names, item numbers, and totals all run in parallel. Your invoice doesn't even have time to feel violated.
- **Self-correcting judge round** — Low-confidence fields get sent back to the model for a second opinion. It's peer review, but for robots.
- **Smart orientation handling** — EXIF transpose so your phone's sideways selfie-mode scans actually work.
- **Zone-aware cropping** — The model first analyzes layout, then crops intelligently. Like a surgeon, but for PDFs.
- **Multi-variant voting** — 3 crop variants per zone, results voted on like a dysfunctional democracy that somehow works.
- **Indian GST invoice native** — Understands GSTIN, HSN codes, CGST/SGST, FSSAI, Drug License numbers. Basically a GST-compliant oracle.
- **Validation** — Checks if item amounts sum to grand total. Calls out discrepancies like an auditor who found something.
- **FastAPI + async** — Handles concurrent requests without breaking a sweat. Runs on `uvicorn`. Ships with a health endpoint for your peace of mind.

---

## 🏗️ Architecture

```
                      YOUR CURSED INVOICE IMAGE
                               │
                    ┌──────────▼──────────┐
                    │  EXIF Transpose      │  ← fixes phone metadata rotation
                    │  Contrast +1.4x      │  ← because scanners hate you
                    │  Sharpness +1.6x     │  ← squint less
                    │  Upscale to 1200px   │  ← the model needs reading glasses too
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   ZONE SCAN PASS     │  ← model detects layout zones
                    │   (portrait/landscape│
                    │    fallback defaults)│
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐    ┌───────▼──────┐    ┌──────▼──────┐
    │  PASS 1     │    │  PASS 2+3    │    │  PASS 4     │
    │  Header     │    │  Items       │    │  Totals     │
    │  (3 crops)  │    │  Names+Nums  │    │  (3 crops)  │
    │  ↓ vote     │    │  (3 crops ea)│    │  ↓ vote     │
    └──────┬──────┘    └──────┬───────┘    └──────┬──────┘
           │                  │ stitch()           │
           └──────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   ROUND 2: JUDGE   │  ← low confidence fields only
                    │   (concurrent)     │  ← sends back to model to verify
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   VALIDATION       │  ← sum(items) ≈ grand_total?
                    │   + CLEAN JSON     │  ← you get clean data or we die trying
                    └───────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (it's free, stop procrastinating)
- An invoice. Hopefully not yours from 2019.

### Installation

```bash
git clone https://github.com/yourusername/groq-invoice-ocr
cd groq-invoice-ocr
pip install -r requirements.txt
```

### Setup

```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
# Or just export it like a normal person:
export GROQ_API_KEY=gsk_your_key_here
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Or if you like living dangerously:

```bash
python main.py
```

---

## 📡 API Reference

### `POST /ocr`

Upload an invoice image. Receive structured JSON. Cry tears of joy.
---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Your Groq API key. Don't commit this. Seriously. |
| `MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | The brain of the operation |
| `TIMEOUT` | `120s` | How long we wait before giving up, like a parking meter |
| `MAX_RETRIES` | `3` | Attempts before we admit defeat (we rarely do) |
| `MAX_FILE_SIZE` | `20MB` | No, you cannot upload a RAW from your DSLR |
| `GROQ_CONCURRENCY` | `8` | Parallel Groq calls. Crank it if you're feeling bold |

---

## 📊 What Gets Extracted

### Header Fields
`store_name` · `address` · `gstin` · `phone` · `bill_no` · `date` · `salesperson` · `retailer_name` · `retailer_address` · `retailer_gstin` · `fssal_no` · `dl_no`

### Per-Item Fields
`sno` · `name` · `hsn` · `mrp` · `qty` · `rate` · `discount` · `cgst_pct` · `sgst_pct` · `taxable` · `cgst_amt` · `sgst_amt` · `amount` · `confidence`

### Totals
`subtotal` · `total_discount` · `total_taxable` · `total_cgst` · `total_sgst` · `grand_total` · `total_items_count` · `amount_in_words`

---

## 🧠 How the Judge Round Works

After Round 1, every field gets a confidence score. Anything marked `low` gets sent back for a Round 2 judge pass — a separate Groq call where the model looks at the same crop with fresh eyes and the extracted value side-by-side.

Think of it as: *"Hey, are you SURE that GSTIN is `27AAAAA0000A1Z5` and not `27AAAAA000OA1Z5`?"*

The judge often catches O vs 0, I vs 1, B vs 8, S vs 5 — the classic OCR villain quartet.

---

## 🗂️ Project Structure

```
groq-invoice-ocr/
├── main.py              # The whole enchilada (~800 lines of beautiful chaos)
├── static/
│   └── index.html       # Optional frontend (you built one, right?)
├── .env                 # Your secrets. Guard them with your life.
├── .env.example         # Safe to commit. Unlike your actual .env.
├── requirements.txt     # pip install -r this and touch grass
└── README.md            # You are here
```

---

## 🔧 Requirements

```
fastapi
uvicorn
httpx
pillow
python-dotenv
```

---

## 🤝 Contributing

PRs welcome. If you're going to refactor the voting logic, please bring snacks.

1. Fork it
2. Branch it (`git checkout -b feat/make-it-faster`)
3. Code it
4. Test it on an actual invoice (we know you have some lying around)
5. PR it

---

*"It's not about the invoices. It's about the JSON we extracted along the way."*
