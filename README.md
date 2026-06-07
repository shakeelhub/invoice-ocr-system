<div align="center">

```
██╗███╗   ██╗██╗   ██╗ ██████╗ ██╗ ██████╗███████╗     ██████╗  ██████╗██████╗ 
██║████╗  ██║██║   ██║██╔═══██╗██║██╔════╝██╔════╝    ██╔═══██╗██╔════╝██╔══██╗
██║██╔██╗ ██║██║   ██║██║   ██║██║██║     █████╗      ██║   ██║██║     ██████╔╝
██║██║╚██╗██║╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝      ██║   ██║██║     ██╔══██╗
██║██║ ╚████║ ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗    ╚██████╔╝╚██████╗██║  ██║
╚═╝╚═╝  ╚═══╝  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝     ╚═════╝  ╚═════╝╚═╝  ╚═╝
```

### 🧾 Because manually typing invoice data is a crime — and you deserve better.

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq_⚡-f55036?style=for-the-badge&logo=lightning&logoColor=white"/>
  <img src="https://img.shields.io/badge/Llama_4_Scout-6366F1?style=for-the-badge&logo=meta&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Invoices-Feared-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accountants-Replaced-red?style=for-the-badge"/>
</p>

</div>

---

<div align="center">

## 〔 WHAT IS THIS? 〕

</div>

**Groq Vision Invoice OCR v9.0** is a blazing-fast, hilariously over-engineered invoice extraction API.

Feed it your messy, blurry, sideways-scanned Indian GST invoices.  
Get back clean structured JSON.  

Like a **caffeinated accountant who never sleeps, never complains, and never asks for a raise.**

It runs **4 concurrent extraction passes** on every invoice, then has the audacity to *second-guess itself* in a Round 2 judge pass.  
Accuracy-obsessed doesn't begin to cover it.

---

## ⚡ Feature Arsenal

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   4-PASS CONCURRENT      Header · Item Names · Item Numbers · Totals    ║
║   EXTRACTION             All run in parallel. Invoice doesn't even       ║
║                          have time to feel violated.                     ║
║                                                                          ║
║   SELF-CORRECTING        Low-confidence fields get a second opinion.    ║
║   JUDGE ROUND            Peer review — but for robots.                  ║
║                                                                          ║
║   SMART ORIENTATION      EXIF transpose so your phone's sideways        ║
║   HANDLING               selfie-mode scans actually work.               ║
║                                                                          ║
║   ZONE-AWARE CROPPING    Model analyzes layout first, then crops.       ║
║                          Like a surgeon — but for PDFs.                 ║
║                                                                          ║
║   MULTI-VARIANT VOTING   3 crop variants per zone, voted on like a      ║
║                          dysfunctional democracy that somehow works.    ║
║                                                                          ║
║   INDIAN GST NATIVE      GSTIN · HSN · CGST/SGST · FSSAI · Drug Lic.  ║
║                          Basically a GST-compliant oracle.              ║
║                                                                          ║
║   VALIDATION             Checks if item amounts sum to grand total.     ║
║                          Calls out discrepancies like a furious auditor.║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🏗️ Architecture

```
                      YOUR CURSED INVOICE IMAGE
                               │
                    ┌──────────▼──────────┐
                    │   IMAGE ENHANCEMENT  │
                    │   EXIF Transpose     │  ← fixes phone metadata rotation
                    │   Contrast  × 1.4   │  ← because scanners hate you
                    │   Sharpness × 1.6   │  ← squint less
                    │   Upscale → 1200px  │  ← the model needs reading glasses
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    ZONE SCAN PASS    │  ← model detects layout zones
                    │  portrait/landscape  │  ← fallback defaults if needed
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐    ┌───────▼──────┐    ┌──────▼──────┐
    │   PASS 1    │    │  PASS 2 + 3  │    │   PASS 4    │
    │   Header    │    │    Items     │    │   Totals    │
    │  (3 crops)  │    │ Names + Nums │    │  (3 crops)  │
    │   ↓ vote    │    │ (3 crops ea) │    │   ↓ vote    │
    └──────┬──────┘    └──────┬───────┘    └──────┬──────┘
           │                  │  stitch()          │
           └──────────────────┼────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   ROUND 2: JUDGE   │  ← low confidence fields only
                    │   (concurrent)     │  ← fresh eyes on bad extractions
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │    VALIDATION      │  ← sum(items) ≈ grand_total?
                    │    + CLEAN JSON    │  ← you get clean data or we die
                    └────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```
✅ Python 3.10+
✅ Groq API key  →  console.groq.com (it's free, stop procrastinating)
✅ An invoice    →  hopefully not yours from 2019
```

### 1️⃣ Clone & Install

```bash
git clone https://github.com/yourusername/groq-invoice-ocr
cd groq-invoice-ocr
pip install -r requirements.txt
```

### 2️⃣ Setup Environment

```bash
cp .env.example .env

# Add your key:
export GROQ_API_KEY=gsk_your_key_here
```

### 3️⃣ Run

```bash
# Normal person mode
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Living dangerously mode
python main.py
```

---

## 📡 API

### `POST /ocr`

> Upload an invoice image. Receive structured JSON. Cry tears of joy.

```bash
curl -X POST http://localhost:8001/ocr \
  -F "file=@your_invoice.jpg"
```

---

## 📊 What Gets Extracted

<table>
<tr>
<td><b>🏷️ Header</b></td>
<td><code>store_name</code> · <code>address</code> · <code>gstin</code> · <code>phone</code> · <code>bill_no</code> · <code>date</code> · <code>salesperson</code> · <code>retailer_name</code> · <code>retailer_gstin</code> · <code>fssai_no</code> · <code>dl_no</code></td>
</tr>
<tr>
<td><b>📦 Per Item</b></td>
<td><code>sno</code> · <code>name</code> · <code>hsn</code> · <code>mrp</code> · <code>qty</code> · <code>rate</code> · <code>discount</code> · <code>cgst_pct</code> · <code>sgst_pct</code> · <code>taxable</code> · <code>cgst_amt</code> · <code>sgst_amt</code> · <code>amount</code> · <code>confidence</code></td>
</tr>
<tr>
<td><b>💰 Totals</b></td>
<td><code>subtotal</code> · <code>total_discount</code> · <code>total_taxable</code> · <code>total_cgst</code> · <code>total_sgst</code> · <code>grand_total</code> · <code>total_items_count</code> · <code>amount_in_words</code></td>
</tr>
</table>

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Your Groq API key. Don't commit this. Seriously. |
| `MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | The brain of the operation |
| `TIMEOUT` | `120s` | How long we wait before giving up, like a parking meter |
| `MAX_RETRIES` | `3` | Attempts before we admit defeat (we rarely do) |
| `MAX_FILE_SIZE` | `20MB` | No, you cannot upload a RAW from your DSLR |
| `GROQ_CONCURRENCY` | `8` | Parallel calls. Crank it if you're feeling bold |

---

## 🧠 How the Judge Round Works

After Round 1, every field gets a **confidence score**.  
Anything marked `low` gets sent back for a Round 2 judge pass.

```
Round 1 extracts:   GSTIN → "27AAAAA000OA1Z5"  ← confidence: LOW
                              ↓
Round 2 judge:      "Are you SURE that's not '27AAAAA0000A1Z5'?"
                              ↓
Judge corrects:     GSTIN → "27AAAAA0000A1Z5"  ✅
```

The classic OCR villain quartet it catches every time:

```
O  vs  0     |     I  vs  1     |     B  vs  8     |     S  vs  5
```

---

## 🗂️ Project Structure

```
groq-invoice-ocr/
│
├── 🧠 main.py              # The whole enchilada (~800 lines of beautiful chaos)
├── 🌐 static/
│   └── index.html          # Optional frontend (you built one, right?)
├── 🔒 .env                 # Your secrets. Guard with your life.
├── 📋 .env.example         # Safe to commit. Unlike your actual .env.
├── 📦 requirements.txt     # pip install -r this and touch grass
└── 📖 README.md            # You are here
```

---

## 🤝 Contributing

PRs welcome. If you're going to refactor the voting logic, please bring snacks.

```bash
git checkout -b feat/make-it-faster
# code it, test it on an actual invoice (we know you have some lying around)
git commit -m "feat: something that slaps"
git push origin feat/make-it-faster
# open a PR 🚀
```

---

<div align="center">

```
"It's not about the invoices.
 It's about the JSON we extracted along the way."
```

**⭐ Star this if it saved you from a spreadsheet. It really means a lot.**

</div>
