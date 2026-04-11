# Equity Research Report Agent
> Automated sell-side equity research report generation using RAG, LangGraph, and bank-specific FCFE valuation — built for Goldman Sachs (GS)

---

## Overview

This project automates the generation of professional equity research reports for bank holding companies using a multi-agent LangGraph pipeline. Built for Goldman Sachs (GS), the system combines three layers:

- **`main.ipynb`** — the driver. Supplies the required parameters: ticker, file paths, API key, filing date. Think of it as the managing director who says "write a report on GS using these files."
- **`orchestrator.py`** — the orchestrator. Uses those parameters to run an 8-section LangGraph pipeline. Think of it as the analyst who knows how to structure and write the report.
- **`rag_store.py`** — the briefing packet. Feeds Goldman-specific data into the LLM that it cannot know from training alone — regulatory VaR numbers, CET1 ratios, RWA methodology, and management commentary from the 2024 Annual Report.

The LLM (Gemini 2.5 Flash via OpenRouter) provides general banking knowledge and report-writing ability. The RAG store provides the Goldman-specific facts. The DCF tool calculates the price target. Together they produce a report that reads like it was written by someone who actually read Goldman's filings.

Built as a term project for the Large Language Models course, Financial Engineering Program, Rutgers University (Spring 2026).

---

## How the three layers work together

```
main.ipynb            orchestrator.py           rag_store.py
──────────────────    ───────────────────────   ─────────────────────────
Supplies:             Uses parameters to:       Feeds LLM with:
  ticker = 'GS'         run 8-section graph      Goldman VaR: $407M (Q3 2025)
  filing_date           call LLM per section     CET1: 14.5% as of Sep 2025
  csv_folder            call tools               RWA approach: IMA
  api_key               compile final report     2024 Annual Report commentary
```

```
LLM knows                           RAG store provides
───────────────────────────────────────────────────────────
General banking knowledge           Goldman Q3 2025 VaR: $407M
What CET1 ratio means               Goldman actual CET1: 14.5%
How to write a valuation section    Goldman RWA approach: IMA methodology
Industry report structure           Management commentary from Annual Report
Basel III framework                 Regulatory data not available in yfinance
```

The LangGraph graph runs 8 nodes per section in a tool-call loop:

```
plan → llm → parse → tool? → llm → ... → save → next section → compile
```

---

## Architecture

```
market_risk.ipynb          main.ipynb
      │                        │
      ▼                        ▼
data_loader.py ──────► rag_store.py ──────► orchestrator.py ──► PDF Report
                              ▲                    │
                         tools.py            llm_engine.py
                         dcf_tool.py         state.py
```

---

## Report Sections

| # | Section | Key Content |
|---|---------|-------------|
| 1 | Executive Summary | Key figures table, rating, price target, upside % |
| 2 | Business Overview | Revenue segments, margins, competitive moat |
| 3 | Financial Performance | 2-year trend, EPS, ROTCE, efficiency ratio |
| 4 | Industry Analysis | TAM, peers, Basel III / CCAR regulatory environment |
| 5 | Valuation | FCFE DCF + P/B + Gordon Growth — blended price target |
| 6 | Risks | 5 bank-specific risks rated H/M/L |
| 7 | Investment Thesis | Bull / base / bear case, final Buy/Hold/Sell |
| 8 | References | All data sources with dates |

---

## Data Sources

| Source | What it provides | File |
|--------|-----------------|------|
| FR Y-9C | Total assets, trading assets, market risk RWA | `FRY9C_<inst_id>_<date>.csv` |
| FFIEC 102 | Regulatory VaR, SVaR, multiplicative factor | `FFIEC102_<inst_id>_<date>.csv` |
| Yahoo Finance | Live price, P/E, P/B, EPS, market cap | via `yfinance` |
| Annual Report | Qualitative risk methodology, RWA approach | PDF ingested into RAG |
| Style references | Report formatting patterns | Qualcomm + Citigroup PDFs |

Goldman Sachs institution ID: `2380443`

---

## Project Structure

```
project/
├── notebook/
│   ├── orchestrator.py      # LangGraph graph — Lead
│   ├── state.py             # ReportState TypedDict — Lead
│   ├── llm_engine.py        # OpenRouter client, section prompts — Person B
│   ├── tools.py             # Tool implementations — Person C
│   ├── data_loader.py       # FR Y-9C + FFIEC 102 pipeline — Person C
│   ├── dcf_tool.py          # Bank FCFE DCF model — Person C
│   ├── rag_store.py         # FAISS vector store — Person D
│   └── main.ipynb           # Driver notebook
├── data/
│   ├── gs_filings/          # Regulatory CSV files
│   └── equity_research_report_templates/
│       ├── Qualcomm.pdf     # Style reference
│       └── Citigroup.pdf    # Style reference
├── .env                     # API keys — never committed
├── .gitignore
└── README.md
```

---

## Valuation Methodology

Bank FCFE model from `dcf_tool.py` — based on regulatory capital reinvestment:

```
Total Assets × RWA%           = Risk-Adjusted Assets
Risk-Adjusted Assets × CET1%  = CET1 Capital
CET1_t − CET1_{t-1}           = Regulatory reinvestment  (the "capex" for a bank)
Net Income − Regulatory reinvest = FCFE
```

CET1 ratio steps from 13.63% → 16.63% at year 3 to reflect Basel III buffer build. Blended price target combines FCFE DCF (33%), P/B multiple (33%), and Gordon Growth implied P/B (33%).

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/manojmatala/Equity-Research-Report-Agent.git
cd Equity-Research-Report-Agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file

Create a file named `.env` in the project root — never commit this file:

```
OPENROUTER_API_KEY=your-openrouter-key-here
```

Get a key at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).

### 4. Download regulatory data

Download FR Y-9C and FFIEC 102 filings for Goldman Sachs (`inst_id = 2380443`) from [ffiec.gov](https://www.ffiec.gov/npw/) and place them in:

```
data/gs_filings/FRY9C_2380443_20250930.csv
data/gs_filings/FFIEC102_2380443_20250930.csv
```

---

## Running the Pipeline

### Step 1 — Generate regulatory data

Open and run all cells in `market_risk.ipynb`. This produces:
```
output/fry9c_data.csv
output/fiec102_data.csv
```

### Step 2 — Generate the report

Open `main.ipynb` and run all cells top to bottom. The pipeline:
1. Ingests style reference PDFs into FAISS
2. Ingests Goldman regulatory data
3. Runs the 8-section LangGraph pipeline (~10-15 minutes)
4. Exports markdown + PDF to `data/`

---

## LLM Configuration

Default model: `google/gemini-2.5-flash` via OpenRouter.

To switch models, edit one line in `notebook/llm_engine.py`:

```python
MODEL_ID = "openai/gpt-4o"           # most reliable
MODEL_ID = "google/gemini-2.5-flash" # faster, lower cost
```



## Requirements

- Python 3.10+
- OpenRouter API key (paid account)
- Pandoc + MiKTeX (for PDF export)

