# Redrob — Intelligent Candidate Discovery & Ranking System

A two-stage retrieval and ranking pipeline for the Redrob AI Hackathon challenge.
Ranks 100,000 candidates against a Senior AI Engineer JD using semantic embeddings and confidence-weighted feature scoring.

## Architecture

```
100,000 Candidates (candidates.jsonl)
        ↓
  [Stage 1 — High Recall Retrieval]
  Semantic Search (MiniLM dot product) + Required Skill + Experience Filter
        ↓
     Top 2,000 Candidates
        ↓
  [Stage 2 — Precision Ranking]
  Three-Score Model: Retrieval × Confidence_Adjusted + Evidence
  + Hard Honeypot Filter
        ↓
     Top 100 → submission.csv
```

**Six modules:** `src/parser/` → `src/index/` → `src/retrieval/` → `src/ranking/` → `src/explanation/` → `src/validation/`

---

## Quickstart (Reproduce in < 5 minutes)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download pre-computed embeddings
```bash
python precompute.py --download --out ./embeddings.npz
```
> This downloads the pre-computed `embeddings.npz` (~150MB) from HuggingFace in ~30 seconds.
> No network access needed after this step.

Alternatively, compute from scratch (takes ~38 minutes, no GPU needed):
```bash
python precompute.py --candidates ./candidates.jsonl --out ./embeddings.npz
```

### Step 3 — Run the ranker (the submission step)
```bash
python -m src.main --candidates ./candidates.jsonl --embeddings ./embeddings.npz --out ./submission.csv
```
Completes in **< 30 seconds** on CPU, 16 GB RAM, **no network access**.

### Step 4 — Validate the output
```bash
python validate_submission.py ./submission.csv
```

---

## File Structure

```
├── src/
│   ├── parser/
│   │   ├── candidate_parser.py   # Schema variance logging + confidence scoring
│   │   └── jd_parser.py          # Required vs preferred skill separation
│   ├── index/
│   │   ├── embedding_index.py    # Fast dot-product semantic search
│   │   └── feature_index.py      # IDF skill weighting + feature caching
│   ├── retrieval/
│   │   └── stage1.py             # Top 2000 high-recall filter
│   ├── ranking/
│   │   ├── stage2.py             # Three-score model + honeypot hard filter
│   │   └── calibration.py        # Spearman weight optimization
│   ├── explanation/
│   │   └── generator.py          # Deterministic 1-2 sentence reasonings
│   └── validation/
│       ├── diagnostics.py        # Pipeline runtime report
│       └── ablation.py           # Signal importance testing
├── honeypots.py                  # Programmatic honeypot detection (6 checks)
├── precompute.py                 # Embedding pre-computation + HF download
├── requirements.txt              # Python dependencies
├── submission.csv                # Final top-100 ranked candidates
└── submission_metadata.yaml      # Submission metadata
```

---

## Scoring Model

**Final Score Formula:**
```python
confidence_adjusted = 0.6 + 0.4 * confidence   # soft floor: range [0.6, 1.0]
final = (retrieval * confidence_adjusted) + evidence
```

| Component | Description |
|-----------|-------------|
| `retrieval` | Semantic similarity + required skill match + experience fit |
| `confidence` | Feature-level extraction confidence (1.0 = explicit skills section, 0.8 = career text, 0.6 = summary) |
| `evidence` | Behavioral signals: response rate, interview completion, recency |

---

## Honeypot Detection

Six hard checks — any single failure drops the candidate:
1. Career timeline math mismatch (claimed vs actual duration > 6 months AND > 20%)
2. Expert/advanced skills with ≤ 3 months usage (3 or more)
3. Education date reversal (start_year > end_year)
4. Platform activity before signup date
5. Future signup date
6. Skill duration exceeds total career length

**Result: 0 honeypots in Top 100.**

---

## Performance

| Metric | Value |
|--------|-------|
| Candidates processed | 100,000 |
| Ranking runtime | **15.55 seconds** |
| GPU required | No |
| Network during ranking | None |
| Honeypots in Top 100 | 0 |

---

## GitHub Repository
[https://github.com/krishna369-dev/Data-and-ai-challenge](https://github.com/krishna369-dev/Data-and-ai-challenge)
