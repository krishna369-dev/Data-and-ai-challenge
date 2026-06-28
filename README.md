# Redrob Candidate Ranker — README

## Overview
This project implements a hybrid AI candidate ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Architecture
```
candidates.jsonl
       ↓
  Honeypot Filter (honeypots.py)
       ↓
  Heuristic Feature Scorer
       ├── Skills (regex + bloom match) — 35%
       ├── Career quality (product vs service)— 25%
       ├── Experience band (5–9 yrs peak)  — 20%
       ├── Location (Pune/Noida preferred)  — 10%
       └── Education tier                   — 10%
       ↓
  Semantic Similarity (MiniLM-L6-v2, precomputed)
       ↓
  Rank Fusion (55% heuristic + 45% semantic)
       ↓
  Behavioral Modifier (notice, activity, responsiveness)
       ↓
  Top 100 → submission.csv
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pre-compute embeddings (offline, run once)
```bash
python precompute.py --candidates ./candidates.jsonl --out ./embeddings.npz
```
This generates `embeddings.npz` (~150 MB). Runtime: ~5–10 minutes on CPU. **Must be committed to the repository.**

### 3. Run the ranker (the submission step)
```bash
python rank.py --candidates ./candidates.jsonl --embeddings ./embeddings.npz --out ./submission.csv
```
Completes in **< 5 minutes** on CPU, 16 GB RAM, no network.

### 4. Validate the submission
```bash
python validate_submission.py ./submission.csv
```

## Files
| File | Purpose |
|------|---------|
| `rank.py` | Main ranking entry point |
| `precompute.py` | Offline embedding generation |
| `honeypots.py` | Programmatic honeypot detection |
| `requirements.txt` | Python dependencies |
| `embeddings.npz` | Pre-computed candidate embeddings (generated, not tracked in git LFS unless needed) |
| `submission.csv` | Final ranked top-100 output |

## Honeypot Detection
Our system detects synthetic anomalies via:
- Career timeline math mismatch (claimed duration vs actual date span)
- Expert/advanced skill with near-zero usage duration
- Education date reversal (start_year > end_year)
- Impossible platform signals (last_active before signup, future signup dates)
- Skill duration exceeding total career length

## Scoring Weights
| Component | Weight |
|-----------|--------|
| Skills match | 35% |
| Career quality | 25% |
| Experience band | 20% |
| Location fit | 10% |
| Education tier | 10% |
| Behavioral modifier | Scaling × 0.4–1.0 |
| Semantic (MiniLM cosine) | Blended at 45% final |
