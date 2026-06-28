"""
precompute.py — Offline embedding pre-computation for Redrob Hackathon.

Run ONCE (no time limit, network OK):
    python precompute.py --candidates ./candidates.jsonl --out ./embeddings.npz

Generates a compressed NumPy array of shape (N, 384) using all-MiniLM-L6-v2.
Candidate IDs are stored in a parallel array so rank.py can align them.

Requirements:
    pip install sentence-transformers numpy tqdm
"""
import argparse
import json
import gzip
import sys
import time

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_text(c: dict) -> str:
    """
    Construct a rich text passage for the candidate that captures
    semantically meaningful information for the Senior AI Engineer JD.
    We intentionally weight the job-relevant parts more by repeating them.
    """
    p = c.get("profile", {})
    parts = []

    # Headline + Summary (most informative)
    headline = p.get("headline", "")
    summary = p.get("summary", "")
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    # Career history – title + description (captures what they actually did)
    for job in c.get("career_history", []):
        title = job.get("title", "")
        desc = job.get("description", "")
        company = job.get("company", "")
        if title or desc:
            parts.append(f"{title} at {company}. {desc}")

    # Skills (name + proficiency level)
    skill_tokens = []
    for sk in c.get("skills", []):
        name = sk.get("name", "")
        prof = sk.get("proficiency", "")
        if name:
            skill_tokens.append(f"{name} ({prof})")
    if skill_tokens:
        parts.append("Skills: " + ", ".join(skill_tokens))

    # Certifications
    cert_tokens = [ce.get("name", "") for ce in c.get("certifications", []) if ce.get("name")]
    if cert_tokens:
        parts.append("Certifications: " + ", ".join(cert_tokens))

    return " | ".join(parts)


def open_candidates(path: str):
    """Open candidates.jsonl or candidates.jsonl.gz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute candidate embeddings.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, help="Output .npz file path")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=512, help="Encoding batch size")
    args = parser.parse_args()

    print(f"[precompute] Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"[precompute] Reading candidates from: {args.candidates}")
    ids = []
    texts = []
    t0 = time.time()

    with open_candidates(args.candidates) as f:
        for line in tqdm(f, desc="Reading", unit="cand"):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            ids.append(c["candidate_id"])
            texts.append(build_candidate_text(c))

    print(f"[precompute] Loaded {len(ids):,} candidates in {time.time()-t0:.1f}s")

    print(f"[precompute] Encoding {len(texts):,} passages (batch_size={args.batch_size}) …")
    t1 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → cosine sim = dot product
    )
    print(f"[precompute] Encoded in {time.time()-t1:.1f}s. Shape: {embeddings.shape}")

    ids_arr = np.array(ids, dtype=object)
    np.savez_compressed(args.out, ids=ids_arr, embeddings=embeddings.astype(np.float32))
    print(f"[precompute] Saved to: {args.out}")
    print("[precompute] Done.")


if __name__ == "__main__":
    main()
