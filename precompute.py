"""
precompute.py — Offline embedding pre-computation for Redrob Hackathon.

Option A — Download pre-computed embeddings (FASTEST, ~30 seconds):
    python precompute.py --download --out ./embeddings.npz

Option B — Compute from scratch (~38 min on CPU):
    python precompute.py --candidates ./candidates.jsonl --out ./embeddings.npz

Generates a compressed NumPy array of shape (N, 384) using all-MiniLM-L6-v2.
Candidate IDs are stored in a parallel array so src/main.py can align them.

Requirements:
    pip install sentence-transformers numpy tqdm huggingface_hub
"""
import argparse
import json
import gzip
import os
import sys
import time

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace download config
# ─────────────────────────────────────────────────────────────────────────────
# After uploading embeddings.npz to HuggingFace, replace with your actual repo:
# e.g.  "krishna369-dev/redrob-embeddings"
HF_REPO_ID = "krishna369-dev/redrob-embeddings"
HF_FILENAME = "embeddings.npz"


def download_embeddings(out_path: str):
    """Download pre-computed embeddings from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[precompute] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"[precompute] Downloading embeddings from HuggingFace ({HF_REPO_ID}) ...")
    t0 = time.time()
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type="dataset",
        local_dir=os.path.dirname(os.path.abspath(out_path)) or ".",
    )
    # Move to the requested output path if different
    if os.path.abspath(local_path) != os.path.abspath(out_path):
        import shutil
        shutil.move(local_path, out_path)
    print(f"[precompute] Downloaded to {out_path} in {time.time()-t0:.1f}s")
    print("[precompute] Ready. You can now run: python -m src.main ...")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# JD Text — embedded once and saved so ranking is fully offline
# ─────────────────────────────────────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer Founding Team at Redrob AI (Series A). Location: Pune/Noida India Hybrid.
Experience: 5-9 years. Role owns the intelligence layer - ranking, retrieval, matching systems.
Must have: Production experience with embeddings-based retrieval systems: sentence-transformers,
OpenAI embeddings, BGE, E5. Production experience with vector databases or hybrid search:
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Strong Python.
Hands-on designing evaluation frameworks for ranking systems: NDCG, MRR, MAP, A/B testing.
Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT). Learning-to-rank models. HR-tech or
marketplace products. Distributed systems, large-scale inference. Open-source AI contributions.
Ideal: 6-8 years total, 4-5 years applied ML at product companies. Shipped end-to-end
ranking/search/recommendation system at scale. Pune/Noida or willing to relocate.
"""


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
    parser = argparse.ArgumentParser(description="Pre-compute or download candidate embeddings.")
    parser.add_argument("--candidates", default=None, help="Path to candidates.jsonl or .jsonl.gz (required unless --download)")
    parser.add_argument("--out", default="./embeddings.npz", help="Output .npz file path")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=512, help="Encoding batch size")
    parser.add_argument("--download", action="store_true", help="Download pre-computed embeddings from HuggingFace instead of computing")
    args = parser.parse_args()

    # Download mode — fast path for evaluators
    if args.download:
        download_embeddings(args.out)
        return

    if not args.candidates:
        parser.error("--candidates is required when not using --download")

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

    # Also pre-compute and save the JD embedding so ranking is 100% offline
    jd_out = os.path.splitext(args.out)[0] + "_jd.npy"
    print(f"[precompute] Encoding JD text and saving to: {jd_out}")
    jd_emb = model.encode([JD_TEXT], normalize_embeddings=True, convert_to_numpy=True)
    np.save(jd_out, jd_emb[0].astype(np.float32))
    print(f"[precompute] JD embedding saved. Shape: {jd_emb.shape}")
    print("[precompute] Done. Ranking step requires NO network access.")


if __name__ == "__main__":
    main()
