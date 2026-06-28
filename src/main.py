import argparse
import time
import csv
import sys
from typing import List, Dict, Any

from src.parser.candidate_parser import CandidateParser
from src.parser.jd_parser import JDParser
from src.index.embedding_index import EmbeddingIndex
from src.index.feature_index import FeatureIndex
from src.retrieval.stage1 import Stage1Retriever
from src.ranking.stage2 import Stage2Ranker
from src.explanation.generator import ExplanationGenerator
from src.validation.diagnostics import Diagnostics

def run_pipeline(candidates_file: str, embeddings_file: str, out_file: str):
    t0 = time.time()
    
    # 1. Parsers
    print("Parsing JD...")
    jd_parser = JDParser()
    jd_features = jd_parser.parse_file("job_description.txt")
    
    print("Parsing candidates...")
    cand_parser = CandidateParser()
    parsed_candidates = cand_parser.parse_file(candidates_file)
    cand_parser.log_variances()
    
    # 2. Indexes
    print("Building Feature Index...")
    feature_index = FeatureIndex()
    feature_index.build(parsed_candidates)
    
    print("Loading Embedding Index...")
    embedding_index = EmbeddingIndex(npz_path=embeddings_file)
    embedding_index.load()
    
    # 3. Stage 1 Retrieval
    print("Running Stage 1 Retrieval (High Recall)...")
    stage1 = Stage1Retriever(embedding_index, feature_index, jd_features)
    stage1_results = stage1.retrieve(top_k_semantic=5000, final_k=2000)
    
    # 4. Stage 2 Ranking
    print("Running Stage 2 Ranking (Precision)...")
    stage2 = Stage2Ranker(feature_index, jd_features)
    final_top_100 = stage2.rank(stage1_results, top_k=100)
    
    # 5. Explanations & Output
    print(f"Generating explanations and writing to {out_file}...")
    generator = ExplanationGenerator()
    
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_i, c in enumerate(final_top_100, start=1):
            reasoning = generator.generate(c)
            # Normalize score visually
            writer.writerow([c["candidate_id"], rank_i, f"{c['score']:.4f}", reasoning])
            
    # 6. Diagnostics
    t1 = time.time()
    diag = Diagnostics()
    diag.run_report(len(parsed_candidates), final_top_100, t1 - t0)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgraded 2-Stage Retrieval & Ranking")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--embeddings", default="./embeddings.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    run_pipeline(args.candidates, args.embeddings, args.out)
