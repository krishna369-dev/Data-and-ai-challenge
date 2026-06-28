from typing import List, Dict, Any, Tuple
from src.index.embedding_index import EmbeddingIndex
from src.index.feature_index import FeatureIndex

class Stage1Retriever:
    def __init__(self, embedding_index: EmbeddingIndex, feature_index: FeatureIndex, jd_features: dict):
        self.embedding_index = embedding_index
        self.feature_index = feature_index
        self.jd_features = jd_features
        
    def retrieve(self, top_k_semantic: int = 10000, final_k: int = 2000) -> List[Tuple[str, float, float]]:
        """
        Two-stage retrieval:
        1. Fast semantic search (dot product) -> top_k_semantic
        2. Fast heuristic filter (required skills + exp) -> final_k
        Returns list of (candidate_id, semantic_score, retrieval_score)
        """
        # 1. Semantic Search using pre-computed JD embedding (fully offline)
        semantic_results = self.embedding_index.query(top_k=top_k_semantic)
        
        # 2. Heuristic Filter
        scored_candidates = []
        for cid, sem_score in semantic_results.items():
            cand = self.feature_index.get_candidate(cid)
            if not cand:
                continue
                
            raw = cand.get("raw", {})
            exp = raw.get("profile", {}).get("years_of_experience", 0)
            
            # Fast experience check
            min_exp = self.jd_features["experience"]["min_years"]
            max_exp = self.jd_features["experience"]["max_years"]
            exp_score = 1.0 if min_exp <= exp <= max_exp else max(0, 1.0 - 0.2 * min(abs(exp - min_exp), abs(exp - max_exp)))
            
            # Fast required skills check (boolean presence for speed)
            req_skills = self.jd_features["required_skills"]
            cand_skills_text = " ".join(cand.get("skills", {}).keys()) + " " + cand.get("career_text", "")
            
            matches = sum(1 for s in req_skills if s in cand_skills_text)
            skill_score = matches / max(len(req_skills), 1)
            
            # Combine for Stage 1 Retrieval Score
            retrieval_score = (sem_score * 0.4) + (skill_score * 0.4) + (exp_score * 0.2)
            scored_candidates.append((cid, sem_score, retrieval_score))
            
        # Sort by retrieval score descending
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        return scored_candidates[:final_k]
