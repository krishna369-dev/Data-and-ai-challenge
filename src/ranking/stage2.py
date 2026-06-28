import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import List, Dict, Any, Tuple
from honeypots import is_honeypot
from src.index.feature_index import FeatureIndex

class Stage2Ranker:
    def __init__(self, feature_index: FeatureIndex, jd_features: dict):
        self.feature_index = feature_index
        self.jd_features = jd_features
        
    def rank(self, stage1_candidates: List[Tuple[str, float, float]], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Takes top 2000 from stage 1 and performs the precision three-score model.
        Returns the final top_k candidates with their scoring breakdowns.
        """
        final_scores = []
        
        for cid, sem_score, retrieval_stage1 in stage1_candidates:
            cand = self.feature_index.get_candidate(cid)
            if not cand:
                continue
                
            raw = cand.get("raw", {})
            
            # Honeypot check
            is_hp, _ = is_honeypot(raw)
            if is_hp:
                continue  # Hard filter
                
            # 1. Recalculate Retrieval & Confidence
            # For simplicity, we use a global confidence average across extracted features.
            # In a full implementation, this is feature-level.
            skills = cand.get("skills", {})
            if skills:
                avg_confidence = sum(conf for _, conf in skills.values()) / len(skills)
            else:
                avg_confidence = 0.6
                
            retrieval_score = retrieval_stage1 # From stage1
            confidence_adjusted = 0.6 + 0.4 * avg_confidence
            
            # 2. Evidence Score (Behavioral + Company Quality)
            evidence_score = self._compute_evidence(raw)
            
            # 3. Final calculation
            final = (retrieval_score * confidence_adjusted) + evidence_score
            
            final_scores.append({
                "candidate_id": cid,
                "score": final,
                "breakdown": {
                    "retrieval": retrieval_score,
                    "confidence": avg_confidence,
                    "evidence": evidence_score
                },
                "raw": raw
            })
            
        # Sort by final score
        final_scores.sort(key=lambda x: x["score"], reverse=True)
        return final_scores[:top_k]

    def _compute_evidence(self, raw: Dict[str, Any]) -> float:
        """Computes evidence score using behavioral signals."""
        sig = raw.get("redrob_signals", {})
        
        # Simple behavioral score
        response_rate = sig.get("recruiter_response_rate", 0.5)
        interview_rate = sig.get("interview_completion_rate", 0.5)
        active_recently = 1.0 if sig.get("last_active_date") else 0.0 # simplified
        
        evidence = (response_rate * 0.4) + (interview_rate * 0.4) + (active_recently * 0.2)
        return evidence * 0.5 # Scale it relative to retrieval
