import math
from typing import List, Dict, Any

class FeatureIndex:
    def __init__(self):
        self.candidates_by_id = {}
        self.doc_frequencies = {} # skill -> number of candidates possessing it
        self.idf_scores = {}
        self.total_docs = 0

    def build(self, parsed_candidates: List[Dict[str, Any]]):
        """Builds the feature index and computes IDF scores for skills."""
        self.total_docs = len(parsed_candidates)
        
        for cand in parsed_candidates:
            cid = cand["candidate_id"]
            self.candidates_by_id[cid] = cand
            
            # Record document frequencies for skills
            for skill in cand.get("skills", {}).keys():
                self.doc_frequencies[skill] = self.doc_frequencies.get(skill, 0) + 1
                
        # Compute IDF
        for skill, df in self.doc_frequencies.items():
            # Standard IDF formula: log(N / df)
            self.idf_scores[skill] = math.log(self.total_docs / (df + 1)) + 1.0
            
    def get_candidate(self, candidate_id: str) -> Dict[str, Any]:
        return self.candidates_by_id.get(candidate_id, {})
        
    def get_idf(self, skill: str) -> float:
        """Returns the IDF score for a skill. If unseen, assumes very rare (high IDF)."""
        if skill in self.idf_scores:
            return self.idf_scores[skill]
        return math.log(self.total_docs / 1.0) + 1.0
