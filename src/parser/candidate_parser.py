import json
import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandidateParser:
    def __init__(self):
        self.schema_variances = []
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parses a jsonl file of candidates."""
        candidates = []
        import gzip
        open_func = gzip.open if filepath.endswith('.gz') else open
        
        with open_func(filepath, 'rt', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                    self._check_schema(candidate, line_no)
                    parsed_cand = self._extract_features(candidate)
                    candidates.append(parsed_cand)
                except json.JSONDecodeError:
                    logger.error(f"JSON decode error at line {line_no}")
        return candidates

    def _check_schema(self, candidate: Dict[str, Any], line_no: int):
        """Logs schema variances without failing."""
        required_keys = ["candidate_id", "profile", "career_history", "education", "skills", "redrob_signals"]
        missing = [k for k in required_keys if k not in candidate]
        if missing:
            self.schema_variances.append(f"Line {line_no}: Candidate {candidate.get('candidate_id', 'UNKNOWN')} missing keys: {missing}")

    def _extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts features and assigns confidence scores.
        Confidence:
        - 1.0 if skill is explicitly in the 'skills' section
        - 0.8 if skill is mentioned in 'career_history' descriptions
        - 0.6 if skill is inferred from 'profile.summary' or 'headline'
        """
        parsed = {
            "candidate_id": candidate["candidate_id"],
            "raw": candidate,
            "skills": {} # skill_name -> (score, confidence)
        }
        
        # 1. Explicit skills (Confidence 1.0)
        for skill in candidate.get("skills", []):
            skill_name = skill.get("name", "").lower()
            if skill_name:
                proficiency = skill.get("proficiency", "beginner")
                prof_score = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}.get(proficiency, 1)
                parsed["skills"][skill_name] = (prof_score, 1.0)
                
        # 2. Implicit skills from Career History (Confidence 0.8)
        # For a full implementation, we'd use a regex or string matching against the JD skills here.
        # This will be populated further during feature extraction or Stage 1 matching.
        parsed["career_text"] = " ".join([c.get("description", "") for c in candidate.get("career_history", [])]).lower()
        parsed["summary_text"] = (candidate.get("profile", {}).get("summary", "") + " " + candidate.get("profile", {}).get("headline", "")).lower()
        
        return parsed

    def log_variances(self):
        if self.schema_variances:
            logger.warning(f"Found {len(self.schema_variances)} schema variances.")
            for v in self.schema_variances[:10]:
                logger.warning(v)
            if len(self.schema_variances) > 10:
                logger.warning("... (truncated)")
