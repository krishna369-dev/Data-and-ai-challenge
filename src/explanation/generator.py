from typing import Dict, Any

class ExplanationGenerator:
    def generate(self, candidate_result: Dict[str, Any]) -> str:
        """
        Generates a 1-2 sentence explanation based on the score breakdown.
        """
        raw = candidate_result.get("raw", {})
        breakdown = candidate_result.get("breakdown", {})
        
        # Primary reason
        yoe = raw.get("profile", {}).get("years_of_experience", 0)
        
        sem_score = breakdown.get("retrieval", 0)
        conf = breakdown.get("confidence", 0)
        
        explanation = f"Strong semantic fit (score: {sem_score:.2f}) with {yoe} years of experience."
        
        # Modifier
        if conf > 0.8:
            explanation += " High confidence in skill extraction from explicit profile sections."
        elif conf < 0.5:
            explanation += " Scores adjusted due to lower confidence in parsed features."
            
        return explanation
