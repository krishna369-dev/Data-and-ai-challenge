import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Diagnostics:
    def run_report(self, candidates_processed: int, top_k_results: List[Dict[str, Any]], elapsed_time: float):
        """Prints a diagnostic report of the pipeline run."""
        logger.info("="*40)
        logger.info("PIPELINE DIAGNOSTICS REPORT")
        logger.info("="*40)
        logger.info(f"Candidates Processed: {candidates_processed}")
        logger.info(f"Top K Generated: {len(top_k_results)}")
        logger.info(f"Elapsed Time: {elapsed_time:.2f} seconds")
        
        if elapsed_time > 300:
            logger.warning("WARNING: Pipeline exceeded 5-minute compute constraint!")
        else:
            logger.info("Compute constraint met (< 5 mins).")
            
        # Basic sanity checks on top 100
        avg_score = sum(c["score"] for c in top_k_results) / max(1, len(top_k_results))
        logger.info(f"Average Top-K Score: {avg_score:.3f}")
        logger.info("="*40)
