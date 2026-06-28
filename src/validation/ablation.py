import logging

logger = logging.getLogger(__name__)

def run_ablation(pipeline_func):
    """
    Runs the pipeline multiple times, zeroing out one signal per run.
    Logs rank changes to provide empirical data for interviews.
    """
    logger.info("Starting Ablation Testing...")
    
    # 1. Baseline run
    logger.info("Running Baseline (All signals active)...")
    baseline_results = pipeline_func(ablate_signal=None)
    
    signals = ["semantic", "behavioral", "company_quality"]
    
    for signal in signals:
        logger.info(f"Running Ablation: Zeroing out '{signal}'...")
        ablation_results = pipeline_func(ablate_signal=signal)
        
        # Compare top 100 overlap
        baseline_ids = {c["candidate_id"] for c in baseline_results}
        ablation_ids = {c["candidate_id"] for c in ablation_results}
        
        overlap = len(baseline_ids.intersection(ablation_ids))
        logger.info(f"Overlap with baseline for '{signal}' ablation: {overlap}/100")
        
    logger.info("Ablation Testing Complete.")

if __name__ == "__main__":
    run_ablation(lambda ablate_signal: [])
