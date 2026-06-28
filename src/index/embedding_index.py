import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingIndex:
    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        # JD embedding is saved alongside the npz as embeddings_jd.npy
        self.jd_emb_path = os.path.splitext(npz_path)[0] + "_jd.npy"
        self.ids = None
        self.embeddings = None
        self.jd_embedding = None

    def load(self):
        """Loads precomputed candidate embeddings and JD embedding into memory.
        No network access required — everything is loaded from local .npz / .npy files.
        """
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(
                f"Embeddings file not found: {self.npz_path}\n"
                "Run: python precompute.py --download --out ./embeddings.npz"
            )
        
        logger.info(f"Loading candidate embeddings from {self.npz_path}...")
        data = np.load(self.npz_path, allow_pickle=True)
        self.ids = data["ids"]
        self.embeddings = data["embeddings"]
        logger.info(f"Loaded {len(self.ids):,} candidate embeddings. Shape: {self.embeddings.shape}")

        # Load pre-computed JD embedding (100% offline, no model needed)
        if os.path.exists(self.jd_emb_path):
            self.jd_embedding = np.load(self.jd_emb_path)
            logger.info(f"Loaded JD embedding from {self.jd_emb_path} (offline mode).")
        else:
            logger.warning(
                f"JD embedding file not found at {self.jd_emb_path}. "
                "Re-run precompute.py to generate it."
            )
            self.jd_embedding = None

    def query(self, top_k: int = 2000) -> dict:
        """
        Computes dot product similarity between the precomputed JD embedding
        and all candidate embeddings. No model loading, no network access.
        Returns dict of {candidate_id: similarity_score}.
        """
        if self.embeddings is None:
            self.load()
        
        if self.jd_embedding is None:
            raise RuntimeError(
                "JD embedding not loaded. Re-run precompute.py to generate embeddings_jd.npy"
            )

        # Pure numpy dot product — fast, offline, deterministic
        similarities = np.dot(self.embeddings, self.jd_embedding)
        
        # Get top K indices efficiently
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        return {self.ids[idx]: float(similarities[idx]) for idx in top_indices}
