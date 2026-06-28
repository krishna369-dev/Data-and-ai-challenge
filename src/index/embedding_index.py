import numpy as np
import logging
import os
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingIndex:
    def __init__(self, npz_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.npz_path = npz_path
        self.model_name = model_name
        self.ids = None
        self.embeddings = None
        self.model = None

    def load(self):
        """Loads precomputed candidate embeddings into memory."""
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(f"Embeddings file not found: {self.npz_path}")
        
        logger.info(f"Loading embeddings from {self.npz_path}...")
        data = np.load(self.npz_path, allow_pickle=True)
        self.ids = data["ids"]
        self.embeddings = data["embeddings"]
        logger.info(f"Loaded {len(self.ids)} embeddings.")
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded.")

    def query(self, text: str, top_k: int = 2000) -> dict:
        """
        Embeds the query text and computes dot product similarity (cosine sim if normalized) 
        against all candidate embeddings.
        Returns a dict of {candidate_id: similarity_score}.
        """
        if self.embeddings is None:
            self.load()
        
        self._load_model()
        
        query_emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Dot product (since both are normalized, this is cosine similarity)
        similarities = np.dot(self.embeddings, query_emb)
        
        # Get top K indices
        # use argpartition for O(N) top K
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            # sort the top K
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = {}
        for idx in top_indices:
            results[self.ids[idx]] = float(similarities[idx])
            
        return results
