import numpy as np
import hashlib

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.use_fallback = False
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            # Try to load the sentence transformer model
            self.model = SentenceTransformer(self.model_name)
            self.dimension = 384
        except Exception as e:
            # Fallback to deterministic pseudo-random embeddings
            self.use_fallback = True
            self.dimension = 384
            print(f"[EmbeddingModel] SentenceTransformer failed to load: {e}. Using deterministic fallback.")

    def get_embedding(self, text: str) -> list:
        """
        Generates a 384-dimensional embedding vector for the given text.
        Vector is normalized to unit length.
        """
        if not self.use_fallback and self.model is not None:
            try:
                emb = self.model.encode(text)
                # Convert to numpy array and normalize
                emb = np.array(emb)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb.tolist()
            except Exception as e:
                print(f"[EmbeddingModel] SentenceTransformer encode failed: {e}. Falling back.")
                return self._fallback_embedding(text)
        else:
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> list:
        """
        Deterministic, hash-based fallback embedding generation.
        Generates a 384-dimensional unit vector from the md5 hash of the text,
        producing the same vector for the same text.
        """
        dimension = self.dimension
        # Create a sequence of sub-hashes to seed our generator
        vec = []
        for i in range(dimension):
            # Salt with the index to get different values for each dimension
            h = hashlib.md5(f"{text}_{i}".encode('utf-8')).hexdigest()
            # Convert hash string to a float between -1.0 and 1.0
            val = int(h[:8], 16) / 4294967295.0 * 2.0 - 1.0
            vec.append(val)
        
        # Add basic term matching boost: if words overlap, alignment should be higher.
        # This makes mock embeddings behave somewhat semantically.
        words = set(text.lower().split())
        for w in words:
            # Seed based on word hash to add a small signal component
            wh = int(hashlib.md5(w.encode('utf-8')).hexdigest()[:8], 16) % dimension
            vec[wh] += 1.0
            
        # Convert to numpy array and normalize to unit sphere
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

# Quick test when executed
if __name__ == "__main__":
    emb_model = EmbeddingModel()
    emb1 = emb_model.get_embedding("hello world")
    emb2 = emb_model.get_embedding("hello world")
    emb3 = emb_model.get_embedding("different sentence")
    
    # Verify same text produces same embedding
    print("Match same text:", np.allclose(emb1, emb2))
    
    # Calculate similarity (dot product of normalized vectors)
    sim_same = np.dot(emb1, emb2)
    sim_diff = np.dot(emb1, emb3)
    print("Sim same:", sim_same)
    print("Sim diff:", sim_diff)
