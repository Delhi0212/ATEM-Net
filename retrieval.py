import numpy as np
import math

class ATEMRetriever:
    def __init__(self, db, emb_model, 
                 decay_rate: float = 0.01,         # lambda: base decay rate (per hour)
                 reinforcement_weight: float = 0.2, # k: how much reinforcement shields decay
                 sentiment_multiplier: float = 0.5, # alpha: sentiment importance
                 coarse_k: int = 20,
                 min_reinforcement_similarity: float = 0.45):
        self.db = db
        self.emb_model = emb_model
        self.decay_rate = decay_rate
        self.reinforcement_weight = reinforcement_weight
        self.sentiment_multiplier = sentiment_multiplier
        self.coarse_k = coarse_k
        self.min_reinforcement_similarity = min_reinforcement_similarity

    def calculate_retrievability(self, 
                                 similarity: float, 
                                 delta_t_hours: float, 
                                 reinforcement: int, 
                                 sentiment: float) -> float:
        """
        Modified Ebbinghaus Forgetting Curve formula:
        R_score = Cosine_Similarity * e^(-(lambda * Delta_t) / (k * R + 1)) * (1 + alpha * S)
        """
        # Delta_t must be non-negative
        delta_t_hours = max(0.0, delta_t_hours)
        
        # Calculate the forgetting exponential decay factor
        exponent_denominator = (self.reinforcement_weight * reinforcement) + 1.0
        decay_factor = math.exp(-(self.decay_rate * delta_t_hours) / exponent_denominator)
        
        # Calculate sentiment booster (emotional shield)
        sentiment_booster = 1.0 + (self.sentiment_multiplier * sentiment)
        
        # Final retrievability score
        r_score = similarity * decay_factor * sentiment_booster
        return r_score

    def retrieve(self, query_text: str, current_time: float, top_n: int = 5) -> list:
        """
        Two-stage hybrid retrieval pipeline:
        1. Fetch top 20 candidates via semantic similarity.
        2. Apply modified Ebbinghaus curve in-memory.
        3. Returns top 5 nodes and updates their reinforcement count asynchronously.
        """
        # Generate query embedding
        query_emb = self.emb_model.get_embedding(query_text)
        
        # Stage 1: Coarse semantic search (up to Top 20)
        candidates = self.db.query_semantic(query_emb, top_k=self.coarse_k)
        if not candidates:
            return []

        # Stage 2: In-memory mathematical re-ranking
        ranked_candidates = []
        for cand in candidates:
            meta = cand["metadata"]
            t_create = meta.get("t_create", current_time)
            reinforcement = meta.get("reinforcement_count", 0)
            sentiment = meta.get("sentiment_score", 0.0)
            
            # Delta t in hours (timestamps are in seconds, divide by 3600)
            delta_t_hours = (current_time - t_create) / 3600.0
            
            # Calculate R_score
            r_score = self.calculate_retrievability(
                similarity=cand["similarity"],
                delta_t_hours=delta_t_hours,
                reinforcement=reinforcement,
                sentiment=sentiment
            )
            
            ranked_cand = cand.copy()
            ranked_cand["r_score"] = r_score
            ranked_cand["delta_t_hours"] = delta_t_hours
            ranked_candidates.append(ranked_cand)
            
        # Sort candidates by R_score descending
        ranked_candidates.sort(key=lambda x: x["r_score"], reverse=True)
        
        # Filter distinct items to avoid duplication of exact text
        seen_docs = set()
        distinct_candidates = []
        for cand in ranked_candidates:
            doc_norm = cand["document"].strip().lower()
            if doc_norm not in seen_docs:
                seen_docs.add(doc_norm)
                distinct_candidates.append(cand)
                
        # Take the top N (usually 5)
        top_nodes = distinct_candidates[:top_n]
        
        # Asynchronously/Immediately update reinforcement and last accessed for retrieved nodes
        for node in top_nodes:
            node_id = node["id"]
            # Only reinforce if the semantic match is strong enough (avoid reinforcing filler matches)
            if node.get("similarity", 1.0) >= self.min_reinforcement_similarity:
                new_r = node["metadata"].get("reinforcement_count", 0) + 1
                self.db.update_metadata(node_id, {
                    "reinforcement_count": new_r,
                    "t_last_accessed": current_time
                })
            
        return top_nodes
