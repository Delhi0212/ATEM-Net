import os
import json
import uuid
import numpy as np

class MemoryDB:
    def __init__(self, persist_dir: str = "chroma_db", in_memory: bool = False):
        self.persist_dir = persist_dir
        self.in_memory = in_memory
        self.use_fallback = False
        self.collection = None
        
        # Try to initialize ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings
            
            if in_memory:
                self.client = chromadb.EphemeralClient()
            else:
                # Ensure the directory exists
                os.makedirs(persist_dir, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_dir)
            
            # Create or get collection. We'll handle embeddings ourselves.
            self.collection = self.client.get_or_create_collection(
                name="atem_memory_nodes"
            )
            print("[MemoryDB] ChromaDB initialized successfully.")
        except Exception as e:
            self.use_fallback = True
            self.db_file = os.path.join(persist_dir, "mock_db.json")
            os.makedirs(persist_dir, exist_ok=True)
            self._init_mock_db()
            print(f"[MemoryDB] ChromaDB failed to initialize: {e}. Using JSON fallback database at {self.db_file}")

    def _init_mock_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    self.mock_data = json.load(f)
            except Exception:
                self.mock_data = {}
        else:
            self.mock_data = {}
            self._save_mock_db()

    def _save_mock_db(self):
        try:
            with open(self.db_file, "w") as f:
                json.dump(self.mock_data, f, indent=2)
        except Exception as e:
            print(f"[MemoryDB] Failed to save JSON database: {e}")

    def add_memory(self, text: str, embedding: list, metadata: dict) -> str:
        """
        Inserts a memory node into the database. Returns the unique node ID.
        """
        node_id = str(uuid.uuid4())
        
        # Ensure metadata has defaults
        full_metadata = {
            "t_create": float(metadata.get("t_create", 0.0)),
            "t_last_accessed": float(metadata.get("t_last_accessed", metadata.get("t_create", 0.0))),
            "reinforcement_count": int(metadata.get("reinforcement_count", 0)),
            "sentiment_score": float(metadata.get("sentiment_score", 0.0)),
            "type": str(metadata.get("type", "episodic"))
        }
        
        if not self.use_fallback and self.collection is not None:
            try:
                self.collection.add(
                    ids=[node_id],
                    embeddings=[embedding],
                    metadatas=[full_metadata],
                    documents=[text]
                )
                return node_id
            except Exception as e:
                print(f"[MemoryDB] ChromaDB add failed: {e}. Saving to fallback.")
                
        # Fallback insertion
        self.mock_data[node_id] = {
            "id": node_id,
            "document": text,
            "embedding": embedding,
            "metadata": full_metadata
        }
        self._save_mock_db()
        return node_id

    def get_all_nodes(self) -> list:
        """
        Returns all nodes in the database.
        """
        if not self.use_fallback and self.collection is not None:
            try:
                results = self.collection.get()
                nodes = []
                # ChromaDB get returns a dict with lists under keys
                ids = results.get("ids", [])
                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])
                embeddings = results.get("embeddings", [])
                
                # Fetch embeddings if they are not returned (sometimes they aren't by get())
                if not embeddings or embeddings[0] is None:
                    # In some chroma versions, get() doesn't return embeddings by default unless requested
                    results = self.collection.get(include=["embeddings", "metadatas", "documents"])
                    ids = results.get("ids", [])
                    documents = results.get("documents", [])
                    metadatas = results.get("metadatas", [])
                    embeddings = results.get("embeddings", [])
                
                for idx in range(len(ids)):
                    emb = embeddings[idx] if idx < len(embeddings) else None
                    nodes.append({
                        "id": ids[idx],
                        "document": documents[idx] if idx < len(documents) else "",
                        "metadata": metadatas[idx] if idx < len(metadatas) else {},
                        "embedding": emb
                    })
                return nodes
            except Exception as e:
                print(f"[MemoryDB] ChromaDB get_all failed: {e}. Using fallback.")
                
        # Fallback fetch
        return list(self.mock_data.values())

    def update_metadata(self, node_id: str, metadata_updates: dict):
        """
        Updates the metadata of a specific node.
        """
        if not self.use_fallback and self.collection is not None:
            try:
                # Fetch existing metadata first to merge
                existing = self.collection.get(ids=[node_id])
                if existing and existing.get("metadatas"):
                    merged_metadata = existing["metadatas"][0].copy()
                    merged_metadata.update(metadata_updates)
                    
                    self.collection.update(
                        ids=[node_id],
                        metadatas=[merged_metadata]
                    )
                    return
            except Exception as e:
                print(f"[MemoryDB] ChromaDB update failed: {e}. Updating fallback.")

        # Fallback update
        if node_id in self.mock_data:
            self.mock_data[node_id]["metadata"].update(metadata_updates)
            self._save_mock_db()

    def delete_nodes(self, node_ids: list):
        """
        Deletes a list of nodes by ID.
        """
        if not node_ids:
            return
            
        if not self.use_fallback and self.collection is not None:
            try:
                self.collection.delete(ids=node_ids)
                return
            except Exception as e:
                print(f"[MemoryDB] ChromaDB delete failed: {e}. Deleting fallback.")

        # Fallback delete
        for nid in node_ids:
            if nid in self.mock_data:
                del self.mock_data[nid]
        self._save_mock_db()

    def query_semantic(self, query_embedding: list, top_k: int = 20) -> list:
        """
        Performs a semantic query (Stage 1) and returns raw similarity results.
        Returns a list of dicts with: id, document, metadata, embedding, similarity.
        """
        if not self.use_fallback and self.collection is not None:
            try:
                # ChromaDB query returns L2 distance by default (can be cosine depending on distance metric)
                # Let's request embeddings as well
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.collection.count()),
                    include=["documents", "metadatas", "embeddings", "distances"]
                )
                
                nodes = []
                ids = results.get("ids", [[]])[0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                embeddings = results.get("embeddings", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                for idx in range(len(ids)):
                    # Convert distance to similarity
                    # ChromaDB distance can be L2, cosine (cosine distance is 1 - similarity, ranging 0 to 2)
                    # Let's normalize it to a similarity score between 0 and 1
                    dist = distances[idx]
                    
                    # Assume cosine distance if it ranges from 0 to 2
                    # Similarity = 1 - distance/2 or 1 - distance if cosine distance in [0, 1]
                    # We will enforce: similarity = max(0.0, 1.0 - dist)
                    similarity = max(0.0, 1.0 - dist)
                    
                    nodes.append({
                        "id": ids[idx],
                        "document": documents[idx],
                        "metadata": metadatas[idx],
                        "embedding": embeddings[idx],
                        "similarity": float(similarity)
                    })
                return nodes
            except Exception as e:
                print(f"[MemoryDB] ChromaDB query failed: {e}. Querying fallback.")

        # Fallback query using numpy for cosine similarity
        results = []
        q_vec = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)
        
        for node_id, node in self.mock_data.items():
            node_emb = np.array(node["embedding"])
            node_norm = np.linalg.norm(node_emb)
            
            if q_norm > 0 and node_norm > 0:
                similarity = np.dot(q_vec, node_emb) / (q_norm * node_norm)
            else:
                similarity = 0.0
                
            results.append({
                "id": node["id"],
                "document": node["document"],
                "metadata": node["metadata"],
                "embedding": node["embedding"],
                "similarity": float(similarity)
            })
            
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
