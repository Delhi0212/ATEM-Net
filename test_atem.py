import unittest
import os
import shutil
import tempfile
from sentiment import SentimentAnalyzer
from embeddings import EmbeddingModel
from database import MemoryDB
from retrieval import ATEMRetriever
from dream_engine import DreamEngine

class TestATEMComponents(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for DB testing
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temp directory
        shutil.rmtree(self.test_dir)

    def test_sentiment_analyzer(self):
        analyzer = SentimentAnalyzer()
        # Test emotional range
        neutral_score = analyzer.get_salience("The clock is ticking.")
        emotional_score = analyzer.get_salience("I got proposed to today! I'm so happy!")
        
        self.assertTrue(0.0 <= neutral_score <= 1.0)
        self.assertTrue(0.0 <= emotional_score <= 1.0)
        self.assertTrue(emotional_score > neutral_score, f"Expected {emotional_score} > {neutral_score}")

    def test_embedding_model(self):
        emb_model = EmbeddingModel()
        emb1 = emb_model.get_embedding("hello world")
        emb2 = emb_model.get_embedding("hello world")
        
        self.assertEqual(len(emb1), 384)
        self.assertEqual(emb1, emb2) # Determinism check

    def test_database_operations(self):
        # Instantiate in temporary directory
        db = MemoryDB(persist_dir=os.path.join(self.test_dir, "db"), in_memory=True)
        emb_model = EmbeddingModel()
        
        text = "This is a test memory."
        emb = emb_model.get_embedding(text)
        metadata = {"t_create": 1000.0, "reinforcement_count": 0, "sentiment_score": 0.1, "type": "episodic"}
        
        # Test add
        node_id = db.add_memory(text, emb, metadata)
        self.assertIsNotNone(node_id)
        
        # Test retrieval
        nodes = db.get_all_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["document"], text)
        self.assertEqual(nodes[0]["metadata"]["reinforcement_count"], 0)
        
        # Test metadata update
        db.update_metadata(node_id, {"reinforcement_count": 5})
        nodes_updated = db.get_all_nodes()
        self.assertEqual(nodes_updated[0]["metadata"]["reinforcement_count"], 5)
        
        # Test delete
        db.delete_nodes([node_id])
        nodes_deleted = db.get_all_nodes()
        self.assertEqual(len(nodes_deleted), 0)

    def test_retriever_decay_logic(self):
        db = MemoryDB(persist_dir=os.path.join(self.test_dir, "db"), in_memory=True)
        emb_model = EmbeddingModel()
        
        # Add a node at time 0.0
        text1 = "Old memory."
        emb1 = emb_model.get_embedding(text1)
        db.add_memory(text1, emb1, {"t_create": 0.0, "reinforcement_count": 0, "sentiment_score": 0.0, "type": "episodic"})
        
        # Add a node at time 100000.0
        text2 = "Fresh memory."
        emb2 = emb_model.get_embedding(text2)
        db.add_memory(text2, emb2, {"t_create": 100000.0, "reinforcement_count": 0, "sentiment_score": 0.0, "type": "episodic"})
        
        retriever = ATEMRetriever(db, emb_model, decay_rate=0.01, reinforcement_weight=0.2, sentiment_multiplier=0.5, min_reinforcement_similarity=-2.0)
        
        # Query at time 100000.0
        # "Fresh memory" should have delta_t = 0, R_score should be equal to similarity
        # "Old memory" should have delta_t = 100000 seconds (~27.7 hours), R_score should decay
        results = retriever.retrieve("memory", current_time=100000.0, top_n=5)
        
        self.assertTrue(len(results) >= 2)
        
        # The fresh memory should be ranked first because the old memory decayed
        self.assertEqual(results[0]["document"], "Fresh memory.")
        self.assertEqual(results[1]["document"], "Old memory.")
        
        # Verify reinforcement count was incremented for retrieved items
        all_nodes = db.get_all_nodes()
        r_counts = {node["document"]: node["metadata"]["reinforcement_count"] for node in all_nodes}
        self.assertEqual(r_counts["Fresh memory."], 1)
        self.assertEqual(r_counts["Old memory."], 1)

    def test_dream_engine_consolidation(self):
        db = MemoryDB(persist_dir=os.path.join(self.test_dir, "db"), in_memory=True)
        emb_model = EmbeddingModel()
        
        # 1. Add fresh memory (will survive)
        db.add_memory("I live in New York.", emb_model.get_embedding("I live in New York."), 
                      {"t_create": 100000.0, "reinforcement_count": 0, "sentiment_score": 0.0, "type": "episodic"})
                      
        # 2. Add highly decayed memory (will consolidate and purge)
        db.add_memory("I live in Paris.", emb_model.get_embedding("I live in Paris."), 
                      {"t_create": 0.0, "reinforcement_count": 0, "sentiment_score": 0.0, "type": "episodic"})
                      
        retriever = ATEMRetriever(db, emb_model, decay_rate=0.05, reinforcement_weight=0.2, sentiment_multiplier=0.5)
        dream_engine = DreamEngine(db, retriever, consolidation_threshold=0.15, 
                                   persona_file=os.path.join(self.test_dir, "persona.json"))
        
        # Run consolidation at time 100000.0 (old memory is 27.7 hours old. e^(-0.05 * 27.7) = e^(-1.385) = 0.25.
        # Wait, if decay_rate is 0.1: e^(-0.1 * 27.7) = 0.062 < 0.15. So let's run at t=200000.0 (~55.5 hours old)
        # to ensure it decays below 0.15.
        current_time = 200000.0
        results = dream_engine.consolidate(current_time)
        
        self.assertEqual(results["status"], "success")
        self.assertEqual(results["consolidated_count"], 1)
        self.assertIn("User previously lived in or was associated with Paris", dream_engine.persona_data["traits"])
        
        # Verify that Paris is deleted from vector database, but New York remains
        remaining_nodes = db.get_all_nodes()
        remaining_docs = [node["document"] for node in remaining_nodes]
        self.assertIn("I live in New York.", remaining_docs)
        self.assertNotIn("I live in Paris.", remaining_docs)

if __name__ == "__main__":
    unittest.main()
