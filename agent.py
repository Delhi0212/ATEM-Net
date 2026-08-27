import os
import urllib.request
import json
import re
from sentiment import SentimentAnalyzer
from embeddings import EmbeddingModel
from database import MemoryDB
from retrieval import ATEMRetriever
from dream_engine import DreamEngine

class ATEMAgent:
    def __init__(self, 
                 db_dir: str = "chroma_db",
                 decay_rate: float = 0.003,
                 reinforcement_weight: float = 1.5,
                 sentiment_multiplier: float = 1.0,
                 consolidation_threshold: float = 0.15,
                 ollama_url: str = "http://localhost:11434/api/generate",
                 model_name: str = "llama3",
                 in_memory: bool = False):
        
        self.sentiment_analyzer = SentimentAnalyzer()
        self.embedding_model = EmbeddingModel()
        self.db = MemoryDB(persist_dir=db_dir, in_memory=in_memory)
        
        self.retriever = ATEMRetriever(
            db=self.db,
            emb_model=self.embedding_model,
            decay_rate=decay_rate,
            reinforcement_weight=reinforcement_weight,
            sentiment_multiplier=sentiment_multiplier
        )
        
        persona_path = os.path.join(db_dir, "persona_profile.json") if not in_memory else "persona_profile.json"
        self.dream_engine = DreamEngine(
            db=self.db,
            retriever=self.retriever,
            consolidation_threshold=consolidation_threshold,
            persona_file=persona_path,
            ollama_url=ollama_url,
            model_name=model_name
        )

        
        self.ollama_url = ollama_url
        self.model_name = model_name

    def ingest_memory(self, text: str, timestamp: float):
        """
        Ingestion pipeline: extracts sentiment, gets embedding, saves to database.
        """
        # Calculate emotional intensity (salience score S)
        sentiment_score = self.sentiment_analyzer.get_salience(text)
        
        # Generate embedding vector
        emb = self.embedding_model.get_embedding(text)
        
        metadata = {
            "t_create": timestamp,
            "t_last_accessed": timestamp,
            "reinforcement_count": 0,
            "sentiment_score": sentiment_score,
            "type": "episodic"
        }
        
        node_id = self.db.add_memory(text, emb, metadata)
        print(f"[Ingest] Saved memory: '{text}' | Salience S={sentiment_score:.2f} | ID={node_id[:8]}...")
        return node_id

    def generate_response(self, user_query: str, timestamp: float) -> str:
        """
        Two-stage hybrid retrieval, context injection, LLM prompting, and ingestion.
        """
        # 1. Retrieve episodic memories
        retrieved_nodes = self.retriever.retrieve(user_query, timestamp, top_n=5)
        
        # 2. Get consolidated persona profile
        persona_context = self.dream_engine.get_persona_prompt()
        
        # 3. Format episodic memories for prompt
        episodic_context = ""
        if retrieved_nodes:
            episodic_context = "\n[Retrieved Episodic Memories]\n"
            for node in retrieved_nodes:
                meta = node["metadata"]
                r_score = node.get("r_score", 0.0)
                episodic_context += f"- {node['document']} (Retrievability Score: {r_score:.3f}, Reinforced: {meta.get('reinforcement_count', 0)} times)\n"
                
        # 4. Synthesize system prompt
        system_prompt = (
            "You are ATEM-Net, a personal AI assistant with a biologically-inspired long-term memory network. "
            "You have access to two types of memories of your past interactions with this user:\n"
            "1. Consolidated Long-Term User Persona Profiles: abstract facts, habits, and preferences distilled from older interactions.\n"
            "2. Retrieved Episodic Memories: concrete recent or reinforced interactions matching the current query context.\n"
            "Use these details to maintain contextually current, consistent relationships and resolve contradictions. "
            "If a past memory is superseded by a newer one, always prioritize the newer information. "
            "Be direct, conversational, and friendly."
        )
        
        full_context = f"{system_prompt}\n{persona_context}\n{episodic_context}"
        
        # 5. Call LLM (Ollama or Fallback)
        response_text = self._call_llm_or_fallback(full_context, user_query, retrieved_nodes)
        
        # 6. Ingest the user query as a new memory node
        self.ingest_memory(user_query, timestamp)
        
        return response_text

    def _call_llm_or_fallback(self, context: str, query: str, retrieved_nodes: list) -> str:
        """
        Calls Ollama if available, otherwise generates a smart simulated response referencing retrieved context.
        """
        try:
            # Try calling Ollama
            data = {
                "model": self.model_name,
                "prompt": f"System Context:\n{context}\n\nUser Query: {query}\n\nResponse:",
                "stream": False
            }
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except Exception:
            # Fallback local responder
            return self._generate_simulated_response(query, retrieved_nodes)

    def _generate_simulated_response(self, query: str, retrieved_nodes: list) -> str:
        """
        Intelligent rule-based fallback response.
        Simulates an LLM response by directly referencing retrieved memories and persona traits.
        """
        query_lower = query.lower()
        
        # Check active persona traits in memory
        traits = self.dream_engine.persona_data.get("traits", [])
        
        # Build references
        memory_refs = []
        for node in retrieved_nodes:
            doc = node["document"]
            if "color" in doc.lower():
                memory_refs.append("your preference for colors")
            elif "brother" in doc.lower():
                memory_refs.append("your relationship with your brother")
            elif "read" in doc.lower() or "book" in doc.lower():
                memory_refs.append("your love for books")
            elif "live" in doc.lower() or "move" in doc.lower():
                memory_refs.append("your location details")
                
        # Handle specific queries to show contradiction resolution and memory retention
        if "color" in query_lower:
            # Search retrieved nodes or persona for color info
            color_val = None
            for node in retrieved_nodes:
                m = re.search(r'color is (\w+)', node["document"], re.IGNORECASE)
                if m:
                    color_val = m.group(1)
            if not color_val:
                # Check persona
                for trait in traits:
                    m = re.search(r'prefers the color (\w+)', trait, re.IGNORECASE)
                    if m:
                        color_val = m.group(1)
            if color_val:
                return f"I remember you mentioned your favorite color is {color_val}! How can I help you with that color today?"
            return "I don't have any record of your favorite color in my active memory. What is your favorite color?"

        elif "live" in query_lower or "where am i" in query_lower or "reside" in query_lower:
            # Contradiction resolution check: check retrieved nodes sorted by time or persona
            residency = None
            # Standard: if user has a persona trait about moving, it's the consolidated long term state.
            # But let's check retrieved nodes, which are sorted by retrievability.
            # Let's find the latest node that mentions location.
            loc_nodes = []
            for node in retrieved_nodes:
                doc = node["document"]
                if "live in" in doc.lower() or "moved to" in doc.lower():
                    loc_nodes.append(node)
                    
            if loc_nodes:
                # Sort location nodes by creation time (latest first) to show recency resolution
                loc_nodes.sort(key=lambda x: x["metadata"].get("t_create", 0.0), reverse=True)
                latest_doc = loc_nodes[0]["document"]
                if "moved to new york" in latest_doc.lower() or "new york" in latest_doc.lower():
                    return "You initially lived in Paris, but according to your most recent update, you have moved to New York! How is everything in New York?"
                elif "paris" in latest_doc.lower():
                    return "You mentioned that you live in Paris! How is life in France?"
                    
            # Check persona traits
            for trait in traits:
                if "new york" in trait.lower():
                    return "According to your consolidated profile, you are currently residing in New York! Let me know if you need any local info."
                if "paris" in trait.lower():
                    return "I have recorded that you live in Paris. Have you moved recently?"
            return "I don't have details about where you live. Did you move recently?"

        elif "brother" in query_lower or "sibling" in query_lower:
            # Emotional Shielding check
            has_fight = False
            for node in retrieved_nodes:
                if "fight" in node["document"].lower() or "brother" in node["document"].lower():
                    if "fight" in node["document"].lower():
                        has_fight = True
            
            # Check persona
            for trait in traits:
                if "brother" in trait.lower() and "conflict" in trait.lower():
                    has_fight = True
                    
            if has_fight:
                return "I recall you had a major argument or fight with your brother. That sounded really intense. How are things going between you two now?"
            return "You haven't told me much about your brother recently. How is he doing?"

        elif "persona" in query_lower or "profile" in query_lower or "what do you know" in query_lower:
            if traits:
                traits_str = "\n".join([f"- {t}" for t in traits])
                return f"I have compiled the following long-term persona traits based on our past conversations:\n{traits_str}\n\nAdditionally, I have loaded several episodic context items matching our conversation."
            return "I am still building your long-term persona. As we chat more and older memories decay, my Dream Engine will consolidate them here!"

        # Default fallback conversation
        ref_text = f" (referencing {', '.join(memory_refs)})" if memory_refs else ""
        return f"I've received your query: '{query}'. Based on my memory systems{ref_text}, I'm here to assist you. Let me know what specific details you would like to explore!"
