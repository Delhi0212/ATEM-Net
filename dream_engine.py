import os
import json
import urllib.request
import re

class DreamEngine:
    def __init__(self, db, retriever, 
                 consolidation_threshold: float = 0.15,
                 persona_file: str = "persona_profile.json",
                 ollama_url: str = "http://localhost:11434/api/generate",
                 model_name: str = "llama3"):
        self.db = db
        self.retriever = retriever
        self.consolidation_threshold = consolidation_threshold
        self.persona_file = persona_file
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.persona_data = {"traits": [], "last_updated": 0.0}
        self._load_persona()

    def _load_persona(self):
        if os.path.exists(self.persona_file):
            try:
                with open(self.persona_file, "r") as f:
                    self.persona_data = json.load(f)
            except Exception:
                pass

    def _save_persona(self):
        try:
            with open(self.persona_file, "w") as f:
                json.dump(self.persona_data, f, indent=2)
        except Exception as e:
            print(f"[DreamEngine] Failed to save persona profile: {e}")

    def consolidate(self, current_time: float) -> dict:
        """
        Scans database for episodic nodes that fall below the consolidation threshold,
        distills them into generalized user persona traits via LLM (or fallback),
        updates the persona profile, and deletes the consolidated nodes.
        """
        all_nodes = self.db.get_all_nodes()
        decayed_nodes = []
        decayed_ids = []
        
        for node in all_nodes:
            meta = node["metadata"]
            # Only consolidate episodic nodes
            if meta.get("type", "episodic") != "episodic":
                continue
                
            t_create = meta.get("t_create", current_time)
            reinforcement = meta.get("reinforcement_count", 0)
            sentiment = meta.get("sentiment_score", 0.0)
            
            # Calculate intrinsic retrievability (using similarity = 1.0)
            delta_t_hours = (current_time - t_create) / 3600.0
            r_score = self.retriever.calculate_retrievability(
                similarity=1.0,
                delta_t_hours=delta_t_hours,
                reinforcement=reinforcement,
                sentiment=sentiment
            )
            
            if r_score < self.consolidation_threshold:
                decayed_nodes.append(node)
                decayed_ids.append(node["id"])
                
        if not decayed_nodes:
            return {"status": "no_decayed_memories", "consolidated_count": 0}
            
        print(f"[DreamEngine] Found {len(decayed_nodes)} decayed memory nodes below threshold ({self.consolidation_threshold}). Consolidating...")
        
        # Extract text of decayed memories
        texts = [node["document"] for node in decayed_nodes]
        
        # Distill traits using local LLM or fallback
        new_traits = self._distill_memories_llm_or_fallback(texts)
        
        # Merge new traits with existing ones
        self.persona_data["traits"] = self._merge_traits(self.persona_data["traits"], new_traits)
        self.persona_data["last_updated"] = current_time
        self._save_persona()
        
        # Purge consolidated episodic nodes from vector database
        self.db.delete_nodes(decayed_ids)
        print(f"[DreamEngine] Purged {len(decayed_ids)} low-resolution episodic vectors from database.")
        
        return {
            "status": "success",
            "consolidated_count": len(decayed_ids),
            "new_traits": new_traits,
            "total_traits": len(self.persona_data["traits"]),
            "purged_texts": texts
        }

    def _distill_memories_llm_or_fallback(self, texts: list) -> list:
        """
        Uses Ollama if running, otherwise falls back to a deterministic rule-based summarizer.
        """
        prompt = (
            "Analyze these faded episodic memories and extract high-level, generalized user persona traits, "
            "habits, preferences, or relationships. Formulate them as short, direct statements about the user "
            "(e.g., 'User loves reading sci-fi books', 'User is close to their brother'). "
            "Return ONLY a JSON list of strings. Do not write explanation or markdown formatting, just the list.\n\n"
            f"Memories:\n" + "\n".join([f"- {t}" for t in texts])
        )
        
        # Attempt to use Ollama
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                response_text = res_data.get("response", "").strip()
                
                # Parse JSON array of strings
                # Llama 3 format or standard JSON array
                match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', response_text, re.DOTALL)
                if match:
                    traits = json.loads(match.group(0))
                else:
                    # Direct attempt
                    traits = json.loads(response_text)
                    
                if isinstance(traits, list):
                    return [str(t) for t in traits if t]
        except Exception as e:
            # Fallback if Ollama is not available or errors
            pass
            
        return self._fallback_distillation(texts)

    def _fallback_distillation(self, texts: list) -> list:
        """
        Rule-based NLP fallback for memory distillation.
        Extracts semantic insights when local LLM is offline.
        """
        traits = []
        # Predefined mapping for rule-based synthesis
        keywords = {
            r"\b(read|book|novel|sci-fi|fiction|fantasy|auth|write)\b": "User has an interest in reading or literature",
            r"\b(color|paint|draw|art|blue|green|red|yellow|purple)\b": "User has aesthetic color or art preferences",
            r"\b(brother|sister|mother|father|mom|dad|family|parents|sibling)\b": "User maintains active family relationships",
            r"\b(work|office|job|boss|project|colleague|meeting|code|coding)\b": "User is focused on professional projects or software development",
            r"\b(eat|food|dinner|lunch|breakfast|pizza|burger|restaurant|cook|cooking)\b": "User has specific dietary habits or culinary interests",
            r"\b(gym|run|fitness|exercise|workout|health|fit|sport)\b": "User maintains a health and fitness routine",
            r"\b(dog|cat|pet|animal|vet)\b": "User is a pet owner or animal lover",
            r"\b(python|javascript|rust|c\+\+|java|developer|program)\b": "User is a programmer or software engineer",
            r"\b(happy|sad|depressed|angry|excited|fight|disagree|agree)\b": "User's emotional state has been influenced by daily events",
            r"\b(travel|trip|flight|holiday|vacation|hotel)\b": "User enjoys traveling or planning trips",
            r"\b(live|move|moved|resident|reside|location|paris|york|london|tokyo|city)\b": "User has residential location details"
        }
        
        for text in texts:
            text_lower = text.lower()
            matched = False
            for pattern, trait in keywords.items():
                if re.search(pattern, text_lower):
                    # Try to customize the trait based on details
                    custom_trait = trait
                    if "brother" in text_lower and "fight" in text_lower:
                        custom_trait = "User has occasional personal conflicts with their brother"
                    elif "green" in text_lower and "color" in text_lower:
                        custom_trait = "User prefers the color green"
                    elif "sci-fi" in text_lower and "read" in text_lower:
                        custom_trait = "User enjoys reading science fiction literature"
                    elif "new york" in text_lower and "move" in text_lower:
                        custom_trait = "User moved to or resides in New York"
                    elif "paris" in text_lower and "live" in text_lower:
                        custom_trait = "User previously lived in or was associated with Paris"
                        
                    traits.append(custom_trait)
                    matched = True
                    
            if not matched:
                # Direct synthesis fallback for generic sentence structure
                clean_text = text.replace("I love", "User loves").replace("I like", "User likes").replace("I have", "User has").replace("I am", "User is").replace("my", "their")
                traits.append(f"User noted: '{clean_text}'")
                
        # Return unique list
        return list(set(traits))

    def _merge_traits(self, existing_traits: list, new_traits: list) -> list:
        """
        Merges existing traits with new traits, resolving contradictions or duplicates.
        """
        # Simplistic resolution: if a new trait is highly overlapping, override it.
        # Otherwise append.
        merged = existing_traits.copy()
        for nt in new_traits:
            # Check for contradiction (e.g. New York vs Paris residency)
            conflict_detected = False
            for idx, et in enumerate(merged):
                # If both mention residing/living somewhere but different places
                if ("live" in et.lower() or "reside" in et.lower() or "moved to" in et.lower()) and \
                   ("live" in nt.lower() or "reside" in nt.lower() or "moved to" in nt.lower()):
                    # Extract city names or check if they differ
                    cities = ["new york", "paris", "london", "tokyo", "san francisco"]
                    et_city = next((c for c in cities if c in et.lower()), None)
                    nt_city = next((c for c in cities if c in nt.lower()), None)
                    if et_city and nt_city and et_city != nt_city:
                        # Overwrite older trait with newer trait
                        print(f"[DreamEngine] Resolving contradiction: '{et}' -> '{nt}' (Updating residency)")
                        merged[idx] = nt
                        conflict_detected = True
                        break
                        
                # Overlap check (avoid duplicate traits)
                if self._similarity_score(et, nt) > 0.7:
                    # Prefer the newer formulation
                    merged[idx] = nt
                    conflict_detected = True
                    break
                    
            if not conflict_detected:
                merged.append(nt)
        return merged

    def _similarity_score(self, s1: str, s2: str) -> float:
        """
        Simple Jaccard similarity of words as a proxy for trait overlap.
        """
        w1 = set(re.findall(r'\b\w+\b', s1.lower()))
        w2 = set(re.findall(r'\b\w+\b', s2.lower()))
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / len(w1.union(w2))

    def get_persona_prompt(self) -> str:
        """
        Returns the persona profile formatted as a system prompt addition.
        """
        if not self.persona_data["traits"]:
            return ""
        return (
            "\n[Consolidated Long-Term User Persona Profiles]\n"
            + "\n".join([f"- {t}" for t in self.persona_data["traits"]])
        )
