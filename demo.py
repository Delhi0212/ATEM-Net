import os
import time
from agent import ATEMAgent

def run_simulation():
    print("=" * 80)
    print("       ATEM-Net: ADAPTIVE TEMPORAL-EPISODIC MEMORY NETWORKS SIMULATION")
    print("=" * 80)
    
    # Initialize the agent
    # We use a custom local directory for simulation database
    db_dir = "simulation_db"
    if os.path.exists(db_dir):
        # Clean up previous runs for clean demo
        import shutil
        try:
            shutil.rmtree(db_dir)
        except Exception:
            pass
            
    # Set parameters to demonstrate decay clearly in a short simulation timeline
    # Base decay rate: 0.003 (about 0.3% per hour)
    # Reinforcement weight: 1.5 (strong spacing effect)
    # Sentiment multiplier: 1.0 (strong emotional shielding)
    # Consolidation threshold: 0.15
    agent = ATEMAgent(
        db_dir=db_dir,
        decay_rate=0.003,
        reinforcement_weight=1.5,
        sentiment_multiplier=1.0,
        consolidation_threshold=0.15
    )
    
    # Base timestamp (Day 1, 09:00 AM)
    start_time = 1782780000.0  # Constant Unix timestamp for reproducibility
    
    def print_db_status(current_time):
        nodes = agent.db.get_all_nodes()
        print("\n--- Current Vector Database Contents ---")
        if not nodes:
            print("[Empty]")
            return
        for n in nodes:
            meta = n["metadata"]
            t_create = meta.get("t_create", current_time)
            reinforcement = meta.get("reinforcement_count", 0)
            sentiment = meta.get("sentiment_score", 0.0)
            delta_t_hours = (current_time - t_create) / 3600.0
            
            # calculate intrinsic retrievability
            r_score = agent.retriever.calculate_retrievability(
                similarity=1.0,
                delta_t_hours=delta_t_hours,
                reinforcement=reinforcement,
                sentiment=sentiment
            )
            
            status = "ACTIVE" if r_score >= 0.15 else "DECAYED (Eligible for Consolidation)"
            print(f"ID: {n['id'][:8]} | Content: '{n['document']}'")
            print(f"  Age: {delta_t_hours:.1f} hrs | Reinforcement (R): {reinforcement} | Sentiment (S): {sentiment:.2f} | R_score: {r_score:.3f} [{status}]")
        print("-" * 40)

    # -------------------------------------------------------------
    # DAY 1: Ingestion of initial memories
    # -------------------------------------------------------------
    print(f"\n>>> DAY 1 (Simulated Time: t = 0 hours)")
    current_time = start_time
    
    # 1. Neutral facts
    agent.ingest_memory("I live in Paris.", current_time)
    agent.ingest_memory("My favorite color is green.", current_time)
    agent.ingest_memory("I love reading sci-fi books.", current_time)
    
    # 2. Emotional memory (high sentiment score)
    # The sentiment analyzer will extract a high intensity score (VADER ~ 0.8)
    agent.ingest_memory("I had a massive fight with my brother today!", current_time)
    
    print_db_status(current_time)
    
    # -------------------------------------------------------------
    # DAY 5: Querying and Reinforcement (Spacing Effect)
    # -------------------------------------------------------------
    # 4 days later = 96 hours later
    current_time = start_time + (4 * 24 * 3600)
    print(f"\n>>> DAY 5 (Simulated Time: t = 96 hours)")
    
    # Ask about favorite color. This query will match "My favorite color is green."
    # The retrieval pipeline will fetch it, return it, and INCREMENT its reinforcement counter.
    print("\n[User Query]: 'What is my favorite color?'")
    response = agent.generate_response("What is my favorite color?", current_time)
    print(f"[Agent Response]: {response}")
    
    # Let's ask about family. This will retrieve the emotional brother memory.
    print("\n[User Query]: 'What do you know about my family?'")
    response = agent.generate_response("What do you know about my family?", current_time)
    print(f"[Agent Response]: {response}")
    
    print_db_status(current_time)
    
    # -------------------------------------------------------------
    # DAY 15: Introduction of Contradiction
    # -------------------------------------------------------------
    # 14 days later = 336 hours later
    current_time = start_time + (14 * 24 * 3600)
    print(f"\n>>> DAY 15 (Simulated Time: t = 336 hours)")
    
    # User informs the agent that they have moved.
    # This creates a direct contradiction with "I live in Paris."
    print("\n[User Interaction]: User updates their location.")
    agent.ingest_memory("I moved to New York today!", current_time)
    
    # Query location
    print("\n[User Query]: 'Where do I live?'")
    response = agent.generate_response("Where do I live?", current_time)
    print(f"[Agent Response]: {response}")
    
    print_db_status(current_time)
    
    # -------------------------------------------------------------
    # DAY 30: Memory Consolidation ("Dream Engine" execution)
    # -------------------------------------------------------------
    # 29 days later = 696 hours later
    current_time = start_time + (29 * 24 * 3600)
    print(f"\n>>> DAY 30 (Simulated Time: t = 696 hours)")
    print("Executing background consolidation routine ('Dream Engine')...")
    
    # Trigger Dream Engine consolidation
    consolidation_results = agent.dream_engine.consolidate(current_time)
    print("\n--- Consolidation Report ---")
    print(f"Status: {consolidation_results['status']}")
    print(f"Consolidated and Purged: {consolidation_results['consolidated_count']} episodic nodes")
    print(f"Faded Fragments Purged: {consolidation_results.get('purged_texts', [])}")
    print(f"Newly Generated Persona Traits: {consolidation_results.get('new_traits', [])}")
    print(f"Total Active Persona Traits: {len(agent.dream_engine.persona_data['traits'])}")
    print("----------------------------")
    
    # See what is left in vector DB (reinforced color memory and recent location memory should remain)
    # Decayed neutral memories like "I live in Paris" and "I love reading sci-fi books" should be consolidated and purged.
    # Emotional "fight with brother" might survive due to sentiment shielding, or consolidate depending on age and reinforcement.
    # Let's inspect the DB:
    print_db_status(current_time)
    
    # Let's display the current persona traits
    print("\n--- Consolidated Persona Profile (persona_profile.json) ---")
    print(agent.dream_engine.get_persona_prompt())
    print("-" * 55)
    
    # -------------------------------------------------------------
    # POST-CONSOLIDATION: Final Verification
    # -------------------------------------------------------------
    # Let's query about favorite color (reinforced and survived)
    print("\n[User Query]: 'What is my favorite color?'")
    response = agent.generate_response("What is my favorite color?", current_time)
    print(f"[Agent Response]: {response}")
    
    # Let's query about residency. "I live in Paris" has been purged, but "I moved to New York" is still active.
    # Additionally, the persona has consolidated location facts. Let's see if it correctly resolves.
    print("\n[User Query]: 'Where do I live now?'")
    response = agent.generate_response("Where do I live now?", current_time)
    print(f"[Agent Response]: {response}")

if __name__ == "__main__":
    run_simulation()
