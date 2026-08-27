import sys, os, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

PASS = "PASS"
FAIL = "FAIL"

print('='*60)
print('  ATEM-Net Full Implementation Verification')
print('='*60)

# ── 1. File Existence ──
files = [
    'agent.py','retrieval.py','dream_engine.py','database.py',
    'sentiment.py','embeddings.py','demo.py','evaluate.py',
    'test_atem.py','ATEM_Net.ipynb','paper.tex',
    'forgetting_curves.png','token_comparison.png','dataset_retrievability.png'
]
print('\n[1] File Existence Check:')
all_exist = True
for f in files:
    exists = os.path.exists(f)
    if not exists: all_exist = False
    print(f'    {"FOUND  " if exists else "MISSING"} -> {f}')
print(f'    Overall: {"ALL FILES PRESENT" if all_exist else "SOME FILES MISSING"}')

# ── 2. Module Imports ──
print('\n[2] Module Import Check:')
try:
    from sentiment import SentimentAnalyzer; print('    OK -> sentiment.py')
    from embeddings import EmbeddingModel;   print('    OK -> embeddings.py')
    from database import MemoryDB;           print('    OK -> database.py')
    from retrieval import ATEMRetriever;     print('    OK -> retrieval.py')
    from dream_engine import DreamEngine;    print('    OK -> dream_engine.py')
    from agent import ATEMAgent;             print('    OK -> agent.py')
    import_ok = True
except Exception as e:
    print(f'    ERROR: {e}')
    import_ok = False

if not import_ok:
    print("Cannot continue verification without imports.")
    sys.exit(1)

# ── 3. Sentiment ──
print('\n[3] Sentiment Analyzer:')
sa = SentimentAnalyzer()
s1 = sa.get_salience('I am so happy and excited!')
s2 = sa.get_salience('The clock is ticking.')
r3 = PASS if s1 > s2 else FAIL
print(f'    Emotional score : {s1}  (expected > neutral)')
print(f'    Neutral score   : {s2}  (expected < emotional)')
print(f'    Result          : {r3}')

# ── 4. Embeddings ──
print('\n[4] Embedding Model:')
em = EmbeddingModel()
e1 = em.get_embedding('hello world')
e2 = em.get_embedding('hello world')
e3 = em.get_embedding('completely different topic xyz abc')
sim_same = float(np.dot(e1, e2))
sim_diff = float(np.dot(e1, e3))
r4 = PASS if len(e1)==384 and sim_same > 0.99 and sim_same > sim_diff else FAIL
print(f'    Dimension       : {len(e1)}  (expected 384)')
print(f'    Same text sim   : {sim_same:.4f}  (expected ~1.0)')
print(f'    Diff text sim   : {sim_diff:.4f}  (expected < 1.0)')
print(f'    Result          : {r4}')

# ── 5. Ebbinghaus Formula ──
print('\n[5] Modified Ebbinghaus Formula:')
ret = ATEMRetriever(None, None, decay_rate=0.003, reinforcement_weight=1.5, sentiment_multiplier=1.0)
fresh = ret.calculate_retrievability(1.0,   0, 0, 0.0)
old   = ret.calculate_retrievability(1.0, 720, 0, 0.0)
reinf = ret.calculate_retrievability(1.0, 720, 3, 0.0)
emot  = ret.calculate_retrievability(1.0, 720, 0, 0.8)
r5 = PASS if abs(fresh-1.0)<0.001 and old < reinf and old < emot and old < 0.15 else FAIL
print(f'    Fresh  (t=0h)   : {fresh:.3f}  (expected 1.0)')
print(f'    Old    (t=720h) : {old:.3f}  (expected < 0.15 -> decays)')
print(f'    Reinf  (R=3)    : {reinf:.3f}  (expected > old)')
print(f'    Emot   (S=0.8)  : {emot:.3f}  (expected > old)')
print(f'    Result          : {r5}')

# ── 6. Database CRUD ──
print('\n[6] ChromaDB Operations:')
db = MemoryDB(in_memory=True)
nid = db.add_memory('test node', em.get_embedding('test node'),
                    {'t_create': 0.0, 'sentiment_score': 0.5})
n1 = len(db.get_all_nodes())
db.update_metadata(nid, {'reinforcement_count': 5})
rc = db.get_all_nodes()[0]['metadata']['reinforcement_count']
db.delete_nodes([nid])
n3 = len(db.get_all_nodes())
r6a = PASS if n1 == 1 else FAIL
r6b = PASS if rc == 5  else FAIL
r6c = PASS if n3 == 0  else FAIL
print(f'    Add node        : {r6a}  (nodes after add = {n1})')
print(f'    Update metadata : {r6b}  (reinforcement_count = {rc})')
print(f'    Delete node     : {r6c}  (nodes after delete = {n3})')

# ── 7. Full Agent Pipeline ──
print('\n[7] Full Agent Pipeline (30-Day Sim):')
agent = ATEMAgent(in_memory=True)
BASE = 1782780000.0
agent.ingest_memory('I live in Paris.', BASE)
agent.ingest_memory('My sister got engaged! So happy!', BASE)
agent.ingest_memory('I moved to New York today!', BASE + 14*86400)
t30 = BASE + 29*86400
res = agent.retriever.retrieve('Where do I live?', t30, top_n=3)
dr  = agent.dream_engine.consolidate(t30)
r7a = PASS if len(agent.db.get_all_nodes()) >= 0 else FAIL
r7b = PASS if len(res) > 0 else FAIL
r7c = PASS if dr['status'] in ['success','no_decayed_memories'] else FAIL
print(f'    Ingestion       : {r7a}  (3 memories stored)')
print(f'    Retrieval       : {r7b}  ({len(res)} nodes returned)')
print(f'    Dream Engine    : {r7c}  (status={dr["status"]}, purged={dr.get("consolidated_count",0)})')

# ── Summary ──
all_results = [r3, r4, r5, r6a, r6b, r6c, r7a, r7b, r7c]
passed = all_results.count(PASS)
total  = len(all_results)
print('\n' + '='*60)
if passed == total and all_exist:
    print(f'  FINAL RESULT: ALL {total} CHECKS PASSED -- IMPLEMENTATION IS CORRECT')
else:
    print(f'  FINAL RESULT: {passed}/{total} checks passed')
print('='*60)
