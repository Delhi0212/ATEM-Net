# ATEM-Net: Adaptive Temporal-Episodic Memory Networks with Emotion-Aware Ebbinghaus Decay for Long-Term Conversational AI

## Overview

**ATEM-Net** (Adaptive Temporal-Episodic Memory Network) is a biologically-inspired, self-evolving memory architecture designed to address the fundamental challenges of long-term conversational AI. Traditional Retrieval-Augmented Generation (RAG) models store unbounded interaction histories indefinitely, leading to context window overflow, factual hallucinations, and high latency. ATEM-Net dynamically ranks, decays, and consolidates memories based on cognitive principles of human memory retention and forgetting.

By integrating a **modified Ebbinghaus forgetting curve**, **VADER emotional salience scoring**, **spaced-repetition reinforcement**, and an offline **Dream Engine consolidation module**, ATEM-Net maintains long-term contextual consistency and resolves temporal contradictions without requiring model fine-tuning or retraining.

---

## Problem Statement

Conversational AI agents deployed across extended time horizons suffer from three primary failure modes:
1. **Context Window Overflow:** Unbounded memory history accumulation exceeds the LLM's processable input context, triggering the *"Lost in the Middle"* attention failure mode.
2. **Temporal Contradiction:** Pure semantic similarity search retrieves outdated facts alongside current user states (e.g., an old residential address vs. a new one), causing factual hallucinations.
3. **Knowledge Drift:** Static retrieve-all mechanisms fail to adapt as user preferences, habits, and life circumstances evolve over weeks and months.

---

## Objectives

- Design a lightweight, model-agnostic conversational memory architecture based on biological forgetting curves.
- Implement an emotional salience filter to shield significant life events from premature decay.
- Develop an autonomous episodic-to-semantic consolidation mechanism (Dream Engine) to distill decayed memories into persistent persona traits.
- Reduce LLM prompt token overhead while improving factual retrieval precision, recall, and contradiction resolution.

---

## Key Features

- **Emotion-Aware Salience Scoring ($S_i$):** Employs VADER sentiment analysis to calculate an absolute compound emotional intensity score ($S_i \in [0, 1]$), protecting emotionally charged events from rapid decay.
- **Modified Ebbinghaus Decay Ranking:** Multi-factor re-ranking formula incorporating semantic cosine similarity, exponential temporal decay, recall count, and emotional salience.
- **Spaced-Recall Reinforcement:** Accessing a memory node increases its recall count ($R$), slowing future exponential decay (mimicking the cognitive spacing effect).
- **Autonomous Dream Engine:** Periodically scans for decayed memory vectors ($R_{\text{score}} < 0.15$), distills them into persistent user persona traits (`persona_profile.json`), and purges the raw vectors from ChromaDB.
- **Automatic Contradiction Handling:** Outdated contradictory memories decay naturally and get processed by the Dream Engine into historical persona traits, leaving active memories unconflicted.
- **Token Clutter Reduction:** Achieves a **63.5% reduction in prompt token usage** (from 520 tokens in Standard RAG down to 190 tokens in ATEM-Net).
- **Robust Fallback Mechanisms:** Includes deterministic hash-based embedding fallbacks, rule-based sentiment routines, and local rule-based response generators if optional external dependencies or LLMs are unavailable.

---

## Technology Stack

- **Programming Language:** Python
- **Vector Database:** ChromaDB
- **Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`, 384-dimensional dense vectors)
- **Sentiment & Salience:** VADER (`vaderSentiment` / `nltk.sentiment.vader`)
- **Numerical & Data Processing:** NumPy
- **Visualization & Plotting:** Matplotlib
- **LLM Integration:** Ollama (`llama3` local REST API client with fallback responder)
- **Interactive Interface & Testing:** Jupyter Notebook, `unittest` framework

---

## Architecture & Data Flow

```text
User Query
   │
   ├──► [Sentiment Analysis (VADER)] ──► Emotional Salience (S)
   ├──► [Embedding (SentenceTransformers)] ──► Dense Vector (x_i)
   │
   ▼
[ChromaDB Vector Store] ── Fetch Top-20 Candidates ──► [ATEMRetriever]
                                                              │
                                                              ▼
                                               [Ebbinghaus Re-Ranker]
                                                              │
                                       ┌──────────────────────┴──────────────────────┐
                                       ▼                                             ▼
                             Top-5 Active Nodes                            Decayed Nodes (R_score < 0.15)
                                       │                                             │
                                       ▼                                             ▼
                              [LLM Prompt Context]                             [Dream Engine]
                                       │                                             │
                                       ▼                                             ▼
                              Generated Response                           Distill Persona Profile &
                                                                             Purge Raw Vectors
```

### Modified Ebbinghaus Retrievability Formula

$$R_{\text{score}} = \text{CosineSim}(\mathbf{q}, \mathbf{x}_i) \times \exp\left(-\frac{\lambda \Delta t}{k R + 1}\right) \times (1 + \alpha S)$$

**Parameter Definitions:**
- $\text{CosineSim}(\mathbf{q}, \mathbf{x}_i)$: Semantic cosine similarity between query vector $\mathbf{q}$ and memory vector $\mathbf{x}_i$ ($\mathbf{q} \cdot \mathbf{x}_i$ for unit-normalized vectors).
- $\lambda$: Base exponential decay rate per hour ($\lambda = 0.003$ per hour by default).
- $\Delta t$: Elapsed time in hours since memory creation ($\Delta t = (t_{\text{current}} - t_{\text{create}}) / 3600$).
- $k$: Spaced-repetition reinforcement weight ($k = 1.5$ by default).
- $R$: Integer recall count tracking how many times the memory node has been retrieved.
- $\alpha$: Emotional salience multiplier ($\alpha = 1.0$ by default).
- $S$: Absolute compound emotional intensity score ($S \in [0.0, 1.0]$).

---

## Project Structure

```text
ATEM-Net/
├── README.md                   # Complete project documentation
├── .gitignore                  # Git exclusion rules
├── requirements.txt            # Python dependencies
├── agent.py                    # ATEMAgent main orchestrator class
├── database.py                 # ChromaDB vector store wrapper with JSON fallback
├── dream_engine.py             # Consolidation engine & vector purging module
├── embeddings.py               # SentenceTransformer embedding engine & fallback
├── retrieval.py                # Two-stage re-ranking engine & retrievability formula
├── sentiment.py                # VADER sentiment intensity salience scorer
├── demo.py                     # Interactive CLI demonstration script
├── evaluate.py                 # 30-day timeline benchmark evaluation
├── verify.py                   # Automated system verification suite
├── test_atem.py                # Unit test suite (unittest)
├── ATEM-Net.ipynb              # Main interactive research Jupyter Notebook
├── paper.tex                   # Complete IEEE conference LaTeX manuscript
├── IEEE_Paper_ATEM_Net.html    # Publication HTML version of research paper
├── ATEM_Net_IEEE_Paper.pdf     # Compiled 1.22 MB IEEE conference paper
├── forgetting_curves.png       # Fig 2: Retrievability decay curves plot
├── evaluation_metrics.png      # Fig 3: SOTA comparison bar chart
├── token_comparison.png        # Fig 4: Context clutter reduction plot
└── dataset_retrievability.png  # Fig 5: 30-day dataset retrievability plot
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ATEM-Net.git
cd ATEM-Net
```

### 2. Create and Activate a Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Ollama Configuration

Ollama is an optional local LLM runtime. ATEM-Net connects by default to:
`http://localhost:11434`

To pull and run the Llama 3 model locally:
```bash
ollama pull llama3
```

*Note: Ollama is optional. If Ollama is not running, ATEM-Net automatically falls back to an intelligent, rule-based simulated responder that references retrieved context items.*

---

## How to Run

### Interactive CLI Demo
```bash
python demo.py
```

### Run 30-Day Timeline Evaluation Benchmark
```bash
python evaluate.py
```

### Run System Verification Suite
```bash
python verify.py
```

### Run Unit Tests
```bash
python -m unittest test_atem.py
```

### Launch Interactive Jupyter Notebook
```bash
jupyter notebook ATEM-Net.ipynb
```

---

## Evaluation & Benchmark Results

The performance of ATEM-Net was validated using a purpose-built synthetic longitudinal personal-memory dataset ($N = 20$ conversational records, 6 semantic categories, 720-hour timeline, explicit contradiction pairs, strict scene-level chronological partitioning):

| Metric / Baseline | Standard RAG (Top-20) | MemGPT | A-MEM | **ATEM-Net (Ours)** |
|---|---|---|---|---|
| **Precision** | 54.2% | 82.0% | 81.0% | **88.0%** |
| **Recall** | 98.0% | 76.5% | 89.5% | **94.0%** |
| **F1-Score** | 69.8% | 79.1% | 85.0% | **91.0%** |
| **Prompt Token Overhead** | 520 tokens | 340 tokens | 280 tokens | **190 tokens (63.5% reduction)** |
| **Contradiction Resolution** | 0% (Confusion) | 60.0% | 80.0% | **100% (Automatic Purge)** |

*Note: These benchmark results were produced on the project's synthetic longitudinal dataset designed for reproducible algorithmic evaluation.*

---

## Research Paper

The repository includes the complete research paper in multiple formats:
- `paper.tex`: IEEE conference double-column LaTeX source code.
- `IEEE_Paper_ATEM_Net.html`: Web-based publication document with embedded figures.
- `ATEM_Net_IEEE_Paper.pdf`: Publication-ready compiled PDF (1.22 MB).

---

## Dataset

The evaluation suite utilizes a synthetic longitudinal personal-memory dataset consisting of 20 conversational records across six categories: Location, Preference, Hobby, Emotion, Education, and Habit. The dataset is chronologically partitioned across a 30-day timeline to test temporal decay and contradiction resolution under controlled conditions.

---

## Future Improvements

- **Multimodal Memory Integration:** Extend vector indexing to support images, audio, and attached documents.
- **Edge-AI Optimization:** Quantize embedding models and optimize re-ranking algorithms for resource-constrained edge hardware.
- **Federated Continual Learning:** Enable privacy-preserving memory consolidation across decentralized user devices.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
