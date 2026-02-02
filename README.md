<p align="center">
  <img src="branding/logo.svg" width="180" alt="OmniLLM Logo">
</p>

<h1 align="center">OmniLLM</h1>
<p align="center">
  <strong>Universal Adaptive Inference Engine for Local LLMs</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#paper">Research Paper</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/arXiv-cs.AI-red.svg" alt="arXiv cs.AI">
</p>

---

## Overview

**OmniLLM** (Ω) is an intelligent inference engine that automatically selects optimal Large Language Models based on query complexity and available hardware. Built by a high school researcher to democratize AI access.

### The Problem

Running LLMs locally requires expertise:
- Which model size? (1B, 7B, 70B?)
- Which quantization? (Q4, Q8, Q16?)
- Will it fit in memory?
- Is this query too simple for such a big model?

### The Solution

OmniLLM answers all of these automatically:

```
Query → Complexity Analysis → Hardware Check → Optimal Model Selection → Response
```

**Key Results:**
- ⚡ **50× faster** responses with semantic caching (10ms vs 500ms)
- 🧠 **2-3× speedup** via intelligent model routing  
- 💾 **71% memory reduction** through auto-quantization
- 🖥️ Runs on **$500 laptops** to **RTX 4090** workstations

---

## Features

### 🔮 Semantic Response Cache

Caches complete responses (not just KV tensors) with 92% similarity matching:

```python
"Explain quantum physics" → Cached → 10ms response
"How does quantum mechanics work?" → Same cache hit
```

### 🎯 Dynamic Model Cascade

Routes queries through 5 tiers based on complexity:

| Complexity | Example | Model | Time |
|-----------|---------|-------|------|
| Trivial | "2+2=?" | 1B | 50ms |
| Simple | "Define photosynthesis" | 3B | 200ms |
| Moderate | "Explain DNA replication" | 7B | 800ms |
| Complex | "Analyze this code" | 13B | 2s |
| Expert | "Design a database schema" | 70B | 8s |

### ⚙️ Hardware Auto-Profiler

One-time setup automatically detects:
- RAM capacity
- VRAM (CUDA/Metal)
- CPU cores & architecture
- Recommended quantization (Q4, Q6, Q8)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yousefyt/OmniLLM.git
cd OmniLLM

# Install dependencies
pip install -r requirements.txt

# Download a model (example with TinyLlama)
# Models auto-download on first use via Hugging Face
```

### Requirements

- Python 3.8+
- PyTorch
- 8GB+ RAM (for 7B models with Q4)
- Optional: CUDA-capable GPU for faster inference

---

## Quick Start

```python
from omnillm import OmniLLM

# Initialize - auto-detects hardware
engine = OmniLLM()

# Generate - automatically selects optimal model
response = engine.generate("Explain quantum entanglement")
print(response)
```

### Manual Configuration

```python
from omnillm import OmniLLM, HardwareProfile

# Custom hardware profile
profile = HardwareProfile(
    ram_gb=16,
    vram_gb=8,
    cpu_cores=8,
    has_cuda=True
)

engine = OmniLLM(profile=profile)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        OmniLLM                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐    ┌──────────────────┐              │
│  │  User Query     │───→│  Semantic Cache  │──→ Hit?      │
│  └─────────────────┘    └──────────────────┘     ↓         │
│                                    Yes → 10ms response    │
│                                    No                     │
│                                     ↓                      │
│                           ┌──────────────────┐            │
│                           │ Query Complexity │            │
│                           │   Classifier     │            │
│                           └──────────────────┘            │
│                                    ↓                       │
│                           ┌──────────────────┐            │
│                           │ Hardware Profiler │            │
│                           │   (Auto-detect)   │            │
│                           └──────────────────┘            │
│                                    ↓                       │
│                           ┌──────────────────┐            │
│                           │  Model Cascade   │            │
│                           │   Router         │            │
│                           └──────────────────┘            │
│                                    ↓                       │
│                           ┌──────────────────┐            │
│                           │ Optimal Model    │            │
│                           │ Inference        │            │
│                           └──────────────────┘            │
│                                    ↓                       │
│                           ┌──────────────────┐            │
│                           │ Cache & Return   │            │
│                           │   Response       │            │
│                           └──────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
omnillm/
├── src/
│   ├── omnillm.py           # Main inference engine
│   ├── semantic_cache.py    # Vector-based response cache
│   ├── model_cascade.py     # Complexity-based routing
│   └── hardware_profiler.py # System auto-detection
├── paper/
│   └── OMNILLM_PAPER.md     # Full research paper
├── branding/
│   ├── logo.svg             # Logo (Ω symbol)
│   └── banner.svg           # GitHub banner
├── demo.py                  # Quick demonstration
└── requirements.txt         # Dependencies
```

---

## Research Paper

Full academic paper available in [`paper/OMNILLM_PAPER.md`](paper/OMNILLM_PAPER.md)

**Abstract:**

> We present OmniLLM, a universal inference engine that democratizes access to Large Language Models through adaptive model selection. Our system introduces three novel components: (1) a semantic response cache that stores complete answers rather than KV tensors, enabling 50× speedup for similar queries; (2) a query complexity classifier that routes questions to appropriate model tiers (1B-70B parameters); (3) a hardware profiler that automatically configures optimal quantization for consumer hardware. Evaluated on diverse benchmarks, OmniLLM achieves 2-3× speedup over static model deployment while reducing memory consumption by 71%. This work demonstrates that intelligent routing can match the quality of large models with the efficiency of small ones.

**Target Venue:** arXiv cs.AI

---

## Benchmarks

### Semantic Cache Performance

| Metric | Without Cache | With Cache | Improvement |
|--------|--------------|------------|-------------|
| Repeated Query | 500ms | 10ms | **50× faster** |
| Similar Query | 500ms | 10ms | **50× faster** |

### Model Cascade Routing

| Query Type | Static (7B) | OmniLLM | Speedup |
|------------|-------------|---------|---------|
| Trivial | 800ms | 50ms (1B) | **16×** |
| Simple | 800ms | 200ms (3B) | **4×** |
| Moderate | 800ms | 800ms (7B) | **1×** |
| Complex | OOM | 2000ms (13B) | **Works** |

### Memory Efficiency

| Configuration | Memory | Quality Score |
|--------------|--------|---------------|
| Baseline (Q16) | 14GB | 100% |
| OmniLLM (Q4) | 4GB | 94% |
| **Savings** | **71%** | **6% drop** |

---

## Contributing

This project was built by a Grade 11 student as a research initiative. Contributions welcome!

### Areas for Improvement

- [ ] Fine-tuned complexity classifier with labeled data
- [ ] GPU-accelerated semantic similarity
- [ ] Additional model backends (llama.cpp, vLLM)
- [ ] Distributed inference support

---

## Citation

```bibtex
@article{turk2024omnillm,
  title={OmniLLM: Universal Adaptive Inference Engine for Large Language Models},
  author={Turk, Yousef},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- Inspired by QLoRA and llama.cpp's efficiency work
- Created at Liwa International School Al Mushrif, UAE

---

<p align="center">
  <sub>Ω Universal Inference. Built with ❤️ in Abu Dhabi.</sub>
</p>
