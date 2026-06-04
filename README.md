# RAG Research Lab

A research-focused repository for studying and benchmarking Retrieval-Augmented Generation (RAG) systems.

This repository serves as the primary environment for conducting experiments on retrieval, retrieval quality, answer generation, evaluation, reliability, and performance trade-offs in modern RAG pipelines.

The goal is not to build a chatbot product, but to systematically investigate research questions related to retrieval-enhanced AI systems and produce reproducible experimental results.

---

# Research Theme

**Applied Generative AI Research for Enterprise Systems**

This repository focuses on:

* Retrieval-Augmented Generation (RAG)
* Information Retrieval
* Embedding Models
* Retrieval Evaluation
* Hallucination Reduction
* Context Optimization
* RAG Reliability
* Enterprise Knowledge Systems

---

# Research Objectives

The primary objective is to answer questions such as:

### Retrieval

* Which chunking strategy performs best?
* How does chunk size affect answer quality?
* Does chunk overlap improve retrieval performance?

### Embeddings

* Which embedding model provides the best retrieval quality?
* How do open-source embeddings compare to proprietary embeddings?

### Retrieval Quality

* Does reranking justify its latency cost?
* How much does retrieval quality influence generation quality?

### Reliability

* How can hallucinations be reduced in RAG systems?
* Which retrieval strategies improve factual grounding?

### Performance

* What are the latency and cost trade-offs of different RAG architectures?

---

# Repository Structure

```text
rag-research-lab/
│
├── configs/
│
├── datasets/
│
├── chunkers/
│   ├── base.py
│   ├── fixed_chunker.py
│   └── recursive_chunker.py
│
├── vectorstores/
│   └── faiss_store.py
│
├── retrievers/
│   ├── base.py
│   └── vector_retriever.py
│
├── pipelines/
│   └── basic_rag_pipeline.py
│
├── evaluators/
│
├── experiments/
│
├── notebooks/
│
├── results/
│
├── reports/
│
├── paper/
│
└── README.md
```

---

# Architecture

```text
Documents
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Vector Store
    │
    ▼
Retriever
    │
    ▼
Prompt Template
    │
    ▼
LLM
    │
    ▼
Generated Answer
    │
    ▼
Evaluation
    │
    ▼
Experiment Tracking
```

The repository uses reusable components from the **GenAI Core Framework**.

---

# Dependencies

This project depends on:

### GenAI Core Framework

* LLM Module
* Embedding Module
* Metrics Module
* Experiment Tracking Module
* Configuration Module
* Prompt Templates Module

---

# Baseline Pipeline

Version 1 implements a simple RAG pipeline.

### Components

#### Chunking

Supported:

* Fixed Chunking
* Recursive Chunking

#### Embeddings

Supported through the GenAI Core Framework.

Examples:

* BAAI/bge-small-en-v1.5
* BAAI/bge-base-en-v1.5
* intfloat/e5-base-v2
* OpenAI Embeddings

#### Vector Store

Current implementation:

* FAISS

#### Retrieval

Current implementation:

* Top-K Vector Retrieval

#### Generation

Supported through the GenAI Core Framework.

#### Evaluation

Current metrics:

* BLEU
* ROUGE
* BERTScore

Future versions will include RAG-specific metrics.


---

# Datasets

Initial datasets:

* HotpotQA
* Natural Questions
* SQuAD

Recommended starting dataset:

**HotpotQA**

---

# Experiment Tracking

All experiments should be recorded using the Experiment Tracking Module.

Tracked information includes:

* Experiment ID
* Parameters
* Metrics
* Metadata
* Timestamps

Example:

```python
tracker.log_param(
    "chunk_size",
    512
)

tracker.log_metric(
    "bertscore",
    0.91
)
```

This ensures reproducibility across experiments.

---

# Reports

All benchmark studies should produce a technical report.

Structure:

```text
reports/
│
├── chunk_size_benchmark.md
├── embedding_comparison.md
├── retriever_comparison.md
└── reranking_study.md
```

Reports should contain:

* Research Question
* Methodology
* Experimental Setup
* Results
* Discussion
* Conclusions

---

# Papers

Research studies can be expanded into publication-ready papers.

Example:

```text
paper/
│
└── chunk_size_study.md
```

Potential paper titles:

* An Empirical Evaluation of Chunk Size Selection in Retrieval-Augmented Generation Systems
* Comparing Embedding Models for Enterprise Retrieval Systems
* Evaluating Retrieval Strategies in Modern RAG Pipelines

---
