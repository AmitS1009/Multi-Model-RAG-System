# 🧠 Multi-Modal RAG System 🚀
*Unlocking insights from Text, Tables, and Images in complex documents.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-00ADD8?style=for-the-badge&logo=Meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![CLIP](https://img.shields.io/badge/Model-CLIP-orange?style=for-the-badge)](https://github.com/openai/CLIP)

## 🌟 Overview

Welcome to the **Multi-Modal Retrieval-Augmented Generation (RAG)** system! This project redefines document interaction by going beyond traditional text-only search. It is engineered to ingest, understand, and retrieve information from **complex PDF documents** containing dense text, structured tables, and rich imagery.

Powered by a **Unified Multi-Modal Embedding Space (CLIP)** and **Hybrid Search**, this system allows users to ask natural language questions and receive context-aware answers backed by precise citations and **visual evidence** (retrieved charts, graphs, and images).

## ✨ Key Features

*   **📄 Multi-Modal Ingestion Pipeline**: 
    *   **Text**: Semantic chunking with `RecursiveCharacterTextSplitter`.
    *   **Tables**: Structured extraction for data-heavy queries.
    *   **Images**: Intelligent extraction with OCR (Tesseract) fallback for scanned content.
*   **🖼️ Unified Vector Space**: Leverages `sentence-transformers/clip-ViT-B-32` to map text and images into a shared semantic space, enabling cross-modal retrieval.
*   **🔍 Advanced Hybrid Search**: 
    *   **Dense Retrieval**: FAISS index for capturing semantic meaning.
    *   **Sparse Retrieval**: BM25 (via `rank_bm25`) for keyword precision.
    *   **Fusion**: Weighted scoring to combine the best of both worlds.
*   **🤖 Context-Aware QA**: Generates accurate, grounded answers using `google/flan-t5-base` (local inference) with strict citation tracking.
*   **📊 Interactive Dashboard**: A Streamlit-based UI featuring:
    *   Real-time Chat with Image Rendering.
    *   One-click Document Summarization.
    *   Evaluation Metrics Visualization.

## 🛠️ Architecture

```mermaid
graph TD
    subgraph Ingestion
    A[PDF Document] --> B(Document Processor)
    B --> C{Chunking Strategy}
    C -->|Text/Tables| D[Text Chunks]
    C -->|Images| E[Image Chunks + OCR]
    end
    
    subgraph Indexing
    D --> F[CLIP Embeddings]
    E --> F
    F --> G[(FAISS Dense Index)]
    D --> H[(BM25 Sparse Index)]
    end
    
    subgraph Retrieval & Generation
    I[User Query] --> J{Hybrid Search}
    J -->|Semantic| G
    J -->|Keyword| H
    G & H --> K[Ranked Results]
    K --> L[Context Construction]
    L --> M[LLM (Flan-T5)]
    M --> N[Answer + Citations + Images]
    end
```

## 🚀 Getting Started

### Prerequisites
*   **Python 3.8+**
*   **Tesseract-OCR** (Optional, for enhanced image text extraction)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/multi-modal-rag.git
    cd multi-modal-rag
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data**
    Place your PDF documents in the `data/` directory.

## 🏃‍♂️ Usage Guide

### 1. Data Ingestion 📥
Process your documents to extract and structure the content.
```bash
python process_document.py
```

### 2. Build the Index 🏗️
Generate multi-modal embeddings and build the Hybrid FAISS + BM25 index.
```bash
python create_embeddings.py
```

### 3. Launch the App 🚀
Start the interactive Streamlit interface.
```bash
streamlit run app.py
```
*Navigate to `http://localhost:8501` in your browser.*

## 📊 Evaluation & Benchmarking

The system includes a comprehensive evaluation suite to measure performance.

*   **Run Benchmarks**:
    ```bash
    python evaluation.py
    ```
*   **View Results**: Open the **Evaluation Dashboard** tab in the Streamlit app to visualize:
    *   ⏱️ Average Retrieval Latency
    *   ⚡ Generation Time
    *   🖼️ Image Retrieval Rates

## 🔮 Future Roadmap

*   **Cross-Modal Reranking**: Implement a dedicated reranker (e.g., ColBERT) for higher precision.
*   **VLM Integration**: Upgrade to Vision-Language Models (like GPT-4o or LLaVA) for deep visual reasoning.
*   **Cloud Deployment**: Dockerize the application for scalable deployment on AWS/Azure.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

---
*Built by AMIT KUSHWAHA*
