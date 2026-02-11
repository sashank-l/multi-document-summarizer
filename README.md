# 📰 Multi-Document Summarization & Analysis System

An end-to-end **AI-powered multi-document summarization framework** that clusters related documents, generates coherent summaries using transformer-based models, and provides rich visual analytics through an interactive Gradio interface.

This system is designed to handle **heterogeneous document formats**, **long-form text**, and **unstructured data**, making it suitable for real-world knowledge analysis, research summarization, and internal knowledge assistants.

---
[![](https://img.youtube.com/vi/HpJiodVeqg0/0.jpg)](https://youtu.be/HpJiodVeqg0)

## 🚀 Key Features

### 🔹 Multi-Format Document Support
Upload and process:
- `.txt`
- `.pdf`
- `.csv`
- `.xlsx`
- `.json`
- `.xml`
- `.pptx`

All documents are automatically cleaned and normalized before analysis.

---

### 🔹 Semantic Embedding & Clustering
- Uses **Sentence-BERT (`all-mpnet-base-v2`)** for dense semantic embeddings
- Applies **HDBSCAN** for:
  - Automatic cluster discovery
  - Noise document detection
  - No need to predefine number of clusters

---

### 🔹 Adaptive Summarization Strategy
The system dynamically selects the summarization model based on input length:

| Text Length | Model Used |
|------------|-----------|
| Short (<1024 tokens) | BART (CNN/DailyMail) |
| Medium (<4096 tokens) | Longformer |
| Very Long | Chunked Longformer with overlap |

This ensures **high-quality summaries even for long documents**.

---

### 🔹 Visual Analytics Dashboard
The interface provides the following visual insights:

- **TF-IDF Bar Plot** – Highlights important terms
- **Hierarchical Dendrogram** – Visualizes document similarity
- **t-SNE Projection** – 2D semantic embedding visualization
- **Word Cloud** – Global keyword distribution

All visualizations are generated dynamically based on uploaded documents.

---

### 🔹 Interactive Gradio Interface
- Upload multiple documents at once
- One-click summarization
- Clean bullet-style output
- Visual analytics rendered inline

---

## 🧠 System Architecture

```
Input Documents
      ↓
Text Extraction & Cleaning
      ↓
Sentence-BERT Embeddings
      ↓
HDBSCAN Clustering
      ↓
Cluster-wise Text Aggregation
      ↓
Adaptive Summarization
      ↓
Summaries + Visual Analytics
```

---

## 📂 Project Structure

```
.
├── app.py          # Main application (Gradio UI + summarization pipeline)
├── eval.py         # Evaluation module (ROUGE + clustering quality)
├── README.md       # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone <repo-url>
cd multi-document-summarizer
```

### 2️⃣ Install Dependencies
```bash
pip install torch gradio sentence-transformers transformers
pip install hdbscan nltk scikit-learn matplotlib seaborn plotly
pip install datasets rouge-score pypdf pandas python-pptx wordcloud
```

### 3️⃣ Download NLTK Resources
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

---

## ▶️ Running the Application

```bash
python app.py
```

The Gradio interface will launch locally and can be accessed via the browser.

---

## 📊 Evaluation Module

The project includes an **automated evaluation pipeline** in `eval.py`.

### Metrics Used
- **ROUGE-1 / ROUGE-L** for summarization quality
- **Silhouette Score** for clustering quality
- Random similarity sanity checks using cosine similarity

### Datasets (Auto-Detected)
- Multi-News
- CNN/DailyMail
- XSum
- GovReport
- BookSum
- PubMed-QA

### Run Evaluation
```bash
python eval.py
```

---

## 🧪 Use Cases

- Multi-document news summarization
- Research paper aggregation
- Internal company knowledge assistants (RAG-ready)
- Legal / policy document analysis
- Academic literature review automation

---

## 🔮 Future Enhancements

- RAG-based query answering over clusters
- Fine-tuned summarization models
- Export summaries as PDF / DOCX
- Topic labeling per cluster
- Multi-language support

---

## 📌 Highlights

- No hard-coded cluster count
- Handles long documents gracefully
- Clean modular design
- Production-ready evaluation
- Internship & resume friendly

---

## 📄 License

This project is released for **educational and research purposes**.
