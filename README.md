
# 🔎 طابق: AI Research Similarity Finder

A bilingual (English/Arabic) web application that helps researchers find semantically similar scientific papers from **Semantic Scholar**, **arXiv**, and **PubMed** using powerful AI embeddings.

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-brightgreen)](https://your-app-url.com)

---

## 🧠 What It Does

**طابق** (Arabic for "match")
uses the [Sentence-Transformers](https://www.sbert.net/) model `all-MiniLM-L6-v2` to compare your research idea to abstracts and titles from top scientific databases.

It ranks results by **semantic similarity**, helping you:

* Discover related research faster
* Save time reviewing papers
* Get inspiration for literature reviews

---

## 🌍 Features

* 🌐 **Bilingual Interface**: Switch between English and Arabic
* 🧬 **Supports 3 Major Sources**:

  * Semantic Scholar
  * arXiv
  * PubMed
* 🎯 **Semantic Search** powered by SentenceTransformers
* 📆 **Year Range Filtering**
* 📊 **Adjustable Results Per Source**
* 📁 **CSV Export** for offline review
* 📱 **Responsive UI** and shareable via Twitter/X and LinkedIn

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ArwaAlanzi/research-similarity-finder.git
cd research-similarity-finder
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

* `streamlit`
* `sentence-transformers`
* `torch`
* `pandas`
* `requests`

### 3. Run the app

```bash
streamlit run app.py
```

