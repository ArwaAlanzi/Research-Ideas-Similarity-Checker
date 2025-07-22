import streamlit as st
from sentence_transformers import SentenceTransformer, util
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import re
import urllib.parse
import pandas as pd
from io import StringIO

# Load model once and cache it
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("🔄 Loading model..."):
        return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Language toggle
lang = st.selectbox("Language / اللغة", ["English", "العربية"])
is_arabic = lang == "العربية"

# Text labels
title = "🔎 طابق: AI Research Similarity Finder" if not is_arabic else "🔎 طابق: أداة الذكاء الاصطناعي لمطابقة الأبحاث"
subtitle = "Find Related Scientific Papers Easily" if not is_arabic else "ابحث عن أبحاث علمية مشابهة بسهولة"
placeholder_text = "E.g. Deep learning for medical imaging" if not is_arabic else "مثال: التعلم العميق في التصوير الطبي"
input_label = "Enter your research idea or keywords..." if not is_arabic else "أدخل فكرتك البحثية أو الكلمات المفتاحية..."
papers_found_text = "🔍 Showing papers from" if not is_arabic else "🔍 عرض الأبحاث من"
download_label = "⬇️ Download results as CSV" if not is_arabic else "⬇️ تحميل النتائج كملف CSV"
share_prompt = "If you find this tool helpful, please share it with your network!" if not is_arabic else "إذا وجدت هذه الأداة مفيدة، شاركها مع الآخرين!"
footer_text = "Made with ❤️ From Saudi Arabia" if not is_arabic else "مصنوع بحب ❤️ في السعودية"

# UI styling
st.markdown(f"""
    <style>
    .main-title {{
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        color: #1e1e1e;
        margin-bottom: 10px;
    }}
    .subtle {{
        font-size: 18px;
        text-align: center;
        color: #666666;
        margin-bottom: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='main-title'>{title}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtle'>{subtitle}</div>", unsafe_allow_html=True)

# Input section
user_input = st.text_input(input_label, placeholder=placeholder_text)

col1, col2, col3 = st.columns(3)
with col1:
    start_year = st.number_input("Start Year", value=2000, step=1, min_value=1900, max_value=datetime.now().year)
with col2:
    end_year = st.number_input("End Year", value=datetime.now().year, step=1, min_value=1900, max_value=datetime.now().year)
with col3:
    num_results = st.slider("Papers per Source", min_value=5, max_value=100, step=5, value=20)

def highlight_keywords(text, keywords):
    for word in keywords:
        text = re.sub(rf"(?i)\b({re.escape(word)})\b", r"<span style='background-color: #ffff66'><b>\1</b></span>", text)
    return text

@st.cache_data(show_spinner=False)
def search_semantic_scholar(query, limit=100):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,abstract,url,year"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        st.error(f"Semantic Scholar error: {e}")
        return []

@st.cache_data(show_spinner=False)
def search_arxiv(query, max_results=20):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            abstract = entry.find('atom:summary', ns).text.strip()
            url = entry.find('atom:id', ns).text.strip()
            published = entry.find('atom:published', ns).text
            year = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").year
            papers.append({"title": title, "abstract": abstract, "url": url, "year": year})
        return papers
    except Exception as e:
        st.error(f"arXiv error: {e}")
        return []

@st.cache_data(show_spinner=False)
def search_pubmed(query, max_results=20):
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi"
        fetch_url = f"{base_url}efetch.fcgi"
        search_params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_resp.raise_for_status()
        ids = search_resp.json()["esearchresult"]["idlist"]
        if not ids:
            return []
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=10)
        fetch_resp.raise_for_status()
        root = ET.fromstring(fetch_resp.content)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            title_elem = article.find(".//ArticleTitle")
            abstract_elem = article.find(".//Abstract/AbstractText")
            date_elem = article.find(".//PubDate/Year")
            year = int(date_elem.text) if date_elem is not None else None
            pmid = article.find(".//PMID").text
            if title_elem is None or abstract_elem is None:
                continue
            papers.append({
                "title": title_elem.text,
                "abstract": abstract_elem.text,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "year": year
            })
        return papers
    except Exception as e:
        st.error(f"PubMed error: {e}")
        return []

@st.cache_data(show_spinner=False)
def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=True)

def main():
    if user_input:
        query = user_input.strip()
        if not query:
            st.warning("Please enter a valid research idea.")
            return

        query_keywords = query.lower().split()
        idea_embedding = embed_texts([query])[0]

        with st.spinner("🔎 Searching papers..."):
            ss_papers = search_semantic_scholar(query, limit=num_results)
            arxiv_papers = search_arxiv(query, max_results=num_results)
            pubmed_papers = search_pubmed(query, max_results=num_results)

        all_papers = ss_papers + arxiv_papers + pubmed_papers
        filtered_papers = [p for p in all_papers if p.get("abstract") and p.get("year") and start_year <= p["year"] <= end_year]

        if not filtered_papers:
            st.warning("No papers found in the selected year range.")
            return

        paper_texts = [f"{p['title']}. {p['abstract']}" for p in filtered_papers]
        paper_embeddings = embed_texts(paper_texts)

        scored_papers = []
        for idx, paper in enumerate(filtered_papers):
            similarity = util.pytorch_cos_sim(idea_embedding, paper_embeddings[idx]).item()
            scored_papers.append((similarity, paper))

        scored_papers.sort(key=lambda x: x[0], reverse=True)

        st.write(f"{papers_found_text} {start_year} - {end_year}: {len(scored_papers)} papers")

        for similarity, paper in scored_papers:
            color = "green" if similarity >= 0.75 else "orange" if similarity >= 0.4 else "red"
            highlighted_title = highlight_keywords(paper["title"], query_keywords)
            highlighted_abstract = highlight_keywords(paper["abstract"], query_keywords)
            st.markdown(
                f"<h4><a href='{paper['url']}' target='_blank' style='text-decoration:none; color:#2e86de;'>{highlighted_title}</a> "
                f"<span style='font-size:14px; color:#555;'>({paper['year']})</span></h4>",
                unsafe_allow_html=True
            )
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>Similarity Score: {similarity:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:15px;'>{highlighted_abstract[:1000]}...</p>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

        # CSV download
        df = pd.DataFrame([{
            "Title": p["title"],
            "Abstract": p["abstract"],
            "Year": p["year"],
            "URL": p["url"],
            "Similarity": s
        } for s, p in scored_papers])
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        st.download_button(download_label, buffer.getvalue(), "results.csv", "text/csv")

if __name__ == "__main__":
    main()

# Share Section
share_url = "https://your-app-url.com"
share_text = "Check out this website, it helps me with my research"
encoded_text = urllib.parse.quote(share_text)
encoded_url = urllib.parse.quote(share_url)

twitter_url = f"https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}"
linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}"

st.markdown(f"""
<section style="text-align:center; margin: 40px 0 20px 0;">
    <p style="font-size:18px; font-weight:500; margin-bottom: 15px;">
        {share_prompt}
    </p>
    <a href="{twitter_url}" target="_blank" 
       style="
          background-color:#000000; color:#fff; 
          padding:12px 28px; margin: 0 10px; 
          border-radius:8px; font-weight:bold;
          font-size:16px; text-decoration:none;
          display:inline-block;">
        Share on X
    </a>
    <a href="{linkedin_url}" target="_blank" 
       style="
          background-color:#0077B5; color:#fff; 
          padding:12px 28px; margin: 0 10px; 
          border-radius:8px; font-weight:bold;
          font-size:16px; text-decoration:none;
          display:inline-block;">
        Share on LinkedIn
    </a>
</section>
<hr style="width: 50%; margin: 40px auto 20px auto; border-color: #ddd;">
<footer style="text-align:center; font-size:14px; color:#555; margin-bottom: 40px;">
    {footer_text}
</footer>
""", unsafe_allow_html=True)

