# 🚀 SiteWise: Retrieval-Augmented Generation Service

**CrawlRAG** is a simple, production-ready Retrieval-Augmented Generation (RAG) pipeline that can:
- 🌐 **Crawl** a website (up to 50 pages) while respecting `robots.txt` and domain boundaries  
- 🧠 **Index** content into 800-character chunks using **FAISS vector search**  
- 💬 **Answer questions** grounded in the retrieved context (with source URLs & snippets)  
- 🚫 **Refuse** gracefully when no relevant evidence is found  

---

## 📚 Table of Contents
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Backend: Run Locally](#backend-run-locally)
- [Frontend: Run Locally](#frontend-run-locally)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Example Requests](#example-requests)
- [Key Decisions & Tradeoffs](#key-decisions--tradeoffs)
- [Testing & Evaluation](#testing--evaluation)
- [Future Work](#future-work)

---

## 🎬 Demo

Example run:
- **Crawl:** [`https://docs.python.org/3/tutorial/`](https://docs.python.org/3/tutorial/)
- **Indexed:** ~3086 chunks
- **Ask:** “What is Python used for?” → ✅ detailed answer + sources  
- **Ask:** “Who invented Python?” → 🚫 clear refusal  

---

## ⚙️ Prerequisites
- Python **3.10+**  
- Node **14+** / npm  
- *(Optional)* Git  

---

## 🧩 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crawlrag.git
cd crawlrag

# Backend dependencies
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
npm install
```

