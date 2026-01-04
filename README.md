# 🚀 SiteWise: Retrieval-Augmented Generation Service

**SiteWise** is a simple, production-ready Retrieval-Augmented Generation (RAG) pipeline that can:
- 🌐 **Crawl** websites or **Upload PDFs** to extract knowledge  
- 🧠 **Index** content into chunks using **FAISS vector search**  
- 💬 **Answer questions** grounded in real sources (no hallucinations)  
- 🚫 **Refuse** gracefully when evidence is missing  

---

## ❓ Why Did We Choose This Problem?

**The Challenge:** Students struggle to learn complex topics from scattered sources, and AI chatbots sometimes give wrong answers (hallucinations).

**Our Solution:** SiteWise is a **learning assistant** that helps students:
- 📖 **Understand any topic** by uploading study materials (PDFs, documents)
- 🎯 **Get accurate answers** without hallucinations—every answer is backed by real sources
- 🧪 **Quiz themselves** with questions from their study material before exams
- ✅ **See sources** for every answer to verify information

Instead of trusting an AI's memory, SiteWise retrieves facts from *your* documents and generates answers only from that knowledge. Perfect for exam prep!

---

## 📚 Table of Contents
- [Why This Problem?](#-why-did-we-choose-this-problem)
- [How It Works](#-how-it-works)
- [AI & Cloud Services](#-ai--cloud-services-used)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Backend: Run Locally](#backend-run-locally)
- [Frontend: Run Locally](#frontend-run-locally)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Example Requests](#example-requests)
- [Key Decisions & Tradeoffs](#key-decisions--tradeoffs)
- [Testing & Evaluation](#testing--evaluation)
- [Future Work](#future-work)

---

## 🔄 How It Works

1. **Upload** a PDF or document with your study material  
2. **Index** it into searchable chunks (SiteWise breaks it into small pieces)  
3. **Ask** any question about that material  
4. **Get answers** with sources—no guessing, no hallucinations!

The system only answers questions from *your uploaded content*—if information isn't there, it honestly tells you.

---
- **Ask:** “What is Python used for?” → ✅ detailed answer + sources  
- **Ask:** “Who invented Python?” → 🚫 clear refusal  

---

## 🤖 AI & Cloud Services Used

### **LLM & Generative AI**
- **Google Gemini API** (`gemini-2.5-flash-lite`)
  - Generates natural language answers grounded in retrieved context
  - Fast, cost-effective, and production-ready

### **Embeddings & Vector Search**
- **Sentence Transformers** (`all-MiniLM-L6-v2`)
  - Converts text into semantic embeddings (384 dimensions)
  - Lightweight model (~80MB), runs locally without GPU
  - Great for educational use—fast inference on CPU
  
- **FAISS** (Facebook AI Similarity Search)
  - Vector database for fast semantic search
  - Handles millions of chunks with sub-millisecond query time
  - Runs entirely locally (no external service)

### **NLP & Text Processing**
- **BeautifulSoup4**
  - Web scraping and HTML parsing
  - Extracts clean text from website pages

- **pdfplumber**
  - Extracts text from PDF documents
  - Handles multi-page PDFs accurately

### **Backend Framework**
- **FastAPI**
  - Modern Python REST API framework
  - Automatic API documentation (/docs endpoint)
  - Built-in request validation with Pydantic

### **Frontend Framework**
- **Next.js 15** & **React 19**
  - Server-side rendering for SEO
  - Modern JavaScript for interactive UI
  - Tailwind CSS for styling

### **Data & Storage**
- **JSON files** (local storage)
  - Crawled documents stored as JSON in `data/crawled/`
  - Config and metadata in JSON for simplicity

- **NumPy**
  - Efficient embedding storage and computation
  - Embeddings saved as `.npy` files

### **Development & Deployment**
- **Python 3.10+** runtime
- **Node.js + npm** for frontend tooling
- **Git/GitHub** for version control

### **Why These Choices?**
✅ **No cloud lock-in** — embeddings run locally, no expensive API calls
✅ **Privacy-first** — all data stays on your machine
✅ **Fast & lightweight** — MiniLM is optimized for CPU inference
✅ **Affordable** — only pay for Gemini API, not embedding generation
✅ **Production-ready** — FAISS, Gemini are battle-tested in industry

---
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

