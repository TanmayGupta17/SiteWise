"use client"
import { useState } from 'react'
import axios from 'axios'

export default function Home() {
  const [crawlUrl, setCrawlUrl] = useState('')
  const [step, setStep] = useState(1)
  const [status, setStatus] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploadFile, setUploadFile] = useState(null)
  const [uploadTitle, setUploadTitle] = useState('')
  const [uploadSource, setUploadSource] = useState('')

  const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  const handleUpload = async () => {
    if (!uploadFile) return
    const formData = new FormData()
    formData.append('file', uploadFile)
    if (uploadTitle) formData.append('title', uploadTitle)
    if (uploadSource) formData.append('source_url', uploadSource)

    setLoading(true)
    setStatus('📤 Uploading document...')
    setStep(1)

    try {
      const res = await axios.post(`${API}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setStatus(`✅ Uploaded: ${res.data.title || res.data.filename}`)
      setStep(2)
      setTimeout(handleIndex, 800)
    } catch (err) {
      setStatus(`❌ Upload error: ${err.message}`)
      setLoading(false)
    }
  }

  const handleCrawl = async () => {
    if (!crawlUrl) return
    setLoading(true)
    setStatus('🕷️ Crawling...')
    setStep(1)
    try {
      const res = await axios.post(`${API}/crawl`, {
        start_url: crawlUrl,
        max_pages: 10,
        max_depth: 2,
        crawl_delay_ms: 1000,
        respect_robots_txt: false  // Bypass robots.txt for educational use
      })
      setStatus(`✅ Crawled ${res.data.page_count} pages`)
      setStep(2)
      setTimeout(handleIndex, 1000)
    } catch (err) {
      setStatus(`❌ Crawl error: ${err.message}`)
      setLoading(false)
    }
  }

  const handleIndex = async () => {
    setStatus('📚 Indexing...')
    try {
      const res = await axios.post(`${API}/index`, {
        chunk_size: 800,
        chunk_overlap: 100,
        embedding_model: 'all-MiniLM-L6-v2'
      })
      setStatus(`🎉 Indexed ${res.data.vector_count} chunks`)
      setStep(3)
    } catch (err) {
      setStatus(`❌ Index error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleAsk = async () => {
    if (!question) return
    setLoading(true)
    setAnswer('')
    setSources([])
    try {
      const res = await axios.post(`${API}/ask`, {
        question,
        top_k: 3
      })
      setAnswer(res.data.answer)
      setSources(res.data.sources)
    } catch (err) {
      setAnswer(`❌ Ask error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20 }}>
      <h1>🤖 RAG Service</h1>
      <p>Simple Next.js frontend for crawling and Q&A</p>

      {/* Upload a local document */}
      <section style={{ marginTop: 30, padding: 20, border: '1px solid #ddd', borderRadius: 8 }}>
        <h2>Upload Document</h2>
        <p style={{ marginBottom: 10 }}>Accepts .txt, .md, .json, .pdf. Uploaded files get indexed like crawled pages.</p>
        <input
          type="file"
          accept=".txt,.md,.json,.pdf"
          onChange={e => setUploadFile(e.target.files?.[0] || null)}
          disabled={loading}
          style={{ marginBottom: 10 }}
        />
        <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
          <input
            type="text"
            placeholder="Optional title"
            value={uploadTitle}
            onChange={e => setUploadTitle(e.target.value)}
            disabled={loading}
            style={{ flex: 1, padding: 8 }}
          />
          <input
            type="url"
            placeholder="Optional source URL"
            value={uploadSource}
            onChange={e => setUploadSource(e.target.value)}
            disabled={loading}
            style={{ flex: 1, padding: 8 }}
          />
        </div>
        <button onClick={handleUpload} disabled={loading || !uploadFile} style={{ padding: '10px 20px' }}>
          {loading && step === 1 ? 'Uploading...' : 'Upload & Index'}
        </button>
        {status && <p style={{ marginTop: 10 }}>{status}</p>}
      </section>

      {/* Step 1: Crawl */}
      <section style={{ marginTop: 30, padding: 20, border: '1px solid #ddd', borderRadius: 8 }}>
        <h2>Step 1: Crawl Website</h2>
        <input
          type="url"
          placeholder="https://docs.python.org/3/tutorial/"
          value={crawlUrl}
          onChange={e => setCrawlUrl(e.target.value)}
          style={{ width: '60%', padding: 10, marginRight: 10 }}
          disabled={loading}
        />
        <button onClick={handleCrawl} disabled={loading || !crawlUrl} style={{ padding: '10px 20px' }}>
          {loading && step === 1 ? 'Crawling...' : 'Crawl'}
        </button>
        {status && <p style={{ marginTop: 10 }}>{status}</p>}
      </section>

      {/* Step 2: Ask */}
      <section style={{
        marginTop: 30,
        padding: 20,
        border: '1px solid #ddd',
        borderRadius: 8,
        opacity: step < 3 ? 0.6 : 1
      }}>
        <h2>Step 2: Ask Questions</h2>
        <textarea
          rows={3}
          placeholder="What is Python?"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          style={{ width: '100%', padding: 10 }}
          disabled={loading || step < 3}
        />
        <button
          onClick={handleAsk}
          disabled={loading || step < 3 || !question}
          style={{ padding: '10px 20px', marginTop: 10 }}
        >
          {loading && step === 3 ? 'Thinking...' : 'Ask'}
        </button>
      </section>

      {/* Answer */}
      {answer && (
        <section style={{ marginTop: 30, padding: 20, border: '1px solid #ddd', borderRadius: 8 }}>
          <h2>Answer</h2>
          <p>{answer}</p>
          {sources.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <h3>Sources</h3>
              {sources.map((s, i) => (
                <div key={i} style={{ marginBottom: 10 }}>
                  <a href={s.url} target="_blank" rel="noopener noreferrer">
                    {s.url}
                  </a>
                  <p style={{ fontStyle: 'italic' }}>"{s.snippet}"</p>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
