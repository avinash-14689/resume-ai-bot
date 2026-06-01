---
title: ResumeAI - Smart Resume Consultant
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 📄 ResumeAI — Smart Resume Consultant

A powerful resume analysis chatbot built with **FastAPI**, **LlamaIndex**, and **Groq LLM**.

## ✨ Features
- Upload your PDF resume and get instant analysis
- Company fit percentages (Product-based, Service-based, AI/ML, PSU, Startups)
- Resume rating out of 5
- Constructive, friendly feedback
- Resume building guidance for any company type
- Beginner-friendly resume tips
- Multi-turn conversation with memory

## 🚀 Setup

### HuggingFace Spaces
1. Fork this repo to your HuggingFace account
2. Go to **Settings → Variables and Secrets**
3. Add secret: `GROQ_API_KEY` = your Groq API key
4. The app will auto-deploy via Docker

### Local (Windows — double click setup.bat)
```
setup.bat
```

### Local (Mac/Linux)
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
python app.py
```
Then open: http://localhost:7860

## 🔑 Getting a Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / log in
3. API Keys → Create new key
4. Paste it in the app header or set as env variable

## 🛠️ Tech Stack
- **Backend**: FastAPI + Uvicorn
- **LLM**: Llama 3 70B via Groq API
- **Indexing**: LlamaIndex + SimpleVectorStore
- **Embeddings**: BAAI/bge-small-en-v1.5 (HuggingFace)
- **Frontend**: Vanilla HTML/CSS/JS + Chart.js
- **Deployment**: HuggingFace Spaces (Docker)
# resume-ai-bot
