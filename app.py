import os
import json
import re
import tempfile
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import PyPDF2

load_dotenv()

app = FastAPI(title="Resume Q&A Bot")

# Load API key from environment on startup (HF Secrets / .env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
chat_engine = None
resume_text_global = ""
company_analysis_cache = None

# ── LLM + Embedding setup ─────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def get_llm():
    return Groq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.4,
        max_tokens=2048,
    )

def get_embed_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ResumeAI — a professional, friendly, and encouraging resume consultant chatbot.

Your personality:
- Warm, supportive, and never rude or harsh
- Professional yet conversational tone
- Always constructive — frame weaknesses as opportunities
- Encouraging for beginners and beginners with few projects

Your capabilities:
1. RESUME ANALYSIS: When given a resume, analyze skills, experience, projects, education
2. COMPANY FIT: Suggest which company categories the candidate fits (PSU, Product-based, Service-based, AI/ML, Startups)
3. RATING: Rate resumes out of 5 with clear justification
4. FEEDBACK: Give specific, actionable, kind feedback
5. RESUME BUILDING: Guide users on building resumes for specific company types
6. BEGINNER HELP: Help people with few/no projects build compelling resumes

When rating, always use format: "Rating: X/5"
When giving company fit, be specific about which categories and why.
Always end responses with an encouraging note or next step suggestion.

Resume context will be provided. Use it to give personalized answers.
If no resume is uploaded yet, guide the user to upload one for personalized advice,
but still answer general resume questions helpfully."""


# ── PDF Extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""

        os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {str(e)}")
    return text.strip()


# ── Build chat engine from resume text ───────────────────────────────────────
def build_chat_engine(resume_text: str):
    global chat_engine

    llm = get_llm()
    embed_model = get_embed_model()

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    document = Document(
        text=resume_text,
        metadata={"source": "uploaded_resume", "type": "resume"}
    )

    index = VectorStoreIndex.from_documents(
        [document],
        embed_model=embed_model,
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        llm=llm,
        context_prompt=(
            f"{SYSTEM_PROMPT}\n\n"
            "Here is the candidate's resume:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Using this resume context and the conversation history, answer the user's question."
        ),
        verbose=False,
    )
    return chat_engine


# ── Company analysis via Groq directly ───────────────────────────────────────
def analyze_company_fit(resume_text: str) -> dict:
    from groq import Groq as GroqClient

    client = GroqClient(api_key=GROQ_API_KEY)

    prompt = f"""Analyze this resume and return ONLY a valid JSON object (no markdown, no explanation) with these exact keys:

{{
  "rating": <number 1-5, can be decimal like 3.5>,
  "rating_reason": "<one sentence reason for the rating>",
  "companies": {{
    "Product-Based": <percentage 0-100>,
    "Service-Based": <percentage 0-100>,
    "AI/ML Companies": <percentage 0-100>,
    "Startups": <percentage 0-100>,
    "PSU/Govt": <percentage 0-100>
  }},
  "top_skills": ["skill1", "skill2", "skill3"],
  "missing_skills": ["skill1", "skill2"],
  "summary": "<2-3 sentence encouraging summary of the candidate>",
  "quick_wins": ["actionable tip 1", "actionable tip 2", "actionable tip 3"]
}}

The percentages represent the candidate's chances/fit for each company type based on their resume.
They don't need to sum to 100 — each is independent.

Resume:
{resume_text[:3000]}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback defaults
        data = {
            "rating": 3.0,
            "rating_reason": "Resume analyzed successfully.",
            "companies": {
                "Product-Based": 40,
                "Service-Based": 60,
                "AI/ML Companies": 30,
                "Startups": 50,
                "PSU/Govt": 20,
            },
            "top_skills": ["Communication", "Problem Solving"],
            "missing_skills": ["Portfolio projects", "Quantified achievements"],
            "summary": "You have a solid foundation to build upon. Keep growing!",
            "quick_wins": [
                "Add measurable impact to your bullet points",
                "Include a GitHub or portfolio link",
                "Tailor your resume for each application",
            ],
        }

    return data


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None


class APIKeyRequest(BaseModel):
    api_key: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.post("/set-key")
async def set_api_key(request: APIKeyRequest):
    global GROQ_API_KEY
    GROQ_API_KEY = request.api_key.strip()
    return {"status": "ok", "message": "API key set successfully"}




@app.get("/check-key")
async def check_key():
    return {"has_key": bool(GROQ_API_KEY)}
@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    global resume_text_global, company_analysis_cache, chat_engine

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="Please set your Groq API key first.")

    file_bytes = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)

    if len(resume_text) < 50:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. Make sure it's not a scanned image.")

    resume_text_global = resume_text

    # Build vector index + chat engine
    build_chat_engine(resume_text)

    # Analyze company fit
    company_analysis_cache = analyze_company_fit(resume_text)

    return JSONResponse({
        "status": "ok",
        "message": "Resume uploaded and indexed successfully!",
        "analysis": company_analysis_cache,
        "char_count": len(resume_text),
    })


@app.post("/chat")
async def chat(request: ChatRequest):
    global chat_engine, resume_text_global

    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="Please set your Groq API key first.")

    # If no resume uploaded yet, use general LLM without vector index
    if chat_engine is None:
        from groq import Groq as GroqClient
        client = GroqClient(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        return {"response": answer, "has_resume": False}

    # Use LlamaIndex chat engine (multi-turn with memory)
    try:
        response = chat_engine.chat(request.message)
        return {"response": str(response), "has_resume": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/analysis")
async def get_analysis():
    if company_analysis_cache is None:
        raise HTTPException(status_code=404, detail="No resume uploaded yet.")
    return company_analysis_cache


@app.post("/reset")
async def reset():
    global chat_engine, resume_text_global, company_analysis_cache
    chat_engine = None
    resume_text_global = ""
    company_analysis_cache = None
    return {"status": "ok", "message": "Session reset."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)