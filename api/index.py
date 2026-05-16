import os
import json
import math
import re
from collections import Counter
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Job Match Agent",
    description="AI-powered job recommendation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load jobs at startup ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_PATH = os.path.join(BASE_DIR, "jobs.json")

with open(JOBS_PATH, "r") as f:
    JOBS: list[dict] = json.load(f)

def job_to_text(job: dict) -> str:
    skills_str = ", ".join(job.get("skills", []))
    remote_str = "remote" if job.get("remote") else "on-site"
    return (
        f"Title: {job['title']}. Company: {job['company']}. "
        f"Domain: {job['domain']}. Location: {job['location']} ({remote_str}). "
        f"Required skills: {skills_str}. "
        f"Experience required: {job['experience_years']} year(s). "
        f"Description: {job['description']}"
    )

JOB_TEXTS = [job_to_text(j) for j in JOBS]

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── TF-IDF Cosine Similarity (Classical ML) ───────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def compute_tfidf_vectors(documents: list[str]):
    N = len(documents)
    tokenized = [tokenize(d) for d in documents]
    df: dict[str, int] = {}
    for tokens in tokenized:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1
    idf = {tok: math.log((N + 1) / (freq + 1)) + 1 for tok, freq in df.items()}
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = {tok: (count / total) * idf.get(tok, 1) for tok, count in tf.items()}
        vectors.append(vec)
    return vectors, idf

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def tfidf_query_vector(query: str, idf: dict) -> dict:
    tokens = tokenize(query)
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {tok: (count / total) * idf.get(tok, 1) for tok, count in tf.items()}

# Pre-compute job vectors at startup
JOB_VECTORS, CORPUS_IDF = compute_tfidf_vectors(JOB_TEXTS)

def rank_jobs_by_similarity(resume_text: str, top_n: int = 10) -> list[tuple[dict, float]]:
    query_vec = tfidf_query_vector(resume_text, CORPUS_IDF)
    scored = [
        (JOBS[i], cosine_similarity(query_vec, JOB_VECTORS[i]))
        for i in range(len(JOBS))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# ── Pydantic models ───────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    resume_text: str

    @field_validator("resume_text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("resume_text must not be empty")
        if len(v) < 50:
            raise ValueError("resume_text is too short")
        return v

class RefineRequest(BaseModel):
    resume_text: str
    clarifying_question: str
    candidate_answer: str

    @field_validator("resume_text", "clarifying_question", "candidate_answer")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty")
        return v

# ── Tool definitions ──────────────────────────────────────────────────────────
PARSE_RESUME_TOOL = {
    "type": "function",
    "function": {
        "name": "parse_resume",
        "description": "Extract structured information from a candidate's raw resume text.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Candidate's full name"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of technical and soft skills mentioned",
                },
                "experience_years": {
                    "type": "number",
                    "description": "Total years of professional experience",
                },
                "preferred_roles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Roles the candidate is targeting",
                },
                "education": {
                    "type": "string",
                    "description": "Highest degree and field, e.g. B.Tech Computer Science",
                },
            },
            "required": ["name", "skills", "experience_years", "preferred_roles", "education"],
        },
    },
}

REASON_MATCHES_TOOL = {
    "type": "function",
    "function": {
        "name": "reason_job_matches",
        "description": "Produce explanations for job matches and a clarifying question.",
        "parameters": {
            "type": "object",
            "properties": {
                "explanations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "integer"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["job_id", "explanation"],
                    },
                    "description": "One explanation per job",
                },
                "clarifying_question": {
                    "type": "string",
                    "description": "One smart follow-up question to ask the candidate",
                },
            },
            "required": ["explanations", "clarifying_question"],
        },
    },
}

# ── Groq agent ────────────────────────────────────────────────────────────────
def run_agent(resume_text: str) -> dict:
    client = Groq(api_key=GROQ_API_KEY)

    # ── Step 1: Parse Resume ──────────────────────────────────────────────────
    msg1 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert resume parser. "
                    "Call the parse_resume tool with accurate data from the resume."
                ),
            },
            {"role": "user", "content": resume_text},
        ],
        tools=[PARSE_RESUME_TOOL],
        tool_choice={"type": "function", "function": {"name": "parse_resume"}},
    )

    tool_call = msg1.choices[0].message.tool_calls
    if not tool_call:
        raise HTTPException(status_code=502, detail="Agent failed to parse resume")

    parsed_candidate = json.loads(tool_call[0].function.arguments)

    # ── Classical ML ranking ──────────────────────────────────────────────────
    top_candidates = rank_jobs_by_similarity(resume_text, top_n=10)
    top5 = top_candidates[:5]

    # ── Step 2: Reason matches ────────────────────────────────────────────────
    jobs_summary = json.dumps(
        [
            {
                "id": j["id"],
                "title": j["title"],
                "company": j["company"],
                "domain": j["domain"],
                "remote": j["remote"],
                "skills": j["skills"],
                "experience_years": j["experience_years"],
                "description": j["description"],
                "similarity_score": round(score, 4),
            }
            for j, score in top5
        ],
        indent=2,
    )

    step2_prompt = (
        f"Candidate profile:\n{json.dumps(parsed_candidate, indent=2)}\n\n"
        f"Top 5 job matches:\n{jobs_summary}\n\n"
        "Call reason_job_matches to write a 2-3 sentence explanation for each job "
        "and generate one smart clarifying question."
    )

    msg2 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior technical recruiter. "
                    "Give honest, specific match reasoning. "
                    "Call the reason_job_matches tool."
                ),
            },
            {"role": "user", "content": step2_prompt},
        ],
        tools=[REASON_MATCHES_TOOL],
        tool_choice={"type": "function", "function": {"name": "reason_job_matches"}},
    )

    tool_call2 = msg2.choices[0].message.tool_calls
    if not tool_call2:
        raise HTTPException(status_code=502, detail="Agent failed to reason matches")

    reasoning_output = json.loads(tool_call2[0].function.arguments)
    explanations_map = {e["job_id"]: e["explanation"] for e in reasoning_output["explanations"]}

    # ── Assemble response ─────────────────────────────────────────────────────
    ranked_jobs = [
        {
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "remote": job["remote"],
            "domain": job["domain"],
            "similarity_score": round(score, 4),
            "explanation": explanations_map.get(job["id"], "No explanation available."),
        }
        for job, score in top5
    ]

    return {
        "candidate": {
            "name": parsed_candidate.get("name", "Unknown"),
            "skills": parsed_candidate.get("skills", []),
            "experience_years": parsed_candidate.get("experience_years", 0),
            "preferred_roles": parsed_candidate.get("preferred_roles", []),
            "education": parsed_candidate.get("education", ""),
        },
        "ranked_jobs": ranked_jobs,
        "clarifying_question": reasoning_output.get("clarifying_question", ""),
    }


# ── /recommend endpoint ───────────────────────────────────────────────────────
@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    return run_agent(req.resume_text)


# ── /refine endpoint (bonus) ──────────────────────────────────────────────────
@app.post("/refine")
def refine(req: RefineRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    client = Groq(api_key=GROQ_API_KEY)
    top_candidates = rank_jobs_by_similarity(req.resume_text, top_n=15)

    RERANK_TOOL = {
        "type": "function",
        "function": {
            "name": "rerank_jobs",
            "description": "Re-rank jobs based on candidate's answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ranked_job_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Job IDs re-ordered best to worst (up to 5)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why the ranking changed based on the answer",
                    },
                },
                "required": ["ranked_job_ids", "reasoning"],
            },
        },
    }

    jobs_summary = json.dumps(
        [{"id": j["id"], "title": j["title"], "company": j["company"],
          "domain": j["domain"], "remote": j["remote"]} for j, _ in top_candidates],
        indent=2,
    )

    prompt = (
        f"Resume:\n{req.resume_text}\n\n"
        f"Question asked: {req.clarifying_question}\n"
        f"Candidate answered: {req.candidate_answer}\n\n"
        f"Jobs to re-rank:\n{jobs_summary}\n\n"
        "Call rerank_jobs to pick the best 5 and explain why."
    )

    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a senior recruiter. Call the rerank_jobs tool."},
            {"role": "user", "content": prompt},
        ],
        tools=[RERANK_TOOL],
        tool_choice={"type": "function", "function": {"name": "rerank_jobs"}},
    )

    tool_block = msg.choices[0].message.tool_calls
    if not tool_block:
        raise HTTPException(status_code=502, detail="Agent failed to re-rank")

    output = json.loads(tool_block[0].function.arguments)
    job_map = {j["id"]: (j, score) for j, score in top_candidates}

    reranked = []
    for jid in output["ranked_job_ids"][:5]:
        if jid in job_map:
            job, score = job_map[jid]
            reranked.append({
                "id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "remote": job["remote"],
                "domain": job["domain"],
                "similarity_score": round(score, 4),
            })

    return {"ranked_jobs": reranked, "reasoning": output.get("reasoning", "")}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Smart Job Match Agent",
        "status": "ok",
        "jobs_loaded": len(JOBS),
        "endpoints": ["/recommend", "/refine"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}
