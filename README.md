# Smart Job Match Agent

An AI-powered job recommendation API that combines classical ML (TF-IDF cosine similarity) with an LLM agentic layer (Claude tool calling) to match candidates to relevant jobs.

## Architecture

```
Resume Text
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 1 — Agent: parse_resume tool call             │
│  → Claude extracts: name, skills, experience,       │
│    preferred_roles, education                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Classical ML: TF-IDF Cosine Similarity Ranking     │
│  → Tokenize resume + all 50 job descriptions        │
│  → Build IDF corpus from all docs                   │
│  → Compute cosine similarity for each job           │
│  → Return top-10 ranked candidates                  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 2 — Agent: reason_job_matches tool call       │
│  → Claude reasons over top-5 matches                │
│  → Writes 2-3 sentence explanation per job          │
│  → Generates one smart clarifying question          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
                 JSON Response
```

## Setup (5 commands)

```bash
git clone <your-repo-url>
cd smart-job-match
cp .env.example .env          # Add your ANTHROPIC_API_KEY
pip install -r requirements.txt
uvicorn api.index:app --reload
```

API is now live at `http://localhost:8000`

## Endpoints

### POST /recommend
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Jane Doe, B.Tech CS. Python, NLP, HuggingFace, PyTorch. 1 year experience building LLM apps and RAG pipelines. Looking for AI/ML engineering roles."}'
```

Response:
```json
{
  "candidate": {
    "name": "Jane Doe",
    "skills": ["Python", "NLP", "HuggingFace", "PyTorch"],
    "experience_years": 1,
    "preferred_roles": ["ML Engineer", "NLP Engineer"],
    "education": "B.Tech Computer Science"
  },
  "ranked_jobs": [
    {
      "id": 8,
      "title": "NLP Engineer",
      "company": "LinguaTech",
      "similarity_score": 0.4821,
      "explanation": "..."
    }
  ],
  "clarifying_question": "Your resume highlights NLP and LLMs but doesn't mention cloud experience — are you comfortable deploying models on AWS or GCP?"
}
```

### POST /refine (Bonus)
```bash
curl -X POST http://localhost:8000/refine \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "...",
    "clarifying_question": "Are you open to remote work?",
    "candidate_answer": "Yes, I strongly prefer remote roles."
  }'
```

## Deployment (Vercel)

```bash
npm i -g vercel
vercel login
vercel --prod
```

Set `ANTHROPIC_API_KEY` in Vercel Dashboard → Project → Settings → Environment Variables.

## Design Decisions

See [WRITEUP.md](./WRITEUP.md) for full technical write-up.

### Key choices:
- **Embedding model**: TF-IDF cosine similarity (pure Python, zero dependencies, Vercel-compatible). See WRITEUP for rationale over sentence-transformers.
- **LLM**: Claude (`claude-sonnet-4-20250514`) via native tool-calling API
- **Agent design**: Two separate tool calls (parse_resume → reason_job_matches) — not prompt chaining
- **Framework**: FastAPI with async/await throughout for non-blocking LLM calls
