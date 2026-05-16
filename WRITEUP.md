# Technical Write-Up — Smart Job Match Agent

## 1. Design Choices: Embedding Model

### What I chose and why

I implemented **TF-IDF cosine similarity** as the classical ML ranking layer — a sparse vector model that represents documents as weighted term frequencies adjusted by inverse document frequency across the corpus.

The decision was driven by hard deployment constraints. Vercel's free tier caps memory at 1024 MB and execution time at 60 seconds per request. Popular sentence-transformers models (`all-MiniLM-L6-v2` is ~80 MB, `all-mpnet-base-v2` is ~420 MB) would need to be downloaded on cold start, likely exceeding timeout limits and potentially hitting the memory ceiling when combined with the LLM API call. Using an API-based embedding service (OpenAI `text-embedding-3-small`, Cohere) would have worked but introduces a second external API dependency with its own latency and key management.

TF-IDF gives genuinely meaningful semantic differentiation for this dataset because the job descriptions are rich, domain-specific text. When a resume mentions "LangChain, RAG, LLMs," those terms have high IDF scores (rare in the corpus), so cosine similarity correctly elevates jobs like "Generative AI Engineer" and "Legal AI Engineer" over generic "Data Analyst" roles. Scores meaningfully range from ~0.05 to ~0.55 — not all clustered at 0.9 — which is the correct behavior the assignment calls out.

### Alternatives I considered and rejected

- **`sentence-transformers` (local)**: Best semantic quality, but ~300-500 MB + cold-start download → Vercel incompatible without a workaround like baking weights into the container.
- **OpenAI `text-embedding-3-small`**: Low latency, good quality, Vercel-safe — but adds a second API key dependency. Adds ~200ms per call and costs per token.
- **Cohere `embed-english-v3.0`**: Similar tradeoffs to OpenAI embeddings. Considered but decided one external API (Anthropic) was cleaner.
- **BM25**: Slightly better retrieval than TF-IDF for long documents (uses saturation/length normalization), but the implementation complexity wasn't justified for a 50-doc corpus.

### Trade-offs made

TF-IDF lacks true semantic understanding — it won't know that "PyTorch" and "deep learning framework" are related if only one appears in a document. However, the LLM reasoning layer in Step 2 compensates for this: it can identify conceptual alignment that the vector model misses, and the write-up treats it as an honest limitation.

---

## 2. Agentic Architecture

### Tool-calling flow

```
User Input (resume_text)
        │
        ▼
 ┌──────────────────────────────────────────┐
 │  LLM Call #1: parse_resume tool          │
 │  Input:  raw resume text                 │
 │  Output: {name, skills, experience_years,│
 │           preferred_roles, education}    │
 └──────────────────┬───────────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────────┐
 │  TF-IDF Cosine Similarity (Python)       │
 │  Input:  resume text + 50 job texts      │
 │  Output: top-10 scored jobs              │
 └──────────────────┬───────────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────────┐
 │  LLM Call #2: reason_job_matches tool    │
 │  Input:  parsed candidate + top-5 jobs  │
 │  Output: {explanations[], clarifying_    │
 │           question}                      │
 └──────────────────┬───────────────────────┘
                    │
                    ▼
             JSON Response
```

### Why two tool calls instead of one?

**Separation of concerns.** The parse_resume tool performs *extraction* — it's a deterministic transformation of unstructured text into a structured schema. The reason_job_matches tool performs *reasoning* — it evaluates fitness across multiple dimensions. Fusing them into one prompt would entangle the tasks: the model would be simultaneously parsing the resume, ranking jobs, and writing prose, increasing the likelihood of format errors and making it harder to debug failures.

**Structured intermediate state.** The parsed candidate profile is a clean JSON artifact that could be reused, cached, or logged independently. This mirrors production system design where parsing and ranking are separate microservices.

**Better tool adherence.** Smaller, single-purpose tool schemas have higher compliance rates in practice. A tool with 7 output fields is easier for the model to fill correctly than one with 15.

### Failure modes

- **Parse failure**: If the resume is non-English, empty, or garbled, the LLM may still call `parse_resume` but with empty arrays or incorrect values. The downstream ranking would still work (TF-IDF on the raw text), but the candidate profile in the response would be poor.
- **Tool call refusal**: Rarely, the model returns a text response instead of a tool call. Handled with an explicit check and 502 error.
- **TF-IDF cold scoring**: Very short resumes (< 50 chars) get blocked by input validation. Resumes with unconventional formatting (all-caps, symbols) may tokenize poorly.
- **Token limits**: Very long resumes fed into Step 2's prompt alongside 5 job descriptions could approach context limits. In production, the resume would be summarized before Step 2.

---

## 3. Honest Weaknesses

### Noisy or poorly written resumes

TF-IDF is lexically sensitive. A resume that writes "deep learning" instead of "PyTorch" or "LLM apps" instead of "LangChain" may score lower against jobs that use the specific tech names. Acronym inconsistency (ML vs Machine Learning) also reduces similarity. The LLM reasoning layer partially compensates, but it only sees the top-10 similarity candidates — if a perfect match is ranked 15th by TF-IDF, it never gets reasoned about.

### At scale (10,000 concurrent requests)

- **Blocking**: Each request makes two sequential LLM API calls (~3-6s combined). At 10K concurrent users, this would require aggressive connection pooling and queuing.
- **Rate limits**: Anthropic's API has per-minute token limits. 10K concurrent users would immediately exhaust them. Solution: a job queue (Celery + Redis) with async result polling.
- **No caching**: Identical resumes re-embed and re-call the LLM. A Redis cache keyed on `hash(resume_text)` would eliminate repeat work.
- **Stateless ranking**: TF-IDF re-computes query vectors per request. At scale, pre-computed job vectors (done at startup) help, but the query vectorization is still per-request.
- **Vercel cold starts**: First request per instance is slow (~2-3s for Python import + JSON load). A warm-up cron job or persistent server (Railway, Render) would be better for production.

### Corners cut due to time

- No resume file upload (PDF/DOCX); only raw text input.
- No persistent logging of recommendations (useful for offline evaluation).
- TF-IDF instead of dense embeddings — acceptable for 50 jobs, but would degrade on a 50,000-job corpus where keyword overlap is too sparse to distinguish roles.
- No retry logic on the Anthropic API (transient errors return 502 immediately).
- The `/refine` endpoint doesn't re-run the full agent — it only re-ranks candidates already retrieved, which means if the answer changes the semantic match entirely, a genuinely new top match may not surface.

---

## 4. Next Steps

**If I had two more days, the single highest-impact improvement would be replacing TF-IDF with a proper dense embedding model.**

Specifically: use OpenAI's `text-embedding-3-small` API to embed all 50 job descriptions at startup, store them as a matrix, and embed each incoming resume at query time. Total latency overhead: ~150ms. Quality improvement: substantial — dense embeddings capture semantic proximity ("NLP" ↔ "language model"), domain signals ("clinical" ↔ "healthcare"), and role-level relationships ("data scientist" ↔ "ML researcher") that TF-IDF completely misses.

This would fix the most common failure mode (two conceptually identical documents scoring low because they use different vocabulary) without any infrastructure changes — it's a 30-line swap that fits within Vercel's constraints and costs fractions of a cent per request.

The second improvement (10 minutes after the first) would be pre-embedding all jobs once at deploy time and saving the vectors as a JSON file, eliminating the startup embedding cost entirely and making cold starts fast.
