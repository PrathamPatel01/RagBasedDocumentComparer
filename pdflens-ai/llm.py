import ollama
from config import MODEL_NAME

SYSTEM_PROMPT = """You are CaseLens AI, a precise document analysis assistant.

Rules:
- Use ONLY the provided context. Never invent or assume facts.
- If information is absent from the documents, say so explicitly.
- Always cite the source filename and chunk number where relevant.
- Be structured, clear, and concise.
- Avoid speculation or external knowledge."""
    

def ask_llm_stream(prompt: str):
    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    except Exception as e:
        yield (
            f"\n\n⚠️ **LLM Error:** `{e}`\n\n"
            f"Make sure Ollama is running and `{MODEL_NAME}` is available.\n\n"
            f"```bash\nollama pull {MODEL_NAME}\n```"
        )


def collect_stream(prompt: str) -> str:
    return "".join(ask_llm_stream(prompt))


def build_qa_prompt(context: str, question: str) -> str:
    return f"""Context:
{context}

Question:
{question}

Answer clearly and concisely. Cite the source filename and chunk number for each fact you reference."""


def build_summary_prompt(filename: str, context: str) -> str:
    return f"""Summarize this ONE document only.

Document filename: {filename}

Context:
{context}

Use this exact format:

## {filename}

- **Document type:**
- **Main purpose:**
- **Core topic:**
- **Key issues:**
- **Important facts:**
- **Main arguments / findings:**
- **Conclusion:**
- **Best use case:**
- **Source references:**

Rules:
- Summarize only this document. Do not mention or compare other documents.
- Do not invent facts. Use only the provided context."""


def build_comparison_presummary_prompt(filename: str, context: str) -> str:
    return f"""Summarize ONE document only for structured comparison.

Document filename: {filename}

Context:
{context}

Return this structure:

Document: {filename}
Document type:
Main purpose:
Core topic:
Legal or research focus:
Key issues:
Important facts:
Main arguments or findings:
Type of evidence or material:
Scope:
Strengths:
Limitations:
Best use case:

Rules:
- Summarize only this document. No cross-references.
- Do not invent facts."""


def build_comparison_table_prompt(filenames: list[str], combined_summaries: str) -> str:
    filename_columns = " | ".join(filenames)

    return f"""Compare the following documents using ONLY the summaries provided below.

{combined_summaries}

Create a markdown comparison table with these exact columns:
Category | {filename_columns} | Key Difference

Include these rows:
- Document type
- Main purpose
- Core topic
- Legal / historical focus
- Main issues discussed
- Type of evidence or material
- Important findings
- Scope
- Strengths
- Limitations
- Best use case

Rules:
- Compare by document filename only.
- Write "Not available" where information is missing.
- Do not invent facts."""


def build_timeline_prompt(timeline_context: str) -> str:
    return f"""Create a chronological timeline using ONLY the context below.

Context:
{timeline_context}

Output a markdown table:

| Date / Period | Event | Source Document | Source Chunk |
|---|---|---|---|

Rules:
- Include ONLY events with an explicit date or period in the context.
- Do NOT include "Not dated" rows.
- Do NOT use document titles as events.
- Do NOT create rows from summaries unless they contain a specific dated event.
- Merge duplicate events.
- Sort chronologically where possible.
- Every row must cite the filename and chunk number.
- If no dated events are present, say exactly: No explicit dated events found.
"""