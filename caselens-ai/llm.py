"""
llm.py — system prompt, Ollama streaming, and all prompt-builder functions.
No Streamlit imports here; returns generators or strings only.
"""

import ollama
from config import MODEL_NAME


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CaseLens AI, a precise document analysis assistant.

Rules:
- Use ONLY the provided context. Never invent or assume facts.
- If information is absent from the documents, say so explicitly.
- Always cite the source filename and chunk number where relevant.
- Be structured, clear, and concise.
- Avoid speculation or external knowledge."""


# ─── Core Streaming ───────────────────────────────────────────────────────────

def ask_llm_stream(prompt: str):
    """Yield response tokens from Ollama one at a time."""
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
    """Run ask_llm_stream and collect the full response as a single string."""
    return "".join(ask_llm_stream(prompt))


# ─── Prompt Builders ──────────────────────────────────────────────────────────

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

Return this structure (plain text, no markdown):

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
- Compare by document filename only. Do not treat sections inside one PDF as separate documents.
- Write "Not available" where information is missing. Do not invent facts."""


def build_timeline_prompt(timeline_context: str) -> str:
    return f"""Create a combined chronological timeline from the documents below.

Context:
{timeline_context}

Output a markdown table using exactly this format:

| Date / Period | Event | Source Document |
|---|---|---|

Rules:
- Include only dates or periods explicitly found in the documents.
- If an event has no date, write "Not dated" in the Date column.
- Order events chronologically where possible.
- Always include the source document filename.
- Do not invent events or dates."""