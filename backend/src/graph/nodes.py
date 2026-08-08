"""LangGraph node implementations for the conversational RAG pipeline.

Mirrors the assignment's flow: an intent classifier for UX routing only
(NOT a security control), a history-aware rephraser, a deterministic
ACL-prefiltered retrieval step, an answer builder, and a grounding/safety
guardrail before the response is returned.
"""

from __future__ import annotations

from functools import lru_cache

from src.config.logger import get_logger
from src.config.settings import settings
from src.ingest.embed_and_store import Embedder
from src.llm.client import chat_completion, parse_json_response
from src.prompts import load_prompt
from src.retrieval.retriever import retrieve

from .state import ChatState

logger = get_logger(__name__)

_GREET_MESSAGE = (
    "Hello! Ask me anything about the earnings calls for the companies "
    "you have access to."
)
_DENY_MESSAGE = (
    "I can only answer questions about the earnings-call documents you're "
    "authorized to access. Could you rephrase your question?"
)
_NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in the documents you have access to that "
    "answers this question."
)
_GUARDRAIL_FALLBACK_MESSAGE = (
    "I don't have enough grounded information in the documents to "
    "confidently answer that."
)


@lru_cache(maxsize=1)
def _get_embedder() -> Embedder:
    return Embedder()


def classify_node(state: ChatState) -> dict:
    prompt = load_prompt("classifier")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["question"]},
    ]
    try:
        raw = chat_completion(messages, temperature=0.0)
        decision = parse_json_response(raw, default={"route": "continue"})
    except Exception:
        logger.exception("Classifier call failed; defaulting to 'continue'")
        decision = {"route": "continue"}

    route = decision.get("route")
    if route not in ("greet", "deny", "continue"):
        route = "continue"
    return {"route": route}


def canned_response_node(state: ChatState) -> dict:
    message = _GREET_MESSAGE if state["route"] == "greet" else _DENY_MESSAGE
    return {"final_answer": message, "citations": []}


def rephrase_node(state: ChatState) -> dict:
    history = state.get("history") or []
    if not history:
        return {"standalone_question": state["question"]}

    prompt = load_prompt("rephraser")
    history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Conversation history:\n{history_text}\n\n"
                f"Follow-up question: {state['question']}\n\n"
                "Standalone question:"
            ),
        },
    ]
    try:
        standalone = chat_completion(messages, temperature=0.0).strip()
    except Exception:
        logger.exception("Rephraser call failed; falling back to raw question")
        standalone = ""
    return {"standalone_question": standalone or state["question"]}


def fetch_node(state: ChatState) -> dict:
    query = state.get("standalone_question") or state["question"]
    chunks = retrieve(
        query,
        state["allowed_companies"],
        top_k=settings.chat_top_k,
        embedder=_get_embedder(),
    )
    return {"chunks": chunks}


def build_answer_node(state: ChatState) -> dict:
    chunks = state.get("chunks") or []
    if not chunks:
        return {"answer": _NO_CONTEXT_MESSAGE, "citations": []}

    context = "\n\n".join(
        f"[{c['chunk_id']}] ({c.get('company_id')}, "
        f"{c.get('fiscal_quarter')} {c.get('fiscal_year')}, "
        f"{c.get('speaker_name')}): {c['text']}"
        for c in chunks
    )
    question = state.get("standalone_question") or state["question"]
    prompt = load_prompt("answer")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    try:
        answer = chat_completion(messages, temperature=settings.chat_temperature).strip()
    except Exception:
        logger.exception("Answer generation failed")
        return {
            "answer": "I ran into an error generating an answer. Please try again.",
            "citations": [],
        }

    cited = [c for c in chunks if c["chunk_id"] in answer]
    citations = cited or chunks[:3]
    return {"answer": answer, "citations": citations}


def guardrail_node(state: ChatState) -> dict:
    answer = state["answer"]
    chunks = state.get("chunks") or []

    if not settings.guardrail_enabled or not chunks:
        return {"final_answer": answer, "guardrail_passed": True}

    context = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    prompt = load_prompt("guardrail")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nAnswer to verify:\n{answer}"},
    ]
    try:
        raw = chat_completion(messages, temperature=0.0)
        verdict = parse_json_response(raw, default={"grounded": True, "safe": True, "reason": ""})
    except Exception:
        # Fail open: the guardrail checks answer *quality*, not access control
        # (that's already enforced deterministically in fetch_node), so an
        # LLM/network hiccup here shouldn't block a correctly-scoped answer.
        logger.exception("Guardrail check failed; failing open")
        return {"final_answer": answer, "guardrail_passed": True}

    grounded = bool(verdict.get("grounded", True))
    safe = bool(verdict.get("safe", True))
    if grounded and safe:
        return {"final_answer": answer, "guardrail_passed": True}

    logger.warning("Guardrail blocked answer: %s", verdict.get("reason"))
    return {
        "final_answer": _GUARDRAIL_FALLBACK_MESSAGE,
        "guardrail_passed": False,
        "citations": [],
    }
