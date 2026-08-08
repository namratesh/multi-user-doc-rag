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
NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in the documents you have access to that "
    "answers this question."
)
GUARDRAIL_FALLBACK_MESSAGE = (
    "I don't have enough grounded information in the documents to "
    "confidently answer that."
)


@lru_cache(maxsize=1)
def _get_embedder() -> Embedder:
    return Embedder()


def classify_node(state: ChatState) -> dict:
    logger.info("[classify] input question=%r", state["question"])
    prompt = load_prompt("classifier")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["question"]},
    ]
    try:
        raw = chat_completion(messages, temperature=0.0)
        decision = parse_json_response(raw, default={"route": "continue"})
    except Exception:
        logger.exception("[classify] LLM call failed; defaulting to 'continue'")
        decision = {"route": "continue"}

    route = decision.get("route")
    if route not in ("greet", "deny", "continue"):
        route = "continue"
    logger.info("[classify] output route=%r", route)
    return {"route": route}


def canned_response_node(state: ChatState) -> dict:
    logger.info("[canned_response] input route=%r", state["route"])
    message = _GREET_MESSAGE if state["route"] == "greet" else _DENY_MESSAGE
    logger.info("[canned_response] output message=%r", message)
    return {"final_answer": message, "citations": []}


def rephrase_node(state: ChatState) -> dict:
    history = state.get("history") or []
    logger.info(
        "[rephrase] input question=%r history_turns=%d", state["question"], len(history)
    )
    if not history:
        logger.info("[rephrase] no history; output standalone_question=question")
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
        logger.exception("[rephrase] LLM call failed; falling back to raw question")
        standalone = ""
    logger.info("[rephrase] output standalone_question=%r", standalone or state["question"])
    return {"standalone_question": standalone or state["question"]}


def fetch_node(state: ChatState) -> dict:
    query = state.get("standalone_question") or state["question"]
    allowed_companies = state["allowed_companies"]
    logger.info(
        "[fetch] input query=%r allowed_companies=%s top_k=%d",
        query,
        allowed_companies,
        settings.chat_top_k,
    )
    chunks = retrieve(
        query,
        allowed_companies,
        top_k=settings.chat_top_k,
        embedder=_get_embedder(),
    )
    logger.info(
        "[fetch] output chunks=%d chunk_ids=%s",
        len(chunks),
        [c["chunk_id"] for c in chunks],
    )
    return {"chunks": chunks}


def build_answer_messages(question: str, chunks: list[dict]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{c['chunk_id']}] ({c.get('company_id')}, "
        f"{c.get('fiscal_quarter')} {c.get('fiscal_year')}, "
        f"{c.get('speaker_name')}): {c['text']}"
        for c in chunks
    )
    prompt = load_prompt("answer")
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def select_citations(answer: str, chunks: list[dict]) -> list[dict]:
    cited = [c for c in chunks if c["chunk_id"] in answer]
    return cited or chunks[:3]


def build_answer_node(state: ChatState) -> dict:
    chunks = state.get("chunks") or []
    question = state.get("standalone_question") or state["question"]
    logger.info("[build_answer] input question=%r chunks=%d", question, len(chunks))
    if not chunks:
        logger.info("[build_answer] output no context available")
        return {"answer": NO_CONTEXT_MESSAGE, "citations": []}

    messages = build_answer_messages(question, chunks)
    try:
        answer = chat_completion(messages, temperature=settings.chat_temperature).strip()
    except Exception:
        logger.exception("[build_answer] LLM call failed")
        return {
            "answer": "I ran into an error generating an answer. Please try again.",
            "citations": [],
        }

    citations = select_citations(answer, chunks)
    logger.info(
        "[build_answer] output answer_len=%d citations=%d", len(answer), len(citations)
    )
    return {"answer": answer, "citations": citations}


def guardrail_verdict(answer: str, chunks: list[dict]) -> tuple[bool, str]:
    """Returns (passed, reason). Fails open on LLM/network errors and when
    the guardrail is disabled or there's no context to check groundedness
    against -- it verifies answer *quality*, not access control (that's
    already enforced deterministically in `fetch_node`)."""
    if not settings.guardrail_enabled or not chunks:
        return True, ""

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
        logger.exception("[guardrail] LLM call failed; failing open")
        return True, ""

    grounded = bool(verdict.get("grounded", True))
    safe = bool(verdict.get("safe", True))
    return (grounded and safe), verdict.get("reason", "")


def guardrail_node(state: ChatState) -> dict:
    answer = state["answer"]
    chunks = state.get("chunks") or []
    logger.info(
        "[guardrail] input answer_len=%d chunks=%d enabled=%s",
        len(answer),
        len(chunks),
        settings.guardrail_enabled,
    )

    passed, reason = guardrail_verdict(answer, chunks)
    if passed:
        logger.info("[guardrail] output passed=True")
        return {"final_answer": answer, "guardrail_passed": True}

    logger.warning("[guardrail] output passed=False reason=%s", reason)
    return {
        "final_answer": GUARDRAIL_FALLBACK_MESSAGE,
        "guardrail_passed": False,
        "citations": [],
    }
