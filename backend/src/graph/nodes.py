"""LangGraph node implementations for the conversational RAG pipeline.

Mirrors the assignment's flow: an intent classifier for UX routing only
(NOT a security control), a history-aware rephraser, a deterministic
ACL-prefiltered retrieval step, an answer builder, and a grounding/safety
guardrail before the response is returned.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from langsmith import traceable

from src.config import observability
from src.config.logger import get_logger
from src.config.settings import settings
from src.config.users import (
    COMPANY_DISPLAY_NAMES,
    get_company_catalog_text,
    get_company_display_names,
)
from src.ingest.embed_and_store import Embedder
from src.llm.client import chat_completion, parse_json_response
from src.prompts import load_prompt
from src.retrieval.retriever import retrieve

from .state import AnswerOneInput, ChatState, FetchOneInput

logger = get_logger(__name__)

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


def _format_company_list(companies: list[str]) -> str:
    names = get_company_display_names(companies)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _build_greet_message(allowed_companies: list[str]) -> str:
    if not allowed_companies:
        return (
            "Hello! You don't currently have access to any companies' "
            "earnings calls, so I won't be able to answer questions yet."
        )
    return (
        f"Hello! Ask me anything about the earnings calls for "
        f"{_format_company_list(allowed_companies)}, the companies you "
        "have access to."
    )


@traceable(name="classify", run_type="chain", client=observability.langsmith_client)
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


def _generate_greet_message(question: str, allowed_companies: list[str]) -> str:
    names = get_company_display_names(allowed_companies)
    companies_line = ", ".join(names) if names else "(none)"
    prompt = load_prompt("greet")
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"User's message: {question}\n"
                f"Authorized companies: {companies_line}"
            ),
        },
    ]
    try:
        return chat_completion(messages, temperature=0.3).strip()
    except Exception:
        logger.exception("[canned_response] LLM call failed; falling back to static greeting")
        return _build_greet_message(allowed_companies)


@traceable(name="canned_response", run_type="chain", client=observability.langsmith_client)
def canned_response_node(state: ChatState) -> dict:
    logger.info("[canned_response] input route=%r", state["route"])
    if state["route"] == "greet":
        message = _generate_greet_message(state["question"], state.get("allowed_companies") or [])
    else:
        message = _DENY_MESSAGE
    logger.info("[canned_response] output message=%r", message)
    return {"final_answer": message, "citations": []}


@traceable(name="rephrase", run_type="chain", client=observability.langsmith_client)
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


@traceable(name="decompose", run_type="chain", client=observability.langsmith_client)
def decompose_node(state: ChatState) -> dict:
    """Splits a question that names several companies into one
    self-contained sub-question per company (a single-item list, tagged
    company_id=None, for a single-company or general question).

    Downstream, each sub-question is fetched and answered independently and
    scoped to its own company -- so a question about companies the caller
    has mixed access to can still be answered for the ones it does, instead
    of the whole reply being withheld because one part of it can't be. This
    is UX/quality only, same as `classify_node`: access control is still
    enforced deterministically in `fetch_one_node` via `allowed_companies`.
    """
    question = state.get("standalone_question") or state["question"]
    logger.info("[decompose] input question=%r", question)
    fallback = [{"company_id": None, "question": question}]

    prompt = load_prompt("decompose").replace(
        "{company_catalog}", get_company_catalog_text()
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    try:
        raw = chat_completion(messages, temperature=0.0)
        decision = parse_json_response(raw, default={"sub_queries": fallback})
    except Exception:
        logger.exception("[decompose] LLM call failed; falling back to single sub-query")
        decision = {"sub_queries": fallback}

    sub_queries: list[dict] = []
    for sq in decision.get("sub_queries") or []:
        if not isinstance(sq, dict):
            continue
        question_text = sq.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            continue
        company_id = sq.get("company_id")
        if company_id not in COMPANY_DISPLAY_NAMES:
            company_id = None
        sub_queries.append({"company_id": company_id, "question": question_text.strip()})

    if not sub_queries:
        sub_queries = fallback
    logger.info("[decompose] output sub_queries=%s", sub_queries)
    return {"sub_queries": sub_queries}


def _fetch_for_sub_query(sub_query: dict, allowed_companies: list[str], top_k: int) -> dict:
    company_id = sub_query.get("company_id")
    if company_id is not None:
        # The sub-question targets one specific company -- search only that
        # company, and only if the caller is authorized for it. This is the
        # deterministic access-control check: an unauthorized company never
        # even reaches the vector search.
        companies = [company_id] if company_id in allowed_companies else []
    else:
        companies = allowed_companies

    chunks = (
        retrieve(sub_query["question"], companies, top_k=top_k, embedder=_get_embedder())
        if companies
        else []
    )
    return {**sub_query, "chunks": chunks}


@traceable(name="fetch_one", run_type="retriever", client=observability.langsmith_client)
def fetch_one_node(state: FetchOneInput) -> dict:
    """Fetch branch for exactly one sub_query, dispatched via `Send` from
    `route_to_fetch` -- LangGraph runs however many of these are fanned out
    concurrently, then merges their single-item lists into `sub_results` via
    that field's `operator.add` reducer once every branch has completed."""
    result = _fetch_for_sub_query(
        state["sub_query"], state["allowed_companies"], settings.chat_top_k
    )
    logger.info(
        "[fetch_one] company_id=%s chunks=%d", result.get("company_id"), len(result["chunks"])
    )
    return {"sub_results": [result]}


def fetch_sub_queries_parallel(sub_queries: list[dict], allowed_companies: list[str]) -> list[dict]:
    """Non-graph counterpart to `fetch_one_node`/`route_to_fetch`, for the
    streaming path (`stream_chat_graph_answer`), which drives nodes by hand
    outside `graph.invoke` to control token-level SSE output and so can't
    use LangGraph's `Send` fan-out. Runs the same per-sub-query fetch, just
    via a plain thread pool instead of graph-managed concurrency."""
    top_k = settings.chat_top_k
    logger.info(
        "[fetch] input sub_queries=%d allowed_companies=%s top_k=%d",
        len(sub_queries),
        allowed_companies,
        top_k,
    )

    if len(sub_queries) == 1:
        sub_results = [_fetch_for_sub_query(sub_queries[0], allowed_companies, top_k)]
    else:
        with ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
            sub_results = list(
                pool.map(
                    lambda sq: _fetch_for_sub_query(sq, allowed_companies, top_k), sub_queries
                )
            )

    logger.info(
        "[fetch] output sub_results=%d total_chunks=%d",
        len(sub_results),
        sum(len(r["chunks"]) for r in sub_results),
    )
    return sub_results


def build_answer_messages(question: str, chunks: list[dict]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{c['chunk_id']}] ({c.get('company_id')}, "
        f"{c.get('fiscal_quarter')} {c.get('fiscal_year')}, "
        f"{c.get('speaker_name')}): {c['text']}"
        for c in chunks
    )
    prompt = load_prompt("answer").format(no_context_message=NO_CONTEXT_MESSAGE)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def select_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Returns every retrieved chunk, each tagged with whether the model's
    raw answer cited its chunk_id.

    Trusting the model's [chunk_id] markers to *select* which chunks are
    shown (rather than just to flag them) hid retrieved evidence from the
    user whenever the model cited the wrong neighboring chunk -- e.g. a
    fact split across a chunk boundary can get attributed to the chunk on
    the wrong side of it. Returning the full set keeps the real supporting
    chunk visible even when the model's own citation is off.
    """
    return [{**c, "cited": c["chunk_id"] in answer} for c in chunks]


_CITATION_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?])")
_CITATION_SPACE_RE = re.compile(r"[ \t]{2,}")


def strip_citation_markers(answer: str, chunks: list[dict]) -> str:
    """Removes inline [chunk_id] citation markers from answer text.

    Citations are surfaced separately in the UI as clickable source chips
    (built from `select_citations`, which must run on the *unstripped*
    answer since it looks for chunk ids in the text) -- leaving the raw
    ids inline in the prose as well is redundant and leaks internal
    identifiers to the user.
    """
    cleaned = answer
    for c in chunks:
        cleaned = cleaned.replace(f"[{c['chunk_id']}]", "")
    cleaned = _CITATION_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _CITATION_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def finalize_answer(answer: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Cleans the raw model answer and decides which chunks to surface as
    sources.

    The answer prompt instructs the model to reply with exactly
    `NO_CONTEXT_MESSAGE` when the retrieved chunks don't actually answer the
    question. Retrieval can still return chunks in that case (they were
    similar enough to be fetched, just not relevant) -- without this check
    those chunks would be shown as "sources" for an answer that isn't
    grounded in them at all.
    """
    citations = select_citations(answer, chunks)
    cleaned = strip_citation_markers(answer, chunks)
    if cleaned == NO_CONTEXT_MESSAGE:
        return cleaned, []
    return cleaned, citations


_LLM_ERROR_MESSAGE = "I ran into an error generating an answer. Please try again."


def finalize_sub_answer(sub_result: dict, raw_answer: str) -> dict:
    """Cleans one sub-query's raw model answer into a `{status, ...}` record
    that `combine_sub_answers` can merge across sub-queries. status is
    "ok" (answer + citations) or "no_context" (the model decided its
    context didn't actually cover this sub-question, per `finalize_answer`)."""
    cleaned, citations = finalize_answer(raw_answer, sub_result["chunks"])
    if cleaned == NO_CONTEXT_MESSAGE:
        return {"status": "no_context", "company_id": sub_result.get("company_id")}
    return {
        "status": "ok",
        "company_id": sub_result.get("company_id"),
        "answer": cleaned,
        "citations": citations,
    }


def _answer_for_sub_result(sub_result: dict) -> dict:
    chunks = sub_result["chunks"]
    if not chunks:
        # No chunks means either this company wasn't authorized or nothing
        # relevant was found -- either way, silently contribute nothing to
        # the combined answer rather than mentioning it (never confirm or
        # deny which companies exist/are restricted).
        return {"status": "no_context", "company_id": sub_result.get("company_id")}

    messages = build_answer_messages(sub_result["question"], chunks)
    try:
        raw_answer = chat_completion(messages, temperature=settings.chat_temperature).strip()
    except Exception:
        logger.exception(
            "[build_answer] LLM call failed for company_id=%s", sub_result.get("company_id")
        )
        return {"status": "error", "company_id": sub_result.get("company_id")}
    return finalize_sub_answer(sub_result, raw_answer)


def combine_sub_answers(answered: list[dict]) -> tuple[str, list[dict]]:
    """Merges per-sub-query answer records into one final answer + citation
    list. A single successful sub-answer is returned as-is (no heading, so a
    plain single-company question reads exactly as it did before
    decomposition existed); multiple are stitched into headed sections."""
    ok = [a for a in answered if a["status"] == "ok"]
    if not ok:
        if any(a["status"] == "error" for a in answered):
            return _LLM_ERROR_MESSAGE, []
        return NO_CONTEXT_MESSAGE, []

    if len(ok) == 1:
        return ok[0]["answer"], ok[0]["citations"]

    name_by_id = dict(
        zip(
            (a["company_id"] for a in ok if a["company_id"]),
            get_company_display_names([a["company_id"] for a in ok if a["company_id"]]),
        )
    )
    sections: list[str] = []
    citations: list[dict] = []
    for a in ok:
        heading = name_by_id.get(a["company_id"])
        sections.append(f"**{heading}**\n{a['answer']}" if heading else a["answer"])
        citations.extend(a["citations"])
    return "\n\n".join(sections), citations


@traceable(name="answer_one", run_type="chain", client=observability.langsmith_client)
def answer_one_node(state: AnswerOneInput) -> dict:
    """Answer branch for exactly one sub_result, dispatched via `Send` from
    `route_to_answer` once `sub_results` has fully merged -- mirrors
    `fetch_one_node`'s fan-out/reduce shape, feeding `answered`."""
    result = _answer_for_sub_result(state["sub_result"])
    logger.info(
        "[answer_one] company_id=%s status=%s", result.get("company_id"), result["status"]
    )
    return {"answered": [result]}


@traceable(name="combine_answer", run_type="chain", client=observability.langsmith_client)
def combine_answer_node(state: ChatState) -> dict:
    """Fan-in step after `answer_one`: stitches the per-company answer
    records into one final answer/citation list (`combine_sub_answers`) and
    unions all retrieved chunks for the guardrail's groundedness check."""
    sub_results = state.get("sub_results") or []
    answered = state.get("answered") or []
    logger.info("[combine_answer] input sub_queries=%d", len(sub_results))
    if not sub_results:
        logger.info("[combine_answer] output no context available")
        return {"answer": NO_CONTEXT_MESSAGE, "citations": [], "chunks": []}

    answer, citations = combine_sub_answers(answered)
    chunks = [c for r in sub_results for c in r["chunks"]]
    logger.info(
        "[combine_answer] output answer_len=%d citations=%d chunks=%d",
        len(answer),
        len(citations),
        len(chunks),
    )
    return {"answer": answer, "citations": citations, "chunks": chunks}


@traceable(name="guardrail", run_type="tool", client=observability.langsmith_client)
def guardrail_verdict(answer: str, chunks: list[dict]) -> tuple[bool, str]:
    """Returns (passed, reason). Fails open on LLM/network errors and when
    the guardrail is disabled or there's no context to check groundedness
    against -- it verifies answer *quality*, not access control (that's
    already enforced deterministically in `fetch_one_node`)."""
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
