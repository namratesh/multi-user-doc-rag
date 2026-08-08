"""Builds and runs the conversational RAG LangGraph.

    classify --continue--> rephrase -> decompose =Send=> fetch_one =Send=> answer_one -> combine_answer -> guardrail -> END
             --greet/deny-> canned_response ---------------------------------------------------------------------------> END

`classify` is UX routing only; access control is enforced deterministically
in `fetch_one_node` via `allowed_companies`, which callers must derive from
a fresh permissions lookup (see `api/routes/chat.py`), not from the graph.

`decompose` splits a question naming several companies into one
sub-question per company. `route_to_fetch`/`route_to_answer` fan each one
out to its own `fetch_one`/`answer_one` run via LangGraph's `Send` API
(dynamic map-reduce: https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
so they execute concurrently and get their own traced runs, then
`ChatState`'s `operator.add` reducers on `sub_results`/`answered` merge the
branches back together once every one of them has finished, before
`combine_answer` stitches the per-company answers into one reply. This way
a question spanning companies the caller has mixed access to is answered
for the ones it does have access to, rather than the whole reply being
withheld because part of it can't be answered.

The streaming path (`stream_chat_graph_answer`) mirrors this same
fetch-then-answer shape but drives nodes by hand outside `graph.invoke`, so
it can't use `Send` -- see its docstring.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langsmith import trace

from src.config import observability
from src.config.logger import get_logger
from src.config.settings import settings
from src.config.users import get_company_display_names
from src.llm.client import stream_chat_completion

from .nodes import (
    GUARDRAIL_FALLBACK_MESSAGE,
    NO_CONTEXT_MESSAGE,
    answer_one_node,
    build_answer_messages,
    canned_response_node,
    classify_node,
    combine_answer_node,
    combine_sub_answers,
    decompose_node,
    fetch_one_node,
    fetch_sub_queries_parallel,
    finalize_sub_answer,
    guardrail_node,
    guardrail_verdict,
    rephrase_node,
)
from .state import ChatState, ConversationTurn

logger = get_logger(__name__)


def _route_after_classify(state: ChatState) -> Literal["continue", "greet", "deny"]:
    return state.get("route", "continue")


def route_to_fetch(state: ChatState) -> list[Send]:
    """Fans `sub_queries` out to parallel `fetch_one` runs. Each `Send`
    carries only that branch's own sub_query + the caller's ACL (see
    `FetchOneInput`), not the full `ChatState` -- LangGraph merges each
    branch's `{"sub_results": [...]}` return back via the reducer once all
    of them finish."""
    return [
        Send("fetch_one", {"sub_query": sq, "allowed_companies": state["allowed_companies"]})
        for sq in state.get("sub_queries") or []
    ]


def route_to_answer(state: ChatState) -> list[Send]:
    """Fans the now-merged `sub_results` out to parallel `answer_one` runs,
    mirroring `route_to_fetch`."""
    return [Send("answer_one", {"sub_result": r}) for r in state.get("sub_results") or []]


@lru_cache(maxsize=1)
def get_chat_graph():
    builder = StateGraph(ChatState)
    builder.add_node("classify", classify_node)
    builder.add_node("canned_response", canned_response_node)
    builder.add_node("rephrase", rephrase_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("fetch_one", fetch_one_node)
    builder.add_node("answer_one", answer_one_node)
    builder.add_node("combine_answer", combine_answer_node)
    builder.add_node("guardrail", guardrail_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"continue": "rephrase", "greet": "canned_response", "deny": "canned_response"},
    )
    builder.add_edge("canned_response", END)
    builder.add_edge("rephrase", "decompose")
    builder.add_conditional_edges("decompose", route_to_fetch, ["fetch_one"])
    builder.add_conditional_edges("fetch_one", route_to_answer, ["answer_one"])
    builder.add_edge("answer_one", "combine_answer")
    builder.add_edge("combine_answer", "guardrail")
    builder.add_edge("guardrail", END)
    return builder.compile()


def run_chat_graph(
    *,
    email: str,
    allowed_companies: list[str],
    conv_id: str,
    question: str,
    history: list[ConversationTurn],
) -> ChatState:
    graph = get_chat_graph()
    initial_state: ChatState = {
        "email": email,
        "allowed_companies": allowed_companies,
        "conv_id": conv_id,
        "question": question,
        "history": history,
    }
    with trace(
        name="run_chat_graph",
        run_type="chain",
        inputs={"question": question},
        metadata={"user_id": email, "session_id": conv_id},
        client=observability.langsmith_client,
    ) as run:
        result = graph.invoke(initial_state)
        run.end(outputs=result)
        return result


def stream_chat_graph_answer(
    *,
    email: str,
    allowed_companies: list[str],
    conv_id: str,
    question: str,
    history: list[ConversationTurn],
) -> Iterator[dict]:
    """Runs the same classify/rephrase/decompose/fetch steps as
    `run_chat_graph`, but streams each sub-answer's LLM call token-by-token
    as `delta` events.

    This calls node functions directly instead of `graph.invoke`, so it
    can't use `route_to_fetch`'s `Send`-based fan-out (`Send` dispatches
    happen inside the compiled graph's own executor). It gets its
    concurrent retrieval from `fetch_sub_queries_parallel` (a plain thread
    pool) instead, and streams the per-company answer-generation calls
    sequentially -- a single SSE stream can't interleave multiple token
    streams into one coherent response.

    The guardrail still needs the *complete* answer to judge groundedness,
    so it only runs after streaming finishes; the trailing `done` event
    carries the authoritative answer/citations. If the guardrail rejects the
    answer, `done.answer` will differ from the concatenated `delta` text --
    callers must treat `done` as the source of truth and replace whatever
    was rendered from deltas, not just append to it.
    """
    with trace(
        name="stream_chat_graph_answer",
        run_type="chain",
        inputs={"question": question},
        metadata={"user_id": email, "session_id": conv_id},
        client=observability.langsmith_client,
    ) as run:
        state: ChatState = {
            "email": email,
            "allowed_companies": allowed_companies,
            "conv_id": conv_id,
            "question": question,
            "history": history,
        }
        state.update(classify_node(state))
        route = state.get("route", "continue")

        if route in ("greet", "deny"):
            state.update(canned_response_node(state))
            event = {
                "type": "done",
                "route": route,
                "answer": state["final_answer"],
                "citations": [],
                "guardrail_passed": True,
            }
            run.end(outputs=event)
            yield event
            return

        state.update(rephrase_node(state))
        state.update(decompose_node(state))
        sub_results = fetch_sub_queries_parallel(
            state.get("sub_queries") or [], allowed_companies
        )
        chunk_bearing = [r for r in sub_results if r["chunks"]]

        if not chunk_bearing:
            event = {
                "type": "done",
                "route": route,
                "answer": NO_CONTEXT_MESSAGE,
                "citations": [],
                "guardrail_passed": True,
            }
            run.end(outputs=event)
            yield event
            return

        # A multi-company question streams one section per company in
        # sequence (a single SSE stream can't interleave several LLM
        # streams), with a heading per section once there's more than one.
        # Sections are streamed provisionally as they're generated; the
        # `done` event below carries the authoritative combined answer,
        # exactly like the guardrail-rejection case already did before
        # decomposition existed.
        multi = len(chunk_bearing) > 1
        answered: list[dict] = []
        for sub in chunk_bearing:
            if multi:
                heading = (
                    get_company_display_names([sub["company_id"]])[0]
                    if sub["company_id"]
                    else None
                )
                if heading:
                    yield {"type": "delta", "text": f"**{heading}**\n"}

            messages = build_answer_messages(sub["question"], sub["chunks"])
            parts: list[str] = []
            try:
                for delta in stream_chat_completion(
                    messages, temperature=settings.chat_temperature
                ):
                    parts.append(delta)
                    yield {"type": "delta", "text": delta}
            except Exception:
                logger.exception(
                    "[stream_chat_graph_answer] answer stream failed for company_id=%s",
                    sub.get("company_id"),
                )
                answered.append({"status": "error", "company_id": sub.get("company_id")})
                continue

            if multi:
                yield {"type": "delta", "text": "\n\n"}
            raw_answer = "".join(parts).strip()
            answered.append(finalize_sub_answer(sub, raw_answer))

        answer, citations = combine_sub_answers(answered)
        all_chunks = [c for r in chunk_bearing for c in r["chunks"]]
        passed, reason = guardrail_verdict(answer, all_chunks)

        if passed:
            event = {
                "type": "done",
                "route": route,
                "answer": answer,
                "citations": citations,
                "guardrail_passed": True,
            }
        else:
            logger.warning("[stream_chat_graph_answer] guardrail rejected answer reason=%s", reason)
            event = {
                "type": "done",
                "route": route,
                "answer": GUARDRAIL_FALLBACK_MESSAGE,
                "citations": [],
                "guardrail_passed": False,
            }
        run.end(outputs=event)
        yield event
