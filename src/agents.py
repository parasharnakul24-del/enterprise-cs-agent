# src/agents.py
# FlowSync CS Agent — Intent Classifier + Graph
# Stack: LangGraph + Claude Haiku (classifier) + Claude Sonnet (responder)

import os
import logging
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ─────────────────────────────────────────────
# BLOCK 0 — STRUCTURED LOGGING SETUP
# ─────────────────────────────────────────────
# One line per event: timestamp, level, message.
# This is what makes a LangSmith trace legible later — you already
# know what "intent classified" or "tool called" means because you
# logged it here first.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# BLOCK 1 — STATE DEFINITION
# ─────────────────────────────────────────────
class AgentState(MessagesState):
    """
    Extends MessagesState with custom fields.
    MessagesState gives us 'messages' key with add_messages reducer.
    We add intent and customer_id on top.
    """
    intent: str          # BILLING / TECHNICAL / SALES / GENERAL
    customer_id: str     # one thread per customer session

# ─────────────────────────────────────────────
# BLOCK 2 — MODELS
# ─────────────────────────────────────────────
# Haiku for fast cheap classification
classifier_llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=100,
    temperature=0
)

# Sonnet for high quality customer responses
responder_llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    temperature=0.3
)

# ─────────────────────────────────────────────
# BLOCK 2B — RETRY-WRAPPED API CALLS (Step 3, unchanged)
# ─────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_classifier(messages):
    """Claude Haiku call — retries on transient failures."""
    return classifier_llm.invoke(messages)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_responder(messages):
    """Claude Sonnet call — retries on transient failures."""
    return responder_llm.invoke(messages)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_retrieve(query: str, n_results: int = 3):
    """Pinecone hybrid-search retrieval call — retries on transient failures."""
    from src.knowledge_base import retrieve
    return retrieve(query, n_results=n_results)


# ─────────────────────────────────────────────
# BLOCK 3 — INTENT CLASSIFIER NODE
# ─────────────────────────────────────────────
def classify_intent(state: AgentState) -> dict:
    """
    Classify customer message into one of 4 intents.
    Uses Claude Haiku — fast and cheap for routing decisions.
    Returns intent as structured output, never relies on LLM confidence.
    CCA-F: programmatic enforcement via allowed_intents check.
    """
    last_message = state["messages"][-1].content
    customer_id = state.get("customer_id", "UNKNOWN")

    system = SystemMessage(content="""You are an intent classifier for a B2B SaaS customer service system.

Classify the customer message into EXACTLY one of these categories:
- BILLING: invoices, payments, subscriptions, pricing questions
- TECHNICAL: bugs, errors, how-to, product features, integrations
- SALES: upgrades, new features, demos, purchasing decisions
- GENERAL: greetings, account info, anything else

Respond with ONLY the category name. Nothing else. No explanation.""")

    response = _call_classifier([system, HumanMessage(content=last_message)])
    raw_intent = response.content.strip().upper()

    # CCA-F: programmatic enforcement — never trust LLM output blindly
    allowed_intents = {"BILLING", "TECHNICAL", "SALES", "GENERAL"}
    intent = raw_intent if raw_intent in allowed_intents else "GENERAL"

    # Structured log — one line, everything you'd want to grep for later
    usage = getattr(response, "usage_metadata", None) or {}
    logger.info(
        "INTENT_CLASSIFIED customer_id=%s intent=%s raw_model_output=%s "
        "input_tokens=%s output_tokens=%s",
        customer_id, intent, raw_intent,
        usage.get("input_tokens", "n/a"), usage.get("output_tokens", "n/a"),
    )

    return {"intent": intent}

# ─────────────────────────────────────────────
# BLOCK 4 — ROUTER (conditional edge logic)
# ─────────────────────────────────────────────
def route_intent(state: AgentState) -> Literal["lookup_kb", "escalate", "check_order", "respond"]:
    """
    Routes to the right node based on classified intent.
    This is a conditional edge function — returns node name as string.
    """
    intent = state.get("intent", "GENERAL")
    customer_id = state.get("customer_id", "UNKNOWN")

    if intent == "TECHNICAL":
        next_node = "lookup_kb"
    elif intent == "BILLING":
        next_node = "check_order"
    elif intent == "SALES":
        next_node = "escalate"
    else:
        next_node = "respond"

    logger.info("ROUTED customer_id=%s intent=%s -> node=%s", customer_id, intent, next_node)
    return next_node

# ─────────────────────────────────────────────
# BLOCK 5 — TOOL NODES
# ─────────────────────────────────────────────
def lookup_kb(state: AgentState) -> dict:
    """
    retrieve() already returns one formatted string (chunks joined with
    '---' separators) — NOT a list. Assign it directly.
    """
    last_message = state["messages"][-1].content
    customer_id = state.get("customer_id", "UNKNOWN")

    context = _call_retrieve(last_message, n_results=3)

    logger.info(
        "TOOL_CALLED customer_id=%s tool=lookup_kb query=%r context_chars=%d",
        customer_id, last_message, len(context),
    )
    return {"messages": [SystemMessage(content=f"[KB CONTEXT]\n{context}")]}

from src.tools import check_order_status

def check_order(state: AgentState) -> dict:
    customer_id = state.get("customer_id", "UNKNOWN")
    result = check_order_status.invoke({"customer_id": customer_id})

    logger.info("TOOL_CALLED customer_id=%s tool=check_order_status", customer_id)
    return {"messages": [SystemMessage(content=f"[ORDER STATUS]\n{result}")]}

def escalate(state: AgentState) -> dict:
    """Escalate to human agent for Sales queries."""
    customer_id = state.get("customer_id", "UNKNOWN")
    logger.info("TOOL_CALLED customer_id=%s tool=escalate", customer_id)
    return {"messages": [SystemMessage(content="[ESCALATE STUB] Routing to sales team.")]}

# ─────────────────────────────────────────────
# BLOCK 6 — RESPONDER NODE
# ─────────────────────────────────────────────
def respond(state: AgentState) -> dict:
    """
    Final response generation using Claude Sonnet.
    Filters out SystemMessages from state to avoid multiple system message error.
    """
    customer_id = state.get("customer_id", "UNKNOWN")
    intent = state.get("intent", "GENERAL")

    system = SystemMessage(content="""You are a helpful enterprise customer service agent for FlowSync.
Be professional, concise and helpful.
If context from knowledge base or order system is provided, use it in your response.""")

    # Filter out SystemMessages from state — only keep Human, AI, Tool messages
    filtered_messages = [
        m for m in state["messages"]
        if not isinstance(m, SystemMessage)
    ]

    response = _call_responder([system] + filtered_messages)

    usage = getattr(response, "usage_metadata", None) or {}
    logger.info(
        "RESPONSE_GENERATED customer_id=%s intent=%s input_tokens=%s output_tokens=%s",
        customer_id, intent,
        usage.get("input_tokens", "n/a"), usage.get("output_tokens", "n/a"),
    )

    return {"messages": [response]}

# ─────────────────────────────────────────────
# BLOCK 7 — GRAPH ASSEMBLY
# ─────────────────────────────────────────────
def build_graph():
    """
    Assembles the full LangGraph StateGraph.
    LangGraph style: add_edge(START,...), conditional edges, MemorySaver.
    """
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("lookup_kb", lookup_kb)
    builder.add_node("check_order", check_order)
    builder.add_node("escalate", escalate)
    builder.add_node("respond", respond)

    builder.add_edge(START, "classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "lookup_kb": "lookup_kb",
            "check_order": "check_order",
            "escalate": "escalate",
            "respond": "respond"
        }
    )

    builder.add_edge("lookup_kb", "respond")
    builder.add_edge("check_order", "respond")
    builder.add_edge("escalate", "respond")

    builder.add_edge("respond", END)

    memory = MemorySaver()
    return builder.compile()

# Build the graph
graph = build_graph()