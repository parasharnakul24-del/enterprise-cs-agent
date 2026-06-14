# src/agents.py
# FlowSync CS Agent — Intent Classifier + Graph
# Stack: LangGraph + Claude Haiku (classifier) + Claude Sonnet (responder)

import os
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

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
# BLOCK 3 — INTENT CLASSIFIER NODE
# ─────────────────────────────────────────────
def classify_intent(state: AgentState) -> dict:
    """
    Classify customer message into one of 4 intents.
    Uses Claude Haiku — fast and cheap for routing decisions.
    Returns intent as structured output, never relies on LLM confidence.
    CCA-F: programmatic enforcement via allowed_intents check.
    """
    # Get the latest customer message
    last_message = state["messages"][-1].content

    # System prompt — strict output enforcement
    system = SystemMessage(content="""You are an intent classifier for a B2B SaaS customer service system.

Classify the customer message into EXACTLY one of these categories:
- BILLING: invoices, payments, subscriptions, pricing questions
- TECHNICAL: bugs, errors, how-to, product features, integrations  
- SALES: upgrades, new features, demos, purchasing decisions
- GENERAL: greetings, account info, anything else

Respond with ONLY the category name. Nothing else. No explanation.""")

    # Invoke Haiku
    response = classifier_llm.invoke([system, HumanMessage(content=last_message)])
    raw_intent = response.content.strip().upper()

    # CCA-F: programmatic enforcement — never trust LLM output blindly
    allowed_intents = {"BILLING", "TECHNICAL", "SALES", "GENERAL"}
    intent = raw_intent if raw_intent in allowed_intents else "GENERAL"

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

    if intent == "TECHNICAL":
        return "lookup_kb"
    elif intent == "BILLING":
        return "check_order"
    elif intent == "SALES":
        return "escalate"
    else:
        return "respond"

# ─────────────────────────────────────────────
# BLOCK 5 — TOOL NODES (stubs for now)
# ─────────────────────────────────────────────
def lookup_kb(state: AgentState) -> dict:
    """Look up knowledge base — ChromaDB integration (Week 4 Thu)."""
    # Stub — will wire ChromaDB Thursday
    return {"messages": [SystemMessage(content="[KB LOOKUP STUB] Technical content here.")]}

def check_order(state: AgentState) -> dict:
    """Check order/subscription status."""
    # Stub — will wire real logic Thursday
    return {"messages": [SystemMessage(content="[ORDER CHECK STUB] Billing info here.")]}

def escalate(state: AgentState) -> dict:
    """Escalate to human agent for Sales queries."""
    return {"messages": [SystemMessage(content="[ESCALATE STUB] Routing to sales team.")]}

# ─────────────────────────────────────────────
# BLOCK 6 — RESPONDER NODE
# ─────────────────────────────────────────────
def respond(state: AgentState) -> dict:
    """
    Final response generation using Claude Sonnet.
    Filters out SystemMessages from state to avoid multiple system message error.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    system = SystemMessage(content="""You are a helpful enterprise customer service agent for FlowSync.
Be professional, concise and helpful.
If context from knowledge base or order system is provided, use it in your response.""")

    # Filter out SystemMessages from state — only keep Human, AI, Tool messages
    filtered_messages = [
        m for m in state["messages"]
        if not isinstance(m, SystemMessage)
    ]

    response = responder_llm.invoke([system] + filtered_messages)
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

    # Add all nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("lookup_kb", lookup_kb)
    builder.add_node("check_order", check_order)
    builder.add_node("escalate", escalate)
    builder.add_node("respond", respond)

    # Entry point
    builder.add_edge(START, "classify_intent")

    # Conditional routing after classification
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

    # All tool nodes flow to respond
    builder.add_edge("lookup_kb", "respond")
    builder.add_edge("check_order", "respond")
    builder.add_edge("escalate", "respond")

    # Respond goes to END
    builder.add_edge("respond", END)

    # Compile with memory — one thread per customer session
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# Build the graph
graph = build_graph()