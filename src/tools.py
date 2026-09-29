# src/tools.py
# FlowSync CS Agent — 3 Tools

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ─────────────────────────────────────────────
# TOOL 1 — KNOWLEDGE BASE LOOKUP
# ─────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _retrieve_with_retry(query: str) -> str:
    """Pinecone hybrid-search call — retries on transient failures."""
    from src.knowledge_base import retrieve
    return retrieve(query)


@tool
def lookup_knowledge_base(query: str) -> str:
    """
    Semantic search over FlowSync FAQ knowledge base using hybrid search
    (Pinecone dense + BM25).
    Use this for technical questions, how-to queries, and product feature questions.
    
    Args:
        query: The customer's question to search for
        
    Returns:
        Top 3 matching FAQ chunks as context
    """
    return _retrieve_with_retry(query)


# ─────────────────────────────────────────────
# TOOL 2 — ESCALATE TO HUMAN
# ─────────────────────────────────────────────
@tool
def escalate_to_human(reason: str, customer_id: str = "unknown") -> str:
    """
    Escalate the customer conversation to a human agent.
    Use this for sales queries, complex issues, or when customer is frustrated.
    
    Args:
        reason: Why this conversation needs human attention
        customer_id: The customer's ID for the human agent
        
    Returns:
        Confirmation that escalation has been initiated
    """
    # No external call — pure string formatting. No retry needed.
    ticket_id = f"ESC-{customer_id[:4].upper()}-001"
    return (
        f"Escalation initiated. Ticket ID: {ticket_id}. "
        f"Reason: {reason}. "
        f"A human agent will contact you within 2 business hours."
    )


# ─────────────────────────────────────────────
# TOOL 3 — CHECK ORDER STATUS
# ─────────────────────────────────────────────
@tool
def check_order_status(customer_id: str) -> str:
    """
    Check the subscription or order status for a customer.
    Use this for billing queries, invoice questions, and subscription issues.
    
    Args:
        customer_id: The customer's account ID
        
    Returns:
        Current subscription and billing status
    """
    # Mock local dict lookup — no external call. No retry needed.
    mock_data = {
        "CUST001": {
            "plan": "Enterprise",
            "status": "Active",
            "next_billing": "2025-07-01",
            "amount": "$2,400/month"
        },
        "CUST002": {
            "plan": "Professional",
            "status": "Past Due",
            "next_billing": "Overdue since 2025-05-15",
            "amount": "$800/month"
        }
    }

    customer_id = customer_id.strip().upper()
    
    if customer_id in mock_data:
        data = mock_data[customer_id]
        return (
            f"Account: {customer_id} | "
            f"Plan: {data['plan']} | "
            f"Status: {data['status']} | "
            f"Next Billing: {data['next_billing']} | "
            f"Amount: {data['amount']}"
        )
    else:
        return f"No account found for ID: {customer_id}. Please verify the customer ID."


# ─────────────────────────────────────────────
# TOOLS LIST
# ─────────────────────────────────────────────
tools = [lookup_knowledge_base, escalate_to_human, check_order_status]