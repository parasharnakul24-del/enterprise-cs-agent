# tests/test_classifier.py
# FlowSync CS Agent — Intent Classification Tests
# Target: 5/5 correct

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import graph

def run_classification(message: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config
    )
    return result.get("intent", "UNKNOWN")

def test_billing_intent():
    intent = run_classification(
        "My invoice for last month seems incorrect, I was charged twice",
        thread_id="test-billing-001"
    )
    assert intent == "BILLING", f"Expected BILLING, got {intent}"
    print(f"✅ BILLING: {intent}")

def test_technical_intent():
    intent = run_classification(
        "I cannot integrate your API with my CRM, getting 401 errors",
        thread_id="test-technical-001"
    )
    assert intent == "TECHNICAL", f"Expected TECHNICAL, got {intent}"
    print(f"✅ TECHNICAL: {intent}")

def test_sales_intent():
    intent = run_classification(
        "I want to upgrade my plan and add more users to my account",
        thread_id="test-sales-001"
    )
    assert intent == "SALES", f"Expected SALES, got {intent}"
    print(f"✅ SALES: {intent}")

def test_general_intent():
    intent = run_classification(
        "Hi, I need help with my account settings",
        thread_id="test-general-001"
    )
    assert intent == "GENERAL", f"Expected GENERAL, got {intent}"
    print(f"✅ GENERAL: {intent}")

def test_billing_edge_case():
    intent = run_classification(
        "I want to cancel my subscription immediately",
        thread_id="test-billing-002"
    )
    assert intent == "BILLING", f"Expected BILLING, got {intent}"
    print(f"✅ BILLING edge case: {intent}")

if __name__ == "__main__":
    print("\n🧪 Running FlowSync Intent Classification Tests...\n")
    tests = [
        test_billing_intent,
        test_technical_intent,
        test_sales_intent,
        test_general_intent,
        test_billing_edge_case
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
    print(f"\n{'='*40}")
    print(f"Results: {passed}/5 passed")
    print(f"{'='*40}")
