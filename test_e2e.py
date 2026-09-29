from src.agents import graph
from langchain_core.messages import HumanMessage

def test_e2e(message, session_id):
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)], "customer_id": "CUST-001"},
        config
    )
    print(f"Message:  {message}")
    print(f"Intent:   {result['intent']}")
    print(f"Response: {result['messages'][-1].content}")
    print("---")

test_e2e("My invoice is wrong", "test-billing")
test_e2e("I'm getting a 500 error on login", "test-technical")
test_e2e("I want to upgrade my plan", "test-sales")
test_e2e("Hello, I need help", "test-general")
test_e2e("Cancel my subscription immediately", "test-billing-2")