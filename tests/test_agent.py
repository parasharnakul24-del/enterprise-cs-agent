# tests/test_agent.py
import requests

BASE_URL = "http://localhost:8000"

TEST_CASES = [
    # BILLING (5)
    {"message": "My invoice is wrong", "expected_intent": "BILLING"},
    {"message": "I want to cancel my subscription", "expected_intent": "BILLING"},
    {"message": "How do I upgrade my plan?", "expected_intent": "SALES"},
    {"message": "What payment methods do you accept?", "expected_intent": "BILLING"},
    {"message": "I need a refund for last month", "expected_intent": "BILLING"},
    # TECHNICAL (5)
    {"message": "I'm getting a 500 error on the API", "expected_intent": "TECHNICAL"},
    {"message": "My webhook is not firing", "expected_intent": "TECHNICAL"},
    {"message": "How do I authenticate with the SDK?", "expected_intent": "TECHNICAL"},
    {"message": "What are your rate limits?", "expected_intent": "TECHNICAL"},
    {"message": "Where are the API docs?", "expected_intent": "TECHNICAL"},
    # SALES (5)
    {"message": "I want to upgrade to the enterprise plan", "expected_intent": "SALES"},
    {"message": "Can I extend my trial period?", "expected_intent": "SALES"},
    {"message": "What integrations do you support?", "expected_intent": "SALES"},
    {"message": "I'd like to book a demo", "expected_intent": "SALES"},
    {"message": "What's the difference between your pricing tiers?", "expected_intent": "SALES"},
    # GENERAL (5)
    {"message": "Hello, I need some help", "expected_intent": "GENERAL"},
    {"message": "What is FlowSync?", "expected_intent": "GENERAL"},
    {"message": "Do you have SOC 2 certification?", "expected_intent": "GENERAL"},
    {"message": "What are your support hours?", "expected_intent": "GENERAL"},
    {"message": "Where are your servers located?", "expected_intent": "GENERAL"},
]


def run_tests():
    passed = 0
    failed = 0
    results = []

    for i, tc in enumerate(TEST_CASES, 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat",
                json={"session_id": f"test-{i}", "message": tc["message"]},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            actual_intent = data["intent"]
            ok = actual_intent == tc["expected_intent"]
            if ok:
                passed += 1
            else:
                failed += 1
            results.append({
                "id": i,
                "message": tc["message"],
                "expected": tc["expected_intent"],
                "actual": actual_intent,
                "pass": ok,
            })
        except Exception as e:
            failed += 1
            results.append({
                "id": i,
                "message": tc["message"],
                "expected": tc["expected_intent"],
                "actual": f"ERROR: {e}",
                "pass": False,
            })

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/20 passed\n")
    for r in results:
        status = "✅" if r["pass"] else "❌"
        print(f"{status} [{r['id']:2d}] {r['message'][:45]:<45} → {r['actual']}")
    print(f"{'='*60}")
    print(f"\nTarget: 18/20. {'✅ PASSED' if passed >= 18 else '❌ BELOW TARGET'}")
    return passed


if __name__ == "__main__":
    run_tests()