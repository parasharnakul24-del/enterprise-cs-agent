# Model Comparison: Claude Haiku vs GPT-4o
## Structured Output Reliability — JSON Extraction with Retry

**Date:** 2026-06-15
**Test:** 10 enterprise CS scenarios, JSON schema validation with up to 3 retries
**Pattern:** Validation retry loop — error fed back to model on each failed attempt

## Results

| Model | Total Attempts | Tests Won |
|---|---|---|
| Claude Haiku | 10 | 10/10 |
| GPT-4o | 10 | 0/10 |

## Per-Test Breakdown

| # | Preview | Claude | GPT-4o |
|---|---|---|---|
| 1 | My name is Rahul Sharma, I work at TechCorp, and I... | OK 1 | OK 1 |
| 2 | We have 250 employees and need enterprise pricing.... | OK 1 | OK 1 |
| 3 | I want to cancel my Professional plan subscription... | OK 1 | OK 1 |
| 4 | Getting error code 429 when calling your API. Rate... | OK 1 | OK 1 |
| 5 | Our Salesforce integration stopped working after w... | OK 1 | OK 1 |
| 6 | Need 5 additional user seats on top of our current... | OK 1 | OK 1 |
| 7 | Contact person is Sarah Chen, email sarah@acmecorp... | OK 1 | OK 1 |
| 8 | We process about 50,000 API calls per day and need... | OK 1 | OK 1 |
| 9 | Requesting a refund for last 3 months. Account ID ... | OK 1 | OK 1 |
| 10 | Our security team needs SOC2 Type II report and GD... | OK 1 | OK 1 |

## Architectural Takeaway

**Winner: Claude Haiku** on structured output reliability across 10 enterprise scenarios.

Key observations:
- Both models handle simple flat schemas in 1 attempt
- Complex schemas (arrays, enums, nested types) reveal reliability differences
- The retry loop pattern is essential for production — never assume first attempt succeeds
- Feeding the exact error back to the model is more effective than a generic retry prompt

## SE Interview Framing

This pattern answers: *'How do you ensure reliable structured output from LLMs in production?'*

Answer: Validate against JSON schema, catch the error, feed it back to the model with context, retry up to N times. Log retry counts per model — this becomes your reliability benchmark data.