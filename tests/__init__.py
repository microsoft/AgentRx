"""Acceptance tests for AgentRx public-release invariants.

Each test file fences a specific contract that the paper reproduction relies
on. They are intentionally narrow: they exercise the contract directly, not
end-to-end pipelines, so they run in seconds without LLM calls.
"""
