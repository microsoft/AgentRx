"""AgentRx judge package.

Public entry points are exposed via submodules — import them explicitly:

    from agentrx.judge import judge  # main LLM-as-a-judge module

This package intentionally exposes no names from ``__init__`` so the
public API surface stays explicit and grep-able. Earlier revisions of
this file accidentally contained a byte-for-byte duplicate of
``agentrx/ir/trajectory_ir.py``; removing it eliminates the
hidden-name-shadowing footgun (e.g. ``from agentrx.judge import Event``
silently importing IR symbols).
"""
