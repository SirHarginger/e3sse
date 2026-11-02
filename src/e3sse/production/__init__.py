"""Production orchestration of the E3-SSE pipeline: stage graph, scheduling, provenance.

The orchestrator is thin.  It resolves the stage graph, checks evidence, chooses
resources from measured profiles, and calls the tested stage runners.  It never
reimplements a scientific algorithm.  Scientific computation stays in the existing
runners under ``scripts/`` and their libraries under ``e3sse.*``.
"""
