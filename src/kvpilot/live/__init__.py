"""Logic for a REAL running session, as opposed to kvpilot.replay's trace simulation.

Everything here is pure and GPU/vLLM-independent: it is unit tested on any
machine. A real integration (see kvpilot.integrations) supplies the actual
measured numbers; nothing in this package fabricates them.
"""
