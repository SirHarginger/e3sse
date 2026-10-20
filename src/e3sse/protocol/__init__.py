"""Protocol-freeze construction for the E3-SSE static studies."""

from __future__ import annotations

from typing import Any

__all__ = ["ProtocolFreezeError", "materialize_protocol"]


def __getattr__(name: str) -> Any:
    # Lazy so lightweight modules (e.g. ``adapters``) import without the data stack.
    if name in __all__:
        from . import freeze

        return getattr(freeze, name)
    raise AttributeError(name)
