"""
Strategy registry and @strategy decorator.

Provides a global registry for discovering and instantiating strategies
by name, enabling the benchmark runner and API to enumerate all available
strategies without hardcoding them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_bargain.strategies.base import NegotiationStrategy

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[NegotiationStrategy]] = {}


class UnknownStrategyError(KeyError):
    """No strategy is registered under the requested name.

    A ``KeyError`` subclass so existing ``except KeyError`` callers keep
    working, with a readable message: ``str(KeyError("x"))`` is ``"'x'"``.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(name)
        self.name = name
        self.available = available

    def __str__(self) -> str:
        return f"Strategy {self.name!r} not found. Available: [{', '.join(self.available)}]"


class StrategyRegistry:
    """Global singleton registry of all registered negotiation strategies."""

    @staticmethod
    def register(name: str, cls: type[NegotiationStrategy]) -> None:
        """Register a strategy class under the given name."""
        if name in _REGISTRY:
            logger.warning("Strategy %r is being overwritten in the registry.", name)
        cls.name = name
        _REGISTRY[name] = cls
        logger.debug("Registered strategy: %r -> %s", name, cls.__name__)

    @staticmethod
    def get(name: str) -> type[NegotiationStrategy]:
        """Retrieve a strategy class by name.

        Raises:
            UnknownStrategyError: If the strategy name is not registered.
        """
        if name not in _REGISTRY:
            raise UnknownStrategyError(name, sorted(_REGISTRY))
        return _REGISTRY[name]

    @staticmethod
    def instantiate(name: str, **kwargs) -> NegotiationStrategy:
        """Instantiate a strategy by name, passing kwargs to its constructor."""
        cls = StrategyRegistry.get(name)
        return cls(**kwargs)

    @staticmethod
    def list_all() -> list[dict]:
        """Return metadata for all registered strategies (for the API/dashboard)."""
        return [
            {
                "name": name,
                "class": cls.__name__,
                "description": getattr(cls, "description", ""),
            }
            for name, cls in sorted(_REGISTRY.items())
        ]

    @staticmethod
    def names() -> list[str]:
        """Return a sorted list of all registered strategy names."""
        return sorted(_REGISTRY.keys())


def strategy(name: str):
    """Class decorator to register a NegotiationStrategy subclass.

    Usage::

        @strategy("anchoring")
        class AnchoringStrategy(NegotiationStrategy):
            description = "Start with an extreme anchor price."
            ...

    The decorated class will be available globally via StrategyRegistry.get("anchoring").
    """

    def decorator(cls: type[NegotiationStrategy]) -> type[NegotiationStrategy]:
        StrategyRegistry.register(name, cls)
        return cls

    return decorator
