"""
AgentBargain Strategy Framework.

Provides the base classes and plugin registry for custom negotiation strategies.
Users can implement and register their own strategies using the @strategy decorator.

Example:
    from agent_bargain.strategies import strategy, NegotiationStrategy, NegotiationContext, Decision

    @strategy("my_strategy")
    class MyStrategy(NegotiationStrategy):
        def decide(self, context: NegotiationContext) -> Decision:
            # Your logic here
            return Decision(action="counter", price=context.current_price * 0.9)
"""

# Importing the builtin package registers every built-in strategy. This must
# happen here (not only in the benchmark CLI) so the API process can resolve
# strategy names for engine="strategy" requests.
import agent_bargain.strategies.builtin  # noqa: F401
from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.registry import StrategyRegistry, UnknownStrategyError, strategy

__all__ = [
    "Decision",
    "NegotiationContext",
    "NegotiationStrategy",
    "StrategyRegistry",
    "UnknownStrategyError",
    "strategy",
]
