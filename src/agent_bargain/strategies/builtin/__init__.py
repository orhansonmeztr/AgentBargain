"""Built-in negotiation strategies for AgentBargain.

Importing this package registers every built-in strategy with the global
``StrategyRegistry``. The ``agent_bargain.strategies`` package imports it
eagerly so strategies are always discoverable — including in the API process,
which never imports the benchmark CLI.
"""

from agent_bargain.strategies.builtin.aggressive import AggressiveStrategy
from agent_bargain.strategies.builtin.anchoring import AnchoringStrategy
from agent_bargain.strategies.builtin.concession import ConcessionStrategy
from agent_bargain.strategies.builtin.cooperative import CooperativeStrategy
from agent_bargain.strategies.builtin.tit_for_tat import TitForTatStrategy

__all__ = [
    "AggressiveStrategy",
    "AnchoringStrategy",
    "ConcessionStrategy",
    "CooperativeStrategy",
    "TitForTatStrategy",
]
