"""Buyer agent node logic."""

from agent_bargain.agents.base import create_negotiation_node
from agent_bargain.llm.prompts import BUYER_PROMPT

buyer_node = create_negotiation_node(
    prompt_template=BUYER_PROMPT,
    role="buyer",
    budget_key="max_budget",
)
