"""Seller agent node logic."""

from agent_bargain.agents.base import create_negotiation_node
from agent_bargain.llm.prompts import SELLER_PROMPT

seller_node = create_negotiation_node(
    prompt_template=SELLER_PROMPT,
    role="seller",
    budget_key="min_price",
)
