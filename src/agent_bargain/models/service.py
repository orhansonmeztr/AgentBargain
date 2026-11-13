"""Service models inspired by UCP."""

from pydantic import BaseModel, Field


class ServiceCategory(BaseModel):
    """Category classification for a service."""

    id: str
    name: str


class ServiceListing(BaseModel):
    """UCP-inspired service listing definition."""

    service_id: str = Field(..., description="Unique identifier for the service")
    seller_id: str = Field(..., description="ID of the seller agent")
    title: str = Field(..., description="Title of the service")
    description: str = Field(default="No description provided.", description="Detailed description")
    category: ServiceCategory = Field(
        default_factory=lambda: ServiceCategory(id="uncategorized", name="Uncategorized"),
        description="Service category",
    )
    base_price: float = Field(..., description="Advertised asking price")
    #: The seller's minimum acceptable price — their walk-away point. Set by the
    #: seller at registration time and never by a negotiating counterparty;
    #: otherwise a buyer could declare ``floor_price=0`` and win every deal.
    floor_price: float = Field(
        default=0.0,
        ge=0.0,
        description="Seller's minimum acceptable price (hidden from the buyer)",
    )
    sla_terms: str = Field(..., description="Service Level Agreement terms")
