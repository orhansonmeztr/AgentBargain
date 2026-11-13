"""Shared utilities for model definitions."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)
