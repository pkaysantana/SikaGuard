"""Exact monetary amounts held as integer minor units (pence, cents)."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Only currencies with two decimal places, so `amount_minor` is always hundredths.
CURRENCY_SYMBOLS = {"GBP": "£", "EUR": "€", "USD": "$"}


class CurrencyMismatchError(ValueError):
    """Raised when amounts in different currencies are added or compared."""


class Money(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    amount_minor: int = Field(ge=0)
    currency: str

    @field_validator("currency")
    @classmethod
    def _supported_currency(cls, value: str) -> str:
        if value not in CURRENCY_SYMBOLS:
            raise ValueError(f"unsupported currency {value!r}; expected one of {sorted(CURRENCY_SYMBOLS)}")
        return value

    @classmethod
    def of(cls, amount: Decimal | str | int, currency: str = "GBP") -> Self:
        """Build from major units, e.g. Money.of("55.00") is £55.00. Floats are rejected."""
        if isinstance(amount, bool) or not isinstance(amount, Decimal | str | int):
            raise ValueError(f"amount must be Decimal, str or int, not {type(amount).__name__}")
        try:
            minor = (amount if isinstance(amount, Decimal) else Decimal(str(amount).strip())) * 100
        except InvalidOperation as exc:
            raise ValueError(f"invalid monetary amount {amount!r}") from exc
        if not minor.is_finite():
            raise ValueError(f"monetary amount must be finite, got {amount!r}")
        if minor != minor.to_integral_value():
            raise ValueError(f"monetary amount {amount!r} has more than two decimal places")
        return cls(amount_minor=int(minor), currency=currency)

    def _require_same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"cannot combine {self.currency} with {other.currency}")

    def __add__(self, other: object) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return Money(amount_minor=self.amount_minor + other.amount_minor, currency=self.currency)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_minor < other.amount_minor

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_minor <= other.amount_minor

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_minor > other.amount_minor

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_minor >= other.amount_minor

    def __str__(self) -> str:
        major, minor = divmod(self.amount_minor, 100)
        return f"{CURRENCY_SYMBOLS[self.currency]}{major:,}.{minor:02d}"


def sum_money(items: Iterable[Money], currency: str) -> Money:
    """Sum amounts that must all be in `currency`; an empty input gives zero."""
    total = Money(amount_minor=0, currency=currency)
    for item in items:
        total = total + item
    return total
