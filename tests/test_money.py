"""Monetary validation and repayment arithmetic."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from sika_guard.fixtures.builders import fee, gbp, level, make_transaction, make_offer
from sika_guard.models import Fee, FeeKind, RepaymentSchedule
from sika_guard.money import CurrencyMismatchError, Money


@pytest.mark.parametrize(
    "bad",
    [
        lambda: Money.of(-1),
        lambda: Money.of(55.5),  # floats are never accepted
        lambda: Money.of(True),
        lambda: Money.of("12.345"),  # sub-penny precision
        lambda: Money.of("abc"),
        lambda: Money.of("NaN"),
        lambda: Money.of("Infinity"),
        lambda: Money.of("10", "JPY"),  # unsupported currency
        lambda: Money(amount_minor=-1, currency="GBP"),
        lambda: Money(amount_minor=1.5, currency="GBP"),
        lambda: Money(amount_minor="100", currency="GBP"),
        lambda: Money(amount_minor=100, currency="gbp"),
    ],
)
def test_invalid_monetary_inputs_fail_validation(bad):
    with pytest.raises(ValueError):  # pydantic's ValidationError is a ValueError
        bad()


def test_zero_amount_transaction_is_rejected():
    offer = make_offer("OFFER-A", level("55"))
    with pytest.raises(ValidationError):
        make_transaction(offer, amount=Money(amount_minor=0, currency="GBP"))


def test_schedule_with_mixed_currencies_is_rejected():
    with pytest.raises(ValidationError):
        RepaymentSchedule(payments=(gbp(10), Money.of(10, "EUR")))


def test_money_parses_decimal_strings_exactly():
    assert Money.of("57.50").amount_minor == 5750
    assert Money.of(Decimal("0.07")).amount_minor == 7
    assert str(Money.of("1234.5")) == "£1,234.50"


def test_money_refuses_to_mix_currencies():
    with pytest.raises(CurrencyMismatchError):
        gbp(1) + Money.of(1, "EUR")
    with pytest.raises(CurrencyMismatchError):
        gbp(1) < Money.of(1, "EUR")


def test_repayment_total_is_payments_plus_fees():
    assert level("55").total() == gbp(660)
    assert level("55", fees=(fee(FeeKind.ADMINISTRATION, "30"),)).total() == gbp(690)


def test_repayment_total_handles_uneven_schedules():
    schedule = RepaymentSchedule(payments=(gbp(150),) * 2 + (gbp(39),) * 10)
    assert schedule.total() == gbp(690)
    assert schedule.max_payment() == gbp(150)
    assert schedule.term_months == 12


def test_initial_charge_is_first_payment_plus_upfront_fees_only():
    schedule = level(
        "55",
        fees=(fee(FeeKind.ARRANGEMENT, "25", due_upfront=True), Fee(kind=FeeKind.SERVICING, amount=gbp(10))),
    )
    assert schedule.initial_charge() == gbp(80)
    assert schedule.total() == gbp(695)
