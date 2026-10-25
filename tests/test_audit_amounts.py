"""Amount reading for the audit findings F1, F4 and F5: quantity labels, complete consumption of an amount
expression, and recognising a real 'no fees' statement. Offline."""

import pytest

from sika_guard.evaluators.amounts import Quantity, mentions_money, no_fee_expression, parse_amounts, scan_amounts

@pytest.mark.parametrize("text", ["There are no hidden fees to worry about.", "Fee-free financing.", "Zero fees. Ever.", "No arrangement fee or other charges.", "Without any additional charges."])
def test_f1_the_no_fee_expressions_recognised_are_real_ones(text):
    assert no_fee_expression(text)


@pytest.mark.parametrize("text", ["Includes accident cover.", "A fee of £20 applies.", "No questions asked returns.", "Fees may apply.", "Not free of charge."])
def test_f1_text_that_does_not_say_there_are_no_fees_is_not_taken_to(text):
    assert not no_fee_expression(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("£55 a month, total repayable £660", [(55, Quantity.MONTHLY), (660, Quantity.TOTAL)]),
        ("Total repayable £680 including a £20 fee", [(680, Quantity.TOTAL), (20, Quantity.FEE)]),
        ("A £55 monthly payment and total fees of £5 apply", [(55, Quantity.MONTHLY), (5, Quantity.FEE_TOTAL)]),
        ("£660 in all", [(660, Quantity.TOTAL)]),
        ("An administration fee of £20 applies", [(20, Quantity.FEE)]),
        ("12 payments of £55", [(55, Quantity.MONTHLY)]),
        ("Only £99!", [(99, Quantity.UNKNOWN)]),
    ],
)
def test_f4_f5_amounts_are_read_with_the_quantity_the_words_around_them_give(text, expected):
    mentions = scan_amounts(text)
    assert [(int(m.money.amount_minor / 100), m.quantity) for m in mentions if m.money] == expected


# --- F5: quantity, and complete consumption of the amount expression ----------------------------------------------------------------------


@pytest.mark.parametrize("text", ["£680k", "£0.68m", "£680,50", "£680.123", "£6,80", "£1.234,56", "£5 million", "680k GBP", "£1e3", "£12,34"])
def test_f5_malformed_expressions_yield_no_valid_amount_and_are_flagged(text):
    assert parse_amounts(text) == [] and mentions_money(text)
    assert any(m.problem for m in scan_amounts(text))


@pytest.mark.parametrize("text, value", [("£1,234.50", 123450), ("£680", 68000), ("680.00 GBP", 68000), ("GBP 1,234", 123400), ("$5.5", 550), ("€7", 700), ("It costs £680.", 68000), ("£680, payable monthly", 68000)])
def test_f5_well_formed_amounts_still_parse_exactly(text, value):
    assert [m.amount_minor for m in parse_amounts(text)] == [value]


# --- F6: disagreement and incompatibility stay visible ----------------------------------------------------------------------------------------




# --- a unit that follows the amount decides it, even beside a "total" -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Total £55 per month", Quantity.MONTHLY),
        ("Total monthly payment £55", Quantity.MONTHLY),
        ("Fee: £20 total", Quantity.FEE_TOTAL),
        ("Total £660 over 12 months", Quantity.TOTAL),
        ("Total cost including fees £680", Quantity.UNKNOWN),  # two quantities' words around one amount: unknown, never guessed
    ],
)
def test_f4_f5_a_unit_after_the_amount_or_a_label_before_it_decides_its_quantity(text, expected):
    (mention,) = scan_amounts(text)
    assert mention.quantity is expected
