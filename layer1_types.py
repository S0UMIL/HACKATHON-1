"""
Hostel Wallet — Layer 1 shared types and validation helpers.

Financial convention (documented for all Layer 1 calculations):
- ``remaining_days`` counts calendar days from ``as_of_date`` through the day
  before ``next_allowance_date``, inclusive. Example: as_of=1 Jan, next allowance
  31 Jan → 30 remaining spending days. The allowance arrival date is not a
  spending day in this count; income on that date is modeled separately.
- Savings reserve and unpaid in-cycle commitments are subtracted up front when
  computing daily budgets (money mentally set aside). Cash-flow forecasts deduct
  each unpaid commitment once on its due date (or at forecast start if overdue).
  When ``balance_already_reflects_unpaid_liabilities`` is True, overdue unpaid
  items are treated as already netted in ``current_balance``.
- INR amounts are stored as float internally; user-facing whole-rupee values are
  rounded at presentation time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SavingsMode(str, Enum):
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    CUSTOM = "custom"
    NONE = "none"


class SpendingProfile(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# Optional profile hints only — not a substitute for observed spending.
_PROFILE_DAILY_FRACTION = {
    SpendingProfile.LOW: 0.22,
    SpendingProfile.MODERATE: 0.28,
    SpendingProfile.HIGH: 0.35,
}


@dataclass(frozen=True)
class SavingsConfig:
    mode: SavingsMode
    fixed_amount: float = 500.0
    percentage: float = 0.0
    custom_amount: float = 0.0
    emergency_reserve: float = 0.0


@dataclass
class Commitment:
    amount: float
    due_date: date
    description: str = ""
    commitment_id: str = ""
    paid: bool = False


@dataclass
class IncomeEvent:
    """
    Optional ``event_id``: duplicate submissions with the same id are ignored when
    aggregating income. Without ``event_id``, resubmits may double-count; identical
    date and amount alone are not deduplicated.
    """

    date: date
    amount: float
    description: str = ""
    event_id: str = ""


@dataclass
class ExpenseRecord:
    expense_date: date
    amount: float
    category: Optional[str] = None
    description: Optional[str] = None
    transaction_id: Optional[str] = None


@dataclass
class ExpenseLedger:
    """
    In-memory expense log. Deduplication uses ``transaction_id`` when present;
    without stable IDs, resubmitting the same record may double-count.
    """

    expenses: List[ExpenseRecord] = field(default_factory=list)
    accounted_days: Dict[date, bool] = field(default_factory=dict)
    _seen_transaction_ids: set[str] = field(default_factory=set)

    def add_expense(self, record: ExpenseRecord) -> Tuple[bool, str]:
        tid = (record.transaction_id or "").strip()
        if tid:
            if tid in self._seen_transaction_ids:
                return False, "Duplicate transaction_id; expense not added again."
            self._seen_transaction_ids.add(tid)
            record.transaction_id = tid
        self.expenses.append(record)
        return True, "Expense recorded."

    def mark_day_accounted(self, day: date, fully_accounted: bool = True) -> None:
        self.accounted_days[day] = fully_accounted

    def daily_totals(self) -> Dict[date, float]:
        totals: Dict[date, float] = {}
        for exp in self.expenses:
            totals[exp.expense_date] = totals.get(exp.expense_date, 0.0) + exp.amount
        return totals

    def fully_accounted_dates_sorted(self) -> List[date]:
        return sorted(d for d, ok in self.accounted_days.items() if ok)


def round_inr(amount: float, upward: bool = False) -> int:
    if upward:
        return int(math.ceil(amount - 1e-9))
    return int(round(amount))


def parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"Invalid date: {value!r}")


def validate_finite_amount(amount: float, field_name: str = "amount") -> None:
    if not isinstance(amount, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {type(amount).__name__}.")
    if not math.isfinite(amount):
        raise ValueError(f"{field_name} must be a finite number, got {amount}.")


def validate_non_negative_amount(amount: float, field_name: str = "amount") -> None:
    validate_finite_amount(amount, field_name)
    if amount < 0:
        raise ValueError(f"{field_name} must be non-negative, got {amount}.")


def validate_commitment(commitment: Commitment) -> None:
    validate_non_negative_amount(commitment.amount, "commitment.amount")
    if not isinstance(commitment.due_date, date):
        raise ValueError("commitment.due_date must be a date.")


def validate_income_event(event: IncomeEvent) -> None:
    validate_non_negative_amount(event.amount, "income.amount")
    if not isinstance(event.date, date):
        raise ValueError("income.date must be a date.")


def validate_expense_record(record: ExpenseRecord) -> None:
    validate_non_negative_amount(record.amount, "expense.amount")
    if not isinstance(record.expense_date, date):
        raise ValueError("expense.expense_date must be a date.")


def last_spending_day_in_cycle(next_allowance_date: date) -> date:
    """Last calendar day on which routine daily spending is expected in this cycle."""
    return next_allowance_date - timedelta(days=1)


def is_spending_day(day: date, as_of_date: date, next_allowance_date: date) -> bool:
    """Routine spending is expected on as_of_date .. day before next allowance."""
    if next_allowance_date <= as_of_date:
        return False
    last_day = last_spending_day_in_cycle(next_allowance_date)
    return as_of_date <= day <= last_day


def validate_active_allowance_cycle(
    as_of_date: date,
    next_allowance_date: date,
    allowance_received_date: date,
) -> None:
    """Raise ValueError when dates do not describe an active allowance cycle."""
    if next_allowance_date <= as_of_date:
        raise ValueError(
            "next_allowance_date must be after as_of_date for an active spending cycle."
        )
    if allowance_received_date > as_of_date:
        raise ValueError("allowance_received_date cannot be after as_of_date.")


def remaining_calendar_days(as_of_date: date, next_allowance_date: date) -> int:
    """
    Spending days remaining in the allowance cycle (includes as_of_date).

    Returns 0 when ``as_of_date >= next_allowance_date`` (cycle boundary reached).
    """
    if next_allowance_date <= as_of_date:
        return 0
    return (next_allowance_date - as_of_date).days


def unpaid_commitments_total(
    commitments: Sequence[Commitment],
    as_of_date: date,
    *,
    cycle_end_date: Optional[date] = None,
    balance_already_reflects_unpaid_liabilities: bool = False,
) -> Tuple[float, List[Commitment]]:
    """
    Sum unpaid commitment liabilities for budgeting.

    - Includes overdue, due-today, and future unpaid items unless overdue are
      already reflected in ``current_balance`` (see flag below).
    - When ``cycle_end_date`` is set (typically ``next_allowance_date``), only
      commitments with ``due_date < cycle_end_date`` belong to the current cycle.
    - When ``balance_already_reflects_unpaid_liabilities`` is True, unpaid items
      with ``due_date < as_of_date`` are excluded (already netted in balance).
    - Each commitment is counted at most once.
    """
    unpaid: List[Commitment] = []
    total = 0.0
    for c in commitments:
        validate_commitment(c)
        if c.paid:
            continue
        if balance_already_reflects_unpaid_liabilities and c.due_date < as_of_date:
            continue
        if cycle_end_date is not None and c.due_date >= cycle_end_date:
            continue
        unpaid.append(c)
        total += c.amount
    return total, unpaid


def compute_savings_reserve(
    allowance_amount: float,
    config: SavingsConfig,
) -> Dict[str, Any]:
    validate_non_negative_amount(allowance_amount, "allowance_amount")
    mode = config.mode
    target = 0.0
    if mode == SavingsMode.FIXED:
        target = max(0.0, config.fixed_amount)
    elif mode == SavingsMode.PERCENTAGE:
        if not 0 <= config.percentage <= 100:
            raise ValueError("percentage must be between 0 and 100.")
        target = allowance_amount * (config.percentage / 100.0)
    elif mode == SavingsMode.CUSTOM:
        target = max(0.0, config.custom_amount)
    elif mode == SavingsMode.NONE:
        target = 0.0
    else:
        raise ValueError(f"Unknown savings mode: {mode}")

    validate_non_negative_amount(config.fixed_amount, "fixed_amount")
    validate_non_negative_amount(config.custom_amount, "custom_amount")
    validate_non_negative_amount(config.emergency_reserve, "emergency_reserve")
    validate_finite_amount(config.percentage, "percentage")
    emergency = max(0.0, config.emergency_reserve)
    return {
        "cycle_savings_target": target,
        "emergency_reserve": emergency,
        "total_reserved_for_savings": target + emergency,
        "mode": mode.value,
    }
