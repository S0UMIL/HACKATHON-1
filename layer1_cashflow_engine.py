"""
Hostel Wallet — Layer 1 predictive cash-flow and burn-rate engine.

Deterministic, explainable calculations only (no ML, no external services).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from layer1_types import (
    Commitment,
    ExpenseLedger,
    ExpenseRecord,
    IncomeEvent,
    SavingsConfig,
    SavingsMode,
    SpendingProfile,
    _PROFILE_DAILY_FRACTION,
    compute_savings_reserve,
    is_spending_day,
    parse_date,
    remaining_calendar_days,
    round_inr,
    unpaid_commitments_total,
    validate_active_allowance_cycle,
    validate_commitment,
    validate_expense_record,
    validate_finite_amount,
    validate_income_event,
    validate_non_negative_amount,
)


DEFAULT_RECENT_WEIGHT = 0.7
DEFAULT_LONG_TERM_WEIGHT = 0.3
RECENT_WINDOW_DAYS = 7
ONE_TIME_OUTLIER_MULTIPLIER = 2.5


class CashflowValidationError(ValueError):
    """Raised when user inputs fail validation."""


def _require_active_cycle(
    as_of_date: date,
    next_allowance_date: date,
    allowance_received_date: date,
) -> None:
    try:
        validate_active_allowance_cycle(as_of_date, next_allowance_date, allowance_received_date)
    except ValueError as exc:
        raise CashflowValidationError(str(exc)) from exc


def _detect_commitment_ledger_overlap(
    commitments: Sequence[Commitment],
    ledger: ExpenseLedger,
    *,
    amount_tolerance_inr: float = 1.0,
) -> List[str]:
    """
    Warn when an unpaid commitment amount likely duplicates a logged expense on its due date.
    Paying a commitment should set ``paid=True``; do not log the same outflow twice.
    """
    warnings: List[str] = []
    totals = ledger.daily_totals()
    for c in commitments:
        if c.paid:
            continue
        day_total = totals.get(c.due_date, 0.0)
        if day_total <= 0:
            continue
        if abs(day_total - c.amount) <= max(amount_tolerance_inr, c.amount * 0.05):
            warnings.append(
                "commitment_ledger_overlap: unpaid commitment "
                f"{c.commitment_id or c.description or c.due_date.isoformat()} "
                f"({round_inr(c.amount)} INR on {c.due_date.isoformat()}) may duplicate "
                f"ledger spending ({round_inr(day_total)} INR that day). "
                "Mark the commitment paid or omit the duplicate expense."
            )
    return warnings


def _aggregate_income_by_date(
    events: Sequence[IncomeEvent],
) -> Tuple[Dict[date, float], List[str]]:
    totals: Dict[date, float] = {}
    seen_event_ids: set[str] = set()
    notes: List[str] = []
    for event in events:
        validate_income_event(event)
        event_id = (event.event_id or "").strip()
        if event_id:
            if event_id in seen_event_ids:
                notes.append(
                    f"Duplicate income event_id '{event_id}' ignored (not counted twice)."
                )
                continue
            seen_event_ids.add(event_id)
        totals[event.date] = totals.get(event.date, 0.0) + event.amount
    return totals, notes


def _commitment_forecast_adjustments(
    commitments: Sequence[Commitment],
    as_of_date: date,
    horizon_end: date,
    *,
    balance_already_reflects_unpaid_liabilities: bool = False,
) -> Tuple[float, Dict[date, float]]:
    """
    Cash-flow convention: ``current_balance`` is cash on hand at the start of
    ``as_of_date``. Unpaid liabilities with ``due_date < as_of_date`` reduce the
    projection immediately unless the caller confirms they are already reflected
    in the balance. Commitments due on or after ``as_of_date`` are deducted on
    their due date (each at most once).
    """
    opening_deduction = 0.0
    by_date: Dict[date, float] = {}
    for c in commitments:
        validate_commitment(c)
        if c.paid:
            continue
        if c.due_date < as_of_date:
            if not balance_already_reflects_unpaid_liabilities:
                opening_deduction += c.amount
            continue
        if c.due_date > horizon_end:
            continue
        by_date[c.due_date] = by_date.get(c.due_date, 0.0) + c.amount
    return opening_deduction, by_date


def _adjustment_meets_reserve_target(depletion_result: Dict[str, Any]) -> bool:
    return (
        not depletion_result["runs_out_before_next_allowance"]
        and depletion_result["expected_shortfall_end_of_cycle_exact"] <= 0
    )


def _routine_spending_applies(
    day: date,
    as_of_date: date,
    allowance_received_date: date,
    next_allowance_date: date,
) -> bool:
    """
    Routine spending applies on each calendar day except scheduled allowance dates.
    Within the current cycle, spending is limited to as_of_date .. day before next allowance.
    """
    if day < as_of_date:
        return False
    cycle_len = (next_allowance_date - allowance_received_date).days
    if cycle_len <= 0:
        return is_spending_day(day, as_of_date, next_allowance_date)
    if day < next_allowance_date:
        return is_spending_day(day, as_of_date, next_allowance_date)
    days_since_first_allowance = (day - next_allowance_date).days
    if days_since_first_allowance % cycle_len == 0:
        return False
    return True


def resolve_current_balance(
    allowance_amount: float,
    current_balance: Optional[float] = None,
    balance_unknown_assumption: str = "assume_equals_allowance",
) -> Dict[str, Any]:
    """
    If balance is omitted, document the assumption instead of inventing silently.
    balance_unknown_assumption: 'assume_equals_allowance' | 'require_explicit'.
    """
    validate_non_negative_amount(allowance_amount, "allowance_amount")
    if current_balance is not None:
        validate_non_negative_amount(current_balance, "current_balance")
        return {
            "current_balance": current_balance,
            "assumption": None,
            "note": "Using supplied current balance.",
        }
    if balance_unknown_assumption == "require_explicit":
        raise CashflowValidationError(
            "current_balance is unknown; supply a balance or change balance_unknown_assumption."
        )
    return {
        "current_balance": allowance_amount,
        "assumption": "current_balance_assumed_equal_to_allowance",
        "note": (
            "Current balance was not provided; assumed equal to the allowance amount. "
            "Update when the actual balance is known."
        ),
    }


def calculate_initial_budget(
    *,
    allowance_amount: float,
    allowance_received_date: date,
    as_of_date: date,
    next_allowance_date: date,
    savings_config: SavingsConfig,
    current_balance: Optional[float] = None,
    balance_already_excludes_savings: bool = False,
    balance_already_reflects_unpaid_liabilities: bool = False,
    commitments: Sequence[Commitment] = (),
    essential_daily_estimate: Optional[float] = None,
    spending_profile: Optional[SpendingProfile] = None,
    major_expense_categories: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Initial budget estimate (not a data-driven forecast). No spending history required.
    """
    balance_info = resolve_current_balance(allowance_amount, current_balance)
    balance = balance_info["current_balance"]

    savings = compute_savings_reserve(allowance_amount, savings_config)
    savings_reserve = savings["total_reserved_for_savings"]
    if balance_already_excludes_savings:
        savings_subtract = 0.0
        savings_note = (
            "Balance already excludes the savings reserve; savings not subtracted again."
        )
    else:
        savings_subtract = savings_reserve
        savings_note = "Savings reserve subtracted from balance for budgeting."

    _require_active_cycle(as_of_date, next_allowance_date, allowance_received_date)

    commit_total, unpaid_list = unpaid_commitments_total(
        commitments,
        as_of_date,
        cycle_end_date=next_allowance_date,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )
    remaining_days = remaining_calendar_days(as_of_date, next_allowance_date)

    funds_after_reserves = balance - savings_subtract - commit_total
    warnings: List[str] = []

    if funds_after_reserves < 0:
        warnings.append(
            "budget_insufficiency: savings target and unpaid commitments exceed available balance."
        )
        recommended_daily = 0.0
        mathematically_feasible = False
    else:
        recommended_daily = funds_after_reserves / remaining_days
        mathematically_feasible = recommended_daily >= 0
        if recommended_daily < 0:
            recommended_daily = 0.0
            warnings.append("recommended daily budget clamped to zero (negative raw value).")

    discretionary_pool = max(0.0, funds_after_reserves)
    essential_flag = None
    if essential_daily_estimate is not None:
        validate_non_negative_amount(essential_daily_estimate, "essential_daily_estimate")
        if essential_daily_estimate > recommended_daily and mathematically_feasible:
            essential_flag = (
                "essential_daily_estimate exceeds recommended daily budget; "
                "do not cut essentials automatically — seek additional income or adjust savings/commitments."
            )
            warnings.append(essential_flag)

    profile_hint = None
    if spending_profile is not None and spending_profile in _PROFILE_DAILY_FRACTION:
        frac = _PROFILE_DAILY_FRACTION[spending_profile]
        profile_hint = (
            f"Optional self-reported profile '{spending_profile.value}' suggests roughly "
            f"{round_inr(allowance_amount * frac)} INR/day as a rough lifestyle hint only — "
            "not derived from your spending history."
        )

    return {
        "kind": "initial_budget_estimate",
        "is_prediction_from_history": False,
        "recommended_daily_budget_inr": round_inr(recommended_daily),
        "recommended_daily_budget_exact": recommended_daily,
        "remaining_days": remaining_days,
        "remaining_days_convention": (
            "Inclusive calendar days from as_of_date through the day before next_allowance_date."
        ),
        "current_balance": balance,
        "balance_assumption": balance_info.get("assumption"),
        "balance_note": balance_info.get("note"),
        "savings": savings,
        "savings_accounting_note": savings_note,
        "commitments_reserved_inr": round_inr(commit_total),
        "commitments_reserved_exact": commit_total,
        "unpaid_commitments": [
            {
                "id": c.commitment_id,
                "amount": c.amount,
                "due_date": c.due_date.isoformat(),
                "description": c.description,
            }
            for c in unpaid_list
        ],
        "discretionary_funds_inr": round_inr(discretionary_pool),
        "discretionary_funds_exact": discretionary_pool,
        "mathematically_feasible": mathematically_feasible,
        "warnings": warnings,
        "essential_daily_discrepancy": essential_flag,
        "spending_profile_hint": profile_hint,
        "major_expense_categories": major_expense_categories,
        "allowance_received_date": allowance_received_date.isoformat(),
        "as_of_date": as_of_date.isoformat(),
        "next_allowance_date": next_allowance_date.isoformat(),
        "balance_already_reflects_unpaid_liabilities": balance_already_reflects_unpaid_liabilities,
    }


def recalculate_savings_and_budget(
    *,
    allowance_amount: float,
    allowance_received_date: date,
    as_of_date: date,
    next_allowance_date: date,
    savings_config: SavingsConfig,
    current_balance: float,
    balance_already_excludes_savings: bool = False,
    balance_already_reflects_unpaid_liabilities: bool = False,
    commitments: Sequence[Commitment] = (),
    essential_daily_estimate: Optional[float] = None,
) -> Dict[str, Any]:
    """User changed savings target; recompute without silently altering their goal."""
    return calculate_initial_budget(
        allowance_amount=allowance_amount,
        allowance_received_date=allowance_received_date,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        savings_config=savings_config,
        current_balance=current_balance,
        balance_already_excludes_savings=balance_already_excludes_savings,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        commitments=commitments,
        essential_daily_estimate=essential_daily_estimate,
    )


def validate_and_build_expense(record: Dict[str, Any]) -> ExpenseRecord:
    if not isinstance(record, dict):
        raise CashflowValidationError("Expense record must be a dictionary.")
    try:
        expense_date = parse_date(record.get("date"))
    except (ValueError, TypeError) as exc:
        raise CashflowValidationError(f"Invalid expense date: {exc}") from exc
    amount = record.get("amount")
    if amount is None:
        raise CashflowValidationError("Expense amount is required.")
    try:
        amount_f = float(amount)
    except (TypeError, ValueError) as exc:
        raise CashflowValidationError("Expense amount must be numeric.") from exc
    validate_non_negative_amount(amount_f, "amount")
    if amount_f == 0:
        raise CashflowValidationError("Zero-amount expenses are not recorded; omit the entry instead.")
    built = ExpenseRecord(
        expense_date=expense_date,
        amount=amount_f,
        category=record.get("category"),
        description=record.get("description"),
        transaction_id=record.get("transaction_id"),
    )
    validate_expense_record(built)
    return built


def aggregate_daily_expenses(ledger: ExpenseLedger) -> Dict[str, Any]:
    totals = ledger.daily_totals()
    partial_days = [
        d.isoformat()
        for d in sorted(totals)
        if not ledger.accounted_days.get(d, False)
    ]
    return {
        "daily_totals": {d.isoformat(): round_inr(v) for d, v in sorted(totals.items())},
        "daily_totals_exact": {d.isoformat(): v for d, v in sorted(totals.items())},
        "expense_count": len(ledger.expenses),
        "days_with_expenses_not_fully_accounted": partial_days,
        "deduplication": (
            "Records with the same non-empty transaction_id are ignored on resubmit. "
            "Without transaction_id, duplicate submissions may double-count."
        ),
    }


def record_manual_expense(
    ledger: ExpenseLedger,
    record: Dict[str, Any],
    current_balance: float,
    *,
    balance_already_reflects_expense: bool = False,
    mark_day_accounted: Optional[bool] = None,
) -> Dict[str, Any]:
    validate_non_negative_amount(current_balance, "current_balance")
    exp = validate_and_build_expense(record)
    added, msg = ledger.add_expense(exp)
    new_balance = current_balance
    if added and not balance_already_reflects_expense:
        new_balance = current_balance - exp.amount
    if mark_day_accounted is not None:
        ledger.mark_day_accounted(exp.expense_date, mark_day_accounted)
    return {
        "added": added,
        "message": msg,
        "current_balance": new_balance,
        "balance_note": (
            "Expense subtracted from balance."
            if added and not balance_already_reflects_expense
            else "Balance unchanged (duplicate or already reflected in balance)."
        ),
        "expense": {
            "date": exp.expense_date.isoformat(),
            "amount_inr": round_inr(exp.amount),
            "category": exp.category,
            "transaction_id": exp.transaction_id,
        },
    }


def parse_sms_transactions_placeholder(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    """
    Placeholder for future SMS-based transaction parsing.

    NOT IMPLEMENTED: does not read messages, access device storage, call external APIs,
    or create expenses. Integrate a real parser here in a later layer.
    """
    return {
        "implemented": False,
        "message": "SMS transaction parsing is not implemented in Layer 1.",
        "transactions": [],
        "balance_changed": False,
    }


def _validate_burn_weights(recent_weight: float, long_term_weight: float) -> None:
    validate_finite_amount(recent_weight, "recent_weight")
    validate_finite_amount(long_term_weight, "long_term_weight")
    if recent_weight < 0 or long_term_weight < 0:
        raise CashflowValidationError("Burn-rate weights must be non-negative.")
    total = recent_weight + long_term_weight
    if abs(total - 1.0) > 1e-6:
        raise CashflowValidationError("Burn-rate weights must sum to 1.")


def _median_finite(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v >= 0]
    if not finite:
        return 0.0
    ordered = sorted(finite)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _identify_outlier_days(daily_amounts: Dict[date, float]) -> List[date]:
    if len(daily_amounts) < 3:
        return []
    for amt in daily_amounts.values():
        validate_non_negative_amount(amt, "daily_spend")
    values = list(daily_amounts.values())
    median = _median_finite(values)
    if median == 0.0:
        positive = [v for v in values if v > 0]
        if len(positive) < 2:
            return []
        reference = _median_finite(positive)
        threshold = max(reference * ONE_TIME_OUTLIER_MULTIPLIER, reference + 1.0)
    else:
        threshold = max(median * ONE_TIME_OUTLIER_MULTIPLIER, median + 1.0)
    return [d for d, amt in daily_amounts.items() if amt > threshold]


def calculate_burn_rate(
    ledger: ExpenseLedger,
    *,
    as_of_date: date,
    initial_daily_budget: Optional[float] = None,
    recent_weight: float = DEFAULT_RECENT_WEIGHT,
    long_term_weight: float = DEFAULT_LONG_TERM_WEIGHT,
    recent_window_days: int = RECENT_WINDOW_DAYS,
) -> Dict[str, Any]:
    _validate_burn_weights(recent_weight, long_term_weight)
    if recent_window_days <= 0:
        raise CashflowValidationError("recent_window_days must be positive.")
    accounted = [d for d in ledger.fully_accounted_dates_sorted() if d <= as_of_date]
    totals = ledger.daily_totals()
    partial_reporting_days = [
        d.isoformat()
        for d in sorted(totals)
        if d <= as_of_date and not ledger.accounted_days.get(d, False)
    ]

    if not accounted:
        return {
            "kind": "burn_rate_estimate",
            "has_reliable_history": False,
            "label": "estimate_from_initial_budget",
            "adaptive_burn_rate_inr_per_day": round_inr(initial_daily_budget or 0),
            "adaptive_burn_rate_exact": initial_daily_budget or 0.0,
            "long_term_average": None,
            "recent_average": None,
            "recent_window_days_requested": recent_window_days,
            "recent_window_days_used": 0,
            "accounted_days_used": 0,
            "confidence": "none",
            "notes": [
                "No fully accounted spending days; using initial budget estimate, not learned behavior."
            ],
            "missing_days_not_treated_as_zero": True,
            "days_with_incomplete_reporting": partial_reporting_days,
        }

    daily_amounts = {d: totals.get(d, 0.0) for d in accounted}
    outlier_days = _identify_outlier_days(daily_amounts)
    recurring_amounts = {
        d: amt for d, amt in daily_amounts.items() if d not in outlier_days
    }
    use_amounts = recurring_amounts if len(recurring_amounts) >= 2 else daily_amounts

    long_term_avg = sum(use_amounts.values()) / len(use_amounts)

    window_start = as_of_date - timedelta(days=recent_window_days - 1)
    recent_slice = [d for d in accounted if window_start <= d <= as_of_date]
    recent_avg: Optional[float] = None
    if recent_slice:
        recent_avg = sum(daily_amounts[d] for d in recent_slice) / len(recent_slice)

    if recent_avg is None:
        adaptive = long_term_avg
        recent_confidence = "unavailable"
    else:
        adaptive = recent_weight * recent_avg + long_term_weight * long_term_avg
        recent_confidence = (
            "high"
            if len(recent_slice) >= recent_window_days
            else "reduced"
        )

    confidence = recent_confidence if recent_avg is not None else "reduced"
    notes = [
        "Averages include genuine zero-spending days on fully accounted dates only.",
        "Days without a complete report are excluded (not treated as zero).",
        (
            f"Recent window: calendar days {window_start.isoformat()} through "
            f"{as_of_date.isoformat()} ({recent_window_days} days)."
        ),
    ]
    if recent_avg is None:
        notes.append(
            "No fully accounted days in the recent calendar window; "
            "adaptive rate uses long-term average only."
        )
    else:
        notes.append(
            f"Recent average uses {len(recent_slice)} observed day(s) in that window."
        )
    if partial_reporting_days:
        notes.append(
            "Some dates have expenses logged but are not marked fully accounted: "
            + ", ".join(partial_reporting_days)
        )
    if outlier_days:
        notes.append(
            "One-time outlier days excluded from recurring burn estimate: "
            + ", ".join(d.isoformat() for d in outlier_days)
        )
    if recent_avg is not None and len(recent_slice) < recent_window_days:
        notes.append(
            f"Fewer than {recent_window_days} fully accounted days inside the recent calendar window."
        )

    return {
        "kind": "burn_rate_estimate",
        "has_reliable_history": True,
        "label": "estimate_from_observed_spending",
        "is_prediction_guarantee": False,
        "long_term_average_inr_per_day": round_inr(long_term_avg),
        "long_term_average_exact": long_term_avg,
        "recent_average_inr_per_day": round_inr(recent_avg) if recent_avg is not None else None,
        "recent_average_exact": recent_avg,
        "recent_average_available": recent_avg is not None,
        "adaptive_burn_rate_inr_per_day": round_inr(adaptive),
        "adaptive_burn_rate_exact": adaptive,
        "weights": {"recent": recent_weight, "long_term": long_term_weight},
        "recent_window_days_requested": recent_window_days,
        "recent_calendar_window_start": window_start.isoformat(),
        "recent_calendar_window_end": as_of_date.isoformat(),
        "recent_observations_in_window": len(recent_slice),
        "recent_window_days_used": len(recent_slice),
        "accounted_days_used": len(accounted),
        "confidence": confidence,
        "notes": notes,
        "outlier_days_excluded": [d.isoformat() for d in outlier_days],
        "days_with_incomplete_reporting": partial_reporting_days,
    }


def _build_income_schedule(
    allowance_received_date: date,
    allowance_amount: float,
    next_allowance_date: date,
    extra_income: Sequence[IncomeEvent],
    horizon_end: date,
) -> Dict[date, float]:
    validate_non_negative_amount(allowance_amount, "allowance_amount")
    for event in extra_income:
        validate_income_event(event)

    income_by_date, _income_notes = _aggregate_income_by_date(extra_income)
    cycle_days = (next_allowance_date - allowance_received_date).days
    if cycle_days <= 0:
        return income_by_date

    d = next_allowance_date
    while d <= horizon_end:
        income_by_date[d] = income_by_date.get(d, 0.0) + allowance_amount
        d = d + timedelta(days=cycle_days)
    return income_by_date


def _validate_non_negative_int_day_count(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise CashflowValidationError(f"{field_name} must be an integer day count, not a boolean.")
    if not isinstance(value, int):
        raise CashflowValidationError(f"{field_name} must be a whole number of days.")
    if value < 0:
        raise CashflowValidationError(f"{field_name} must be non-negative.")
    return value


def _eligible_spending_days_in_cycle(
    as_of_date: date,
    allowance_received_date: date,
    next_allowance_date: date,
) -> List[date]:
    days: List[date] = []
    last_day = as_of_date + timedelta(days=remaining_calendar_days(as_of_date, next_allowance_date) - 1)
    d = as_of_date
    while d <= last_day:
        if _routine_spending_applies(d, as_of_date, allowance_received_date, next_allowance_date):
            days.append(d)
        d += timedelta(days=1)
    return days


def _essential_shortfall_per_day(
    safe_daily: float, essential_daily_estimate: Optional[float]
) -> Optional[float]:
    if essential_daily_estimate is None:
        return None
    if essential_daily_estimate > safe_daily:
        return essential_daily_estimate - safe_daily
    return None


def _forecast_projection(
    *,
    current_balance: float,
    as_of_date: date,
    horizon_days: int,
    allowance_received_date: date,
    allowance_amount: float,
    next_allowance_date: date,
    commitments: Sequence[Commitment],
    extra_income: Sequence[IncomeEvent],
    savings_reserve: float,
    has_reliable_spending_history: bool,
    balance_already_reflects_unpaid_liabilities: bool,
    spend_for_day: Callable[[date], float],
    daily_spending_label_rate: float,
) -> Dict[str, Any]:
    horizon_end = as_of_date + timedelta(days=horizon_days - 1) if horizon_days else as_of_date
    income_by_date = _build_income_schedule(
        allowance_received_date,
        allowance_amount,
        next_allowance_date,
        extra_income,
        horizon_end,
    )
    opening_liabilities, unpaid_by_date = _commitment_forecast_adjustments(
        commitments,
        as_of_date,
        horizon_end,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )

    trajectory: List[Dict[str, Any]] = []
    balance = current_balance - opening_liabilities
    trajectory.append(
        {
            "date": as_of_date.isoformat(),
            "projected_balance": balance,
            "is_observed": True,
            "label": "observed_start_of_day",
            "opening_unpaid_liabilities_inr": round_inr(opening_liabilities),
        }
    )

    for offset in range(1, horizon_days + 1):
        day = as_of_date + timedelta(days=offset - 1)
        spend_today = max(0.0, spend_for_day(day))
        balance += income_by_date.get(day, 0.0)
        balance -= unpaid_by_date.get(day, 0.0)
        balance -= spend_today
        trajectory.append(
            {
                "date": day.isoformat(),
                "projected_balance": balance,
                "is_observed": False,
                "label": "projected_shortfall" if balance < 0 else "projected",
                "balance_timing": "end_of_day",
                "income_inr": income_by_date.get(day, 0.0),
                "commitments_inr": unpaid_by_date.get(day, 0.0),
                "daily_spend_inr": spend_today,
            }
        )

    forecast_kind = (
        "data_informed_projection"
        if has_reliable_spending_history
        else "initial_estimate_from_assumptions"
    )
    return {
        "kind": forecast_kind,
        "is_guaranteed_outcome": False,
        "horizon_days": horizon_days,
        "daily_spending_assumption_inr": round_inr(daily_spending_label_rate),
        "daily_spending_assumption_exact": daily_spending_label_rate,
        "savings_reserve_reference_inr": round_inr(savings_reserve),
        "trajectory": trajectory,
        "final_projected_balance": balance,
        "notes": [
            "Projection = current balance + future income - future expenses (spend rate + due commitments).",
            "Routine spending applies on as_of_date through the day before next_allowance_date.",
            "Overdue unpaid commitments reduce the projection at the start unless already in balance.",
            "Trajectory[0] is start-of-day observed balance; later rows are end-of-day after flows.",
            "Negative projected balances indicate funding gaps, not spendable cash.",
        ],
    }


def forecast_future_balances(
    *,
    current_balance: float,
    as_of_date: date,
    horizon_days: int,
    daily_spending_rate: float,
    allowance_received_date: date,
    allowance_amount: float,
    next_allowance_date: date,
    commitments: Sequence[Commitment] = (),
    extra_income: Sequence[IncomeEvent] = (),
    savings_reserve: float = 0.0,
    has_reliable_spending_history: bool = False,
    balance_already_reflects_unpaid_liabilities: bool = False,
) -> Dict[str, Any]:
    if horizon_days < 0:
        raise CashflowValidationError("horizon_days must be non-negative.")
    validate_non_negative_amount(current_balance, "current_balance")
    validate_non_negative_amount(daily_spending_rate, "daily_spending_rate")
    validate_non_negative_amount(savings_reserve, "savings_reserve")
    _require_active_cycle(as_of_date, next_allowance_date, allowance_received_date)

    def spend_for_day(day: date) -> float:
        if _routine_spending_applies(
            day, as_of_date, allowance_received_date, next_allowance_date
        ):
            return daily_spending_rate
        return 0.0

    return _forecast_projection(
        current_balance=current_balance,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        next_allowance_date=next_allowance_date,
        commitments=commitments,
        extra_income=extra_income,
        savings_reserve=savings_reserve,
        has_reliable_spending_history=has_reliable_spending_history,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        spend_for_day=spend_for_day,
        daily_spending_label_rate=daily_spending_rate,
    )


def estimate_depletion(
    *,
    current_balance: float,
    as_of_date: date,
    next_allowance_date: date,
    allowance_received_date: date,
    allowance_amount: float,
    daily_spending_rate: float,
    commitments: Sequence[Commitment] = (),
    savings_reserve: float = 0.0,
    extra_income: Sequence[IncomeEvent] = (),
    max_forecast_days: Optional[int] = None,
    balance_already_reflects_unpaid_liabilities: bool = False,
    has_reliable_spending_history: bool = False,
    spend_for_day: Optional[Callable[[date], float]] = None,
) -> Dict[str, Any]:
    validate_non_negative_amount(daily_spending_rate, "daily_spending_rate")
    _require_active_cycle(as_of_date, next_allowance_date, allowance_received_date)
    has_unpaid_liabilities = any(not c.paid for c in commitments)
    if spend_for_day is None and daily_spending_rate <= 0 and not has_unpaid_liabilities:
        return {
            "runway_days": None,
            "depletion_date": None,
            "reserve_breach_date": None,
            "runs_out_before_next_allowance": False,
            "expected_shortfall_end_of_cycle_inr": 0,
            "assumptions": [
                "No positive daily spending rate and no future commitments; depletion not projected."
            ],
            "is_guaranteed": False,
        }

    remaining = remaining_calendar_days(as_of_date, next_allowance_date)
    if max_forecast_days is None:
        max_forecast_days = max(remaining + 30, 60)

    spend_fn = spend_for_day
    if spend_fn is None:

        def spend_fn(day: date) -> float:
            if _routine_spending_applies(
                day, as_of_date, allowance_received_date, next_allowance_date
            ):
                return daily_spending_rate
            return 0.0

    forecast = _forecast_projection(
        current_balance=current_balance,
        as_of_date=as_of_date,
        horizon_days=max_forecast_days,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        next_allowance_date=next_allowance_date,
        commitments=commitments,
        extra_income=extra_income,
        savings_reserve=savings_reserve,
        has_reliable_spending_history=has_reliable_spending_history,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        spend_for_day=spend_fn,
        daily_spending_label_rate=daily_spending_rate,
    )

    depletion_date = None
    reserve_breach_date = None
    runway_days = None

    for point in forecast["trajectory"]:
        if point.get("is_observed"):
            continue
        bal = point["projected_balance"]
        d = date.fromisoformat(point["date"])
        if depletion_date is None and bal < 0:
            depletion_date = d
            runway_days = (d - as_of_date).days
        if reserve_breach_date is None and savings_reserve > 0 and bal < savings_reserve:
            reserve_breach_date = d

    days_to_end = remaining if remaining > 0 else 0
    end_forecast = _forecast_projection(
        current_balance=current_balance,
        as_of_date=as_of_date,
        horizon_days=days_to_end,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        next_allowance_date=next_allowance_date,
        commitments=commitments,
        extra_income=extra_income,
        savings_reserve=savings_reserve,
        has_reliable_spending_history=has_reliable_spending_history,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        spend_for_day=spend_fn,
        daily_spending_label_rate=daily_spending_rate,
    )
    end_balance = end_forecast["final_projected_balance"]
    shortfall = max(0.0, savings_reserve - end_balance) if end_balance < savings_reserve else 0.0
    if end_balance < 0:
        shortfall = max(shortfall, -end_balance)

    runs_out_before = (
        depletion_date is not None and depletion_date < next_allowance_date
    )

    assumptions = [
        f"Daily spend rate: {round_inr(daily_spending_rate)} INR/day (projection, not fact).",
        "Includes allowance on scheduled dates and commitments on due dates.",
        "Deterministic projection — actual results will differ.",
    ]
    if depletion_date is None:
        assumptions.append(
            f"Depletion not detected within {max_forecast_days} forecast days."
        )

    return {
        "runway_days": runway_days,
        "depletion_date": depletion_date.isoformat() if depletion_date else None,
        "reserve_breach_date": reserve_breach_date.isoformat() if reserve_breach_date else None,
        "runs_out_before_next_allowance": runs_out_before,
        "expected_shortfall_end_of_cycle_inr": round_inr(shortfall),
        "expected_shortfall_end_of_cycle_exact": shortfall,
        "assumptions": assumptions,
        "is_guaranteed": False,
    }


def calculate_safe_daily_spend(
    *,
    current_balance: float,
    as_of_date: date,
    next_allowance_date: date,
    savings_reserve: float,
    commitments: Sequence[Commitment] = (),
    balance_already_excludes_savings: bool = False,
    balance_already_reflects_unpaid_liabilities: bool = False,
    essential_daily_estimate: Optional[float] = None,
) -> Dict[str, Any]:
    validate_non_negative_amount(current_balance, "current_balance")
    validate_non_negative_amount(savings_reserve, "savings_reserve")
    if essential_daily_estimate is not None:
        validate_non_negative_amount(essential_daily_estimate, "essential_daily_estimate")
    if next_allowance_date <= as_of_date:
        essential_shortfall = _essential_shortfall_per_day(0.0, essential_daily_estimate)
        return {
            "safe_daily_spend_inr": 0,
            "safe_daily_spend_exact": 0.0,
            "essential_daily_estimate_inr": (
                round_inr(essential_daily_estimate) if essential_daily_estimate is not None else None
            ),
            "essential_shortfall_inr_per_day": (
                round_inr(essential_shortfall, upward=True) if essential_shortfall is not None else None
            ),
            "warnings": ["No remaining spending days in the current allowance cycle."],
            "feasible": essential_shortfall is None,
        }

    savings_subtract = 0.0 if balance_already_excludes_savings else savings_reserve
    commit_total, _ = unpaid_commitments_total(
        commitments,
        as_of_date,
        cycle_end_date=next_allowance_date,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )
    remaining_days = remaining_calendar_days(as_of_date, next_allowance_date)

    numerator = current_balance - savings_subtract - commit_total
    warnings: List[str] = []
    if numerator < 0:
        warnings.append(
            "budget_insufficiency: balance cannot cover savings reserve and unpaid commitments."
        )
        safe_daily = 0.0
        essential_shortfall = _essential_shortfall_per_day(safe_daily, essential_daily_estimate)
        if essential_shortfall is not None:
            warnings.append(
                "essential expenses alone exceed sustainable daily budget; shortfall reported, not auto-cut."
            )
        return {
            "safe_daily_spend_inr": 0,
            "safe_daily_spend_exact": 0.0,
            "remaining_days": remaining_days,
            "savings_reserve_inr": round_inr(savings_reserve),
            "commitments_reserved_inr": round_inr(commit_total),
            "essential_daily_estimate_inr": (
                round_inr(essential_daily_estimate) if essential_daily_estimate is not None else None
            ),
            "essential_shortfall_inr_per_day": (
                round_inr(essential_shortfall, upward=True) if essential_shortfall is not None else None
            ),
            "discretionary_allowance_inr_per_day": 0,
            "discretionary_pool_inr": 0,
            "warnings": warnings,
            "feasible": essential_shortfall is None,
        }

    safe_daily = numerator / remaining_days
    discretionary_total = numerator
    essential_shortfall = _essential_shortfall_per_day(safe_daily, essential_daily_estimate)
    discretionary_part = safe_daily
    if essential_daily_estimate is not None:
        discretionary_part = max(0.0, safe_daily - essential_daily_estimate)
        if essential_shortfall is not None:
            warnings.append(
                "essential expenses alone exceed sustainable daily budget; shortfall reported, not auto-cut."
            )

    return {
        "safe_daily_spend_inr": round_inr(safe_daily),
        "safe_daily_spend_exact": safe_daily,
        "remaining_days": remaining_days,
        "savings_reserve_inr": round_inr(savings_reserve),
        "commitments_reserved_inr": round_inr(commit_total),
        "essential_daily_estimate_inr": (
            round_inr(essential_daily_estimate) if essential_daily_estimate is not None else None
        ),
        "essential_shortfall_inr_per_day": (
            round_inr(essential_shortfall, upward=True) if essential_shortfall is not None else None
        ),
        "discretionary_allowance_inr_per_day": round_inr(discretionary_part),
        "discretionary_pool_inr": round_inr(discretionary_total),
        "warnings": warnings,
        "feasible": essential_shortfall is None,
    }


def spending_reduction_recommendations(
    *,
    current_balance: float,
    as_of_date: date,
    next_allowance_date: date,
    allowance_received_date: date,
    allowance_amount: float,
    expected_daily_spending: float,
    sustainable_daily_spending: float,
    savings_reserve: float,
    commitments: Sequence[Commitment] = (),
    essential_daily_estimate: Optional[float] = None,
    balance_already_reflects_unpaid_liabilities: bool = False,
) -> Dict[str, Any]:
    validate_non_negative_amount(expected_daily_spending, "expected_daily_spending")
    validate_non_negative_amount(sustainable_daily_spending, "sustainable_daily_spending")
    validate_non_negative_amount(savings_reserve, "savings_reserve")
    remaining_days = remaining_calendar_days(as_of_date, next_allowance_date)
    if remaining_days == 0:
        raise CashflowValidationError("No remaining spending days in the current allowance cycle.")

    safe = calculate_safe_daily_spend(
        current_balance=current_balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        savings_reserve=savings_reserve,
        commitments=commitments,
        essential_daily_estimate=essential_daily_estimate,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )
    sustainable_with_reserve = safe["safe_daily_spend_exact"]
    mismatch_note = None
    if abs(sustainable_daily_spending - sustainable_with_reserve) > 1.0:
        mismatch_note = (
            "Caller sustainable_daily_spending differs from canonical safe_daily_spend; "
            "reductions use the canonical safe rate."
        )
    reduction_to_non_negative = max(0.0, expected_daily_spending - sustainable_with_reserve)
    reduction_to_preserve_reserve = max(
        0.0, expected_daily_spending - sustainable_with_reserve
    )

    depletion = estimate_depletion(
        current_balance=current_balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        daily_spending_rate=expected_daily_spending,
        commitments=commitments,
        savings_reserve=savings_reserve,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )
    gap_end = depletion["expected_shortfall_end_of_cycle_exact"]
    one_time_savings_needed = max(0.0, gap_end)
    recurring_daily_cut = reduction_to_preserve_reserve

    essential_shortfall = None
    if essential_daily_estimate is not None and essential_daily_estimate > sustainable_with_reserve:
        essential_shortfall = essential_daily_estimate - sustainable_with_reserve

    return {
        "remaining_days": remaining_days,
        "expected_daily_spending_inr": round_inr(expected_daily_spending),
        "sustainable_daily_spending_inr": round_inr(sustainable_with_reserve),
        "caller_sustainable_daily_spending_inr": round_inr(sustainable_daily_spending),
        "required_daily_reduction_to_avoid_negative_inr": round_inr(reduction_to_non_negative),
        "required_daily_reduction_to_preserve_reserve_inr": round_inr(
            reduction_to_preserve_reserve, upward=True
        ),
        "recurring_daily_reduction": {
            "amount_inr_per_day": round_inr(recurring_daily_cut, upward=True),
            "basis": (
                f"max(0, expected {round_inr(expected_daily_spending)} - "
                f"sustainable {round_inr(sustainable_with_reserve)}) over {remaining_days} days"
            ),
        },
        "one_time_savings": {
            "amount_inr": round_inr(one_time_savings_needed, upward=True),
            "basis": "Projected funding gap at end of allowance cycle from current trajectory.",
        },
        "essential_expense_shortfall_inr_per_day": (
            round_inr(essential_shortfall, upward=True) if essential_shortfall else None
        ),
        "does_not_automatically_cut_essentials": True,
        "is_guaranteed_fix": False,
        "canonical_sustainable_note": mismatch_note,
    }


def evaluate_spending_adjustment(
    *,
    current_balance: float,
    as_of_date: date,
    next_allowance_date: date,
    allowance_received_date: date,
    allowance_amount: float,
    baseline_daily_spending: float,
    commitments: Sequence[Commitment] = (),
    savings_reserve: float = 0.0,
    daily_reduction_inr: float = 0.0,
    one_time_save_inr: float = 0.0,
    one_time_save_in_days: int = 0,
    balance_already_reflects_unpaid_liabilities: bool = False,
) -> Dict[str, Any]:
    """Simulate a proposed adjustment without double-counting."""
    validate_non_negative_amount(current_balance, "current_balance")
    _require_active_cycle(as_of_date, next_allowance_date, allowance_received_date)
    validate_non_negative_amount(baseline_daily_spending, "baseline_daily_spending")
    validate_non_negative_amount(savings_reserve, "savings_reserve")
    validate_non_negative_amount(daily_reduction_inr, "daily_reduction_inr")
    validate_non_negative_amount(one_time_save_inr, "one_time_save_inr")
    spread_days_requested = _validate_non_negative_int_day_count(
        one_time_save_in_days, "one_time_save_in_days"
    )

    adjusted_balance = current_balance
    recurring_daily = max(0.0, baseline_daily_spending - daily_reduction_inr)
    adjustment_mode = "daily_reduction_only"
    spread_reduction_days_applied = 0
    spread_reduction_days_requested = spread_days_requested
    spread_per_day_inr = 0.0
    spread_day_set: set[date] = set()
    spread_cap_note: Optional[str] = None

    if one_time_save_inr > 0 and spread_days_requested > 0:
        adjustment_mode = "one_time_spending_reduction_spread"
        spread_per_day_inr = one_time_save_inr / spread_days_requested
        eligible = _eligible_spending_days_in_cycle(
            as_of_date, allowance_received_date, next_allowance_date
        )
        spread_reduction_days_applied = min(spread_days_requested, len(eligible))
        spread_day_set = set(eligible[:spread_reduction_days_applied])
        if spread_reduction_days_applied < spread_days_requested:
            spread_cap_note = (
                f"One-time reduction spread capped to {spread_reduction_days_applied} eligible "
                f"spending day(s) remaining in the cycle (requested {spread_days_requested})."
            )
    elif one_time_save_inr > 0:
        adjustment_mode = "one_time_balance_increase"
        adjusted_balance = current_balance + one_time_save_inr

    def spend_before(day: date) -> float:
        if not _routine_spending_applies(
            day, as_of_date, allowance_received_date, next_allowance_date
        ):
            return 0.0
        return max(0.0, baseline_daily_spending)

    def spend_after(day: date) -> float:
        if not _routine_spending_applies(
            day, as_of_date, allowance_received_date, next_allowance_date
        ):
            return 0.0
        spend = recurring_daily
        if day in spread_day_set:
            spend = max(0.0, spend - spread_per_day_inr)
        return spend

    depletion_before = estimate_depletion(
        current_balance=current_balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        daily_spending_rate=baseline_daily_spending,
        commitments=commitments,
        savings_reserve=savings_reserve,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        spend_for_day=spend_before,
    )
    depletion_after = estimate_depletion(
        current_balance=adjusted_balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        daily_spending_rate=recurring_daily,
        commitments=commitments,
        savings_reserve=savings_reserve,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        spend_for_day=spend_after,
    )

    sufficient = _adjustment_meets_reserve_target(depletion_after)

    horizon = remaining_calendar_days(as_of_date, next_allowance_date)
    forecast_after = _forecast_projection(
        current_balance=adjusted_balance,
        as_of_date=as_of_date,
        horizon_days=horizon,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        next_allowance_date=next_allowance_date,
        commitments=commitments,
        extra_income=[],
        savings_reserve=savings_reserve,
        has_reliable_spending_history=True,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        spend_for_day=spend_after,
        daily_spending_label_rate=recurring_daily,
    )

    representative_daily = recurring_daily
    if spread_day_set:
        total_spend = sum(spend_after(d) for d in spread_day_set)
        representative_daily = total_spend / len(spread_day_set) if spread_day_set else recurring_daily

    return {
        "adjusted_current_balance": adjusted_balance,
        "adjusted_daily_spending_inr": round_inr(representative_daily),
        "recurring_daily_spending_inr": round_inr(recurring_daily),
        "one_time_spread_reduction_inr_per_day": round_inr(spread_per_day_inr),
        "one_time_spread_days_requested": spread_reduction_days_requested,
        "one_time_spread_days_applied": spread_reduction_days_applied,
        "one_time_spread_cap_note": spread_cap_note,
        "projected_balance_end_of_cycle": forecast_after["final_projected_balance"],
        "depletion_before": depletion_before,
        "depletion_after": depletion_after,
        "adjustment_sufficient": sufficient,
        "adjustment_mode": adjustment_mode,
        "meets_reserve_target": sufficient,
        "explanation": (
            "one_time_save_inr increases balance only when one_time_save_in_days is 0; "
            "when spread, the per-day reduction applies only on eligible spending days in the "
            "cycle (not the full horizon at a flat reduced rate). "
            "Sufficient means projected end-of-cycle shortfall is zero and reserve target is met — not a guarantee."
        ),
    }


def assess_financial_risk(
    *,
    current_balance: float,
    as_of_date: date,
    next_allowance_date: date,
    allowance_received_date: date,
    allowance_amount: float,
    daily_spending_rate: float,
    savings_reserve: float,
    commitments: Sequence[Commitment] = (),
    burn_rate_result: Optional[Dict[str, Any]] = None,
    essential_daily_estimate: Optional[float] = None,
    large_expense_recent_inr: Optional[float] = None,
    balance_already_reflects_unpaid_liabilities: bool = False,
) -> Dict[str, Any]:
    depletion = estimate_depletion(
        current_balance=current_balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        daily_spending_rate=daily_spending_rate,
        commitments=commitments,
        savings_reserve=savings_reserve,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )
    safe = calculate_safe_daily_spend(
        current_balance=current_balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        savings_reserve=savings_reserve,
        commitments=commitments,
        essential_daily_estimate=essential_daily_estimate,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )

    reasons: List[str] = []
    level = "low"

    if depletion["runs_out_before_next_allowance"]:
        level = "high"
        reasons.append("Projected balance goes negative before the next allowance.")
    elif depletion["reserve_breach_date"]:
        level = "moderate" if level == "low" else level
        reasons.append("Projected balance may fall below the savings reserve before cycle end.")

    if burn_rate_result:
        recent = burn_rate_result.get("recent_average_exact")
        long_term = burn_rate_result.get("long_term_average_exact")
        if recent is not None and long_term is not None and recent > long_term * 1.15:
            level = "moderate" if level == "low" else level
            reasons.append("Recent daily spending average exceeds longer-term average.")

    essential_shortfall = safe.get("essential_shortfall_inr_per_day")
    if essential_shortfall or (
        essential_daily_estimate is not None
        and essential_daily_estimate > safe.get("safe_daily_spend_exact", 0.0)
    ):
        level = "high"
        reasons.append(
            "Essential daily spending exceeds the sustainable safe-to-spend amount."
        )

    if large_expense_recent_inr is not None and depletion["runway_days"] is not None:
        if large_expense_recent_inr > daily_spending_rate * 3:
            level = "moderate" if level == "low" else level
            reasons.append("Large recent expense may shorten financial runway.")

    if depletion["expected_shortfall_end_of_cycle_exact"] > 0:
        if level == "low":
            level = "moderate"
        reasons.append(
            f"Projected shortfall of {round_inr(depletion['expected_shortfall_end_of_cycle_exact'])} INR at cycle end."
        )

    if not reasons:
        reasons.append("Spending pace appears sustainable relative to balance and upcoming allowance.")

    return {
        "risk_label": level,
        "reasons": reasons,
        "is_guaranteed_outcome": False,
        "depletion_summary": depletion,
    }


def build_layer1_snapshot(
    *,
    allowance_amount: float,
    allowance_received_date: date,
    as_of_date: date,
    next_allowance_date: date,
    savings_config: SavingsConfig,
    ledger: ExpenseLedger,
    current_balance: Optional[float] = None,
    balance_already_excludes_savings: bool = False,
    balance_already_reflects_unpaid_liabilities: bool = False,
    commitments: Sequence[Commitment] = (),
    essential_daily_estimate: Optional[float] = None,
    spending_profile: Optional[SpendingProfile] = None,
    extra_income: Sequence[IncomeEvent] = (),
) -> Dict[str, Any]:
    """Convenience entry: initial budget, burn rate, safe spend, forecast, depletion, risk."""
    integration_warnings = _detect_commitment_ledger_overlap(commitments, ledger)
    initial = calculate_initial_budget(
        allowance_amount=allowance_amount,
        allowance_received_date=allowance_received_date,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        savings_config=savings_config,
        current_balance=current_balance,
        balance_already_excludes_savings=balance_already_excludes_savings,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        commitments=commitments,
        essential_daily_estimate=essential_daily_estimate,
        spending_profile=spending_profile,
    )
    balance = initial["current_balance"]
    savings_reserve = initial["savings"]["total_reserved_for_savings"]
    initial_daily = initial["recommended_daily_budget_exact"]

    burn = calculate_burn_rate(
        ledger,
        as_of_date=as_of_date,
        initial_daily_budget=initial_daily,
    )
    spend_rate = (
        burn["adaptive_burn_rate_exact"]
        if burn.get("has_reliable_history")
        else initial_daily
    )
    has_history = burn.get("has_reliable_history", False)

    horizon = remaining_calendar_days(as_of_date, next_allowance_date)
    forecast = forecast_future_balances(
        current_balance=balance,
        as_of_date=as_of_date,
        horizon_days=horizon,
        daily_spending_rate=spend_rate,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        next_allowance_date=next_allowance_date,
        commitments=commitments,
        extra_income=extra_income,
        savings_reserve=savings_reserve,
        has_reliable_spending_history=has_history,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )
    safe = calculate_safe_daily_spend(
        current_balance=balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        savings_reserve=savings_reserve,
        commitments=commitments,
        balance_already_excludes_savings=balance_already_excludes_savings,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
        essential_daily_estimate=essential_daily_estimate,
    )
    depletion = estimate_depletion(
        current_balance=balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        daily_spending_rate=spend_rate,
        commitments=commitments,
        savings_reserve=savings_reserve,
        extra_income=extra_income,
        has_reliable_spending_history=has_history,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )
    recommendations = spending_reduction_recommendations(
        current_balance=balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        expected_daily_spending=spend_rate,
        sustainable_daily_spending=safe["safe_daily_spend_exact"],
        savings_reserve=savings_reserve,
        commitments=commitments,
        essential_daily_estimate=essential_daily_estimate,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )
    risk = assess_financial_risk(
        current_balance=balance,
        as_of_date=as_of_date,
        next_allowance_date=next_allowance_date,
        allowance_received_date=allowance_received_date,
        allowance_amount=allowance_amount,
        daily_spending_rate=spend_rate,
        savings_reserve=savings_reserve,
        commitments=commitments,
        burn_rate_result=burn,
        essential_daily_estimate=essential_daily_estimate,
        balance_already_reflects_unpaid_liabilities=balance_already_reflects_unpaid_liabilities,
    )

    return {
        "initial_budget": initial,
        "burn_rate": burn,
        "safe_daily_spend": safe,
        "forecast": forecast,
        "depletion": depletion,
        "spending_reduction": recommendations,
        "risk": risk,
        "expense_summary": aggregate_daily_expenses(ledger),
        "integration_warnings": integration_warnings,
    }
