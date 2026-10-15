# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from syllabus_research.budget import BudgetTracker, CallLimitExceeded, SpendingLimitExceeded
from syllabus_research.models import RunUsage


def test_call_and_spending_limits_are_enforced_before_calls():
    tracker = BudgetTracker(
        2,
        RunUsage(estimated_cost_usd=0.40),
        estimated_usd_limit=0.50,
        per_call_cost_reserve_usd=0.20,
    )
    with pytest.raises(SpendingLimitExceeded):
        tracker.before_call()

    call_tracker = BudgetTracker(1, RunUsage())
    call_tracker.before_call()
    with pytest.raises(CallLimitExceeded):
        call_tracker.before_call()


def test_usage_cost_is_computed_only_from_profile_supplied_rates():
    usage = RunUsage()
    tracker = BudgetTracker(
        3,
        usage,
        usd_per_million_input_tokens=1.0,
        usd_per_million_output_tokens=2.0,
        usd_per_web_search_call=0.01,
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=500_000,
            output_tokens_details=SimpleNamespace(reasoning_tokens=100),
        ),
        output=[SimpleNamespace(type="web_search_call")],
    )
    tracker.record_response(response)
    assert usage.estimated_cost_usd == pytest.approx(2.01)
