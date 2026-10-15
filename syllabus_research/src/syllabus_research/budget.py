# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import RunUsage


class CallLimitExceeded(RuntimeError):
    pass


class SpendingLimitExceeded(RuntimeError):
    pass


@dataclass
class BudgetTracker:
    max_api_calls: int
    usage: RunUsage
    estimated_usd_limit: float | None = None
    per_call_cost_reserve_usd: float | None = None
    usd_per_million_input_tokens: float | None = None
    usd_per_million_output_tokens: float | None = None
    usd_per_web_search_call: float | None = None

    def before_call(self) -> None:
        if self.usage.api_calls >= self.max_api_calls:
            raise CallLimitExceeded(f"Configured API call limit ({self.max_api_calls}) reached.")
        if self.estimated_usd_limit is not None:
            if self.per_call_cost_reserve_usd is None:
                raise ValueError(
                    "per_call_cost_reserve_usd is required when estimated_usd_limit is set."
                )
            spent = self.usage.estimated_cost_usd or 0.0
            if spent + self.per_call_cost_reserve_usd > self.estimated_usd_limit:
                raise SpendingLimitExceeded(
                    f"Configured estimated spending limit (${self.estimated_usd_limit:.2f}) would be exceeded."
                )
        self.usage.api_calls += 1

    def record_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.usage.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None)
        self.usage.reasoning_tokens += int(getattr(details, "reasoning_tokens", 0) or 0)
        rates = (
            self.usd_per_million_input_tokens,
            self.usd_per_million_output_tokens,
            self.usd_per_web_search_call,
        )
        if all(rate is not None for rate in rates):
            web_search_calls = sum(
                getattr(item, "type", "") in {"web_search_call", "web_search"}
                for item in (getattr(response, "output", None) or [])
            )
            incremental = (
                int(getattr(usage, "input_tokens", 0) or 0) * rates[0] / 1_000_000
                + int(getattr(usage, "output_tokens", 0) or 0) * rates[1] / 1_000_000
                + web_search_calls * rates[2]
            )
            self.usage.estimated_cost_usd = (self.usage.estimated_cost_usd or 0.0) + incremental
