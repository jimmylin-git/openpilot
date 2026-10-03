from __future__ import annotations

import math


# Matches sunnypilot's model horizon and E2E alert thresholds.
MODEL_POSITION_COUNT = 33
DEPARTURE_DISTANCE_M = 30.0
DEPARTURE_CONFIRM_SECONDS = 0.3
DEPARTURE_DISPLAY_SECONDS = 3.0
DEPARTURE_MAX_SAMPLE_GAP_SECONDS = 0.5


class DepartureReminder:
    def __init__(self) -> None:
        self._candidate_since: float | None = None
        self._last_model_time: float | None = None
        self._visible_until = 0.0
        self._consumed = False

    def update(
        self,
        now: float,
        *,
        allowed: bool,
        valid: bool,
        model_time: float,
        position_x: tuple[float, ...],
    ) -> bool:
        if not allowed:
            self._candidate_since = None
            self._last_model_time = None
            self._visible_until = 0.0
            self._consumed = False
            return False

        fresh = (
            valid
            and math.isfinite(model_time)
            and 0.0 <= now - model_time <= DEPARTURE_MAX_SAMPLE_GAP_SECONDS
            and len(position_x) == MODEL_POSITION_COUNT
            and all(math.isfinite(x) for x in position_x)
        )
        if not fresh:
            self._candidate_since = None
            self._visible_until = 0.0
            return False

        if self._last_model_time is None or model_time != self._last_model_time:
            if self._last_model_time is not None and not 0.0 < model_time - self._last_model_time <= DEPARTURE_MAX_SAMPLE_GAP_SECONDS:
                self._candidate_since = None
            self._last_model_time = model_time
            if position_x[-1] <= DEPARTURE_DISTANCE_M:
                self._candidate_since = None
                self._visible_until = 0.0
            elif not self._consumed:
                if self._candidate_since is None:
                    self._candidate_since = model_time
                elif model_time - self._candidate_since > DEPARTURE_CONFIRM_SECONDS:
                    self._consumed = True
                    self._visible_until = now + DEPARTURE_DISPLAY_SECONDS
        return now < self._visible_until
