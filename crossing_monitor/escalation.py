"""Track intrusion duration and escalation independently of video or UI."""

import math
import os
from dataclasses import dataclass


@dataclass
class IntrusionState:
    started_at: float
    last_seen: float
    duration: float = 0.0
    level: str = "ENTRY"
    missing: bool = False


class EscalationMonitor:
    def __init__(
        self,
        persistent_after=3.0,
        prolonged_after=6.0,
        missing_grace=None,
    ):
        if missing_grace is None:
            missing_grace = float(
                os.getenv("CROSSING_MISSING_GRACE", "1.0")
            )

        values = (
            persistent_after,
            prolonged_after,
            missing_grace,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Thresholds must be finite.")

        if (
            not 0 < persistent_after < prolonged_after
            or missing_grace < 0
        ):
            raise ValueError(
                "Require 0 < persistent < prolonged and grace >= 0."
            )

        self.persistent_after = persistent_after
        self.prolonged_after = prolonged_after
        self.missing_grace = missing_grace

        self.states: dict[int, IntrusionState] = {}
        self.last_timestamp = -1.0

    def update(
        self,
        inside_ids,
        visible_ids,
        timestamp,
        closed,
    ):
        """Update once per frame; inside IDs must also be visible."""

        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(
                "Timestamp must be finite and nonnegative."
            )

        if timestamp < self.last_timestamp:
            raise ValueError(
                "Timestamps must not move backwards."
            )

        if not inside_ids.issubset(visible_ids):
            raise ValueError(
                "Inside IDs must be included in visible IDs."
            )

        self.last_timestamp = timestamp

        if not closed:
            self.states.clear()
            return self.states

        for track_id, state in list(self.states.items()):
            if track_id in inside_ids:
                if (
                    state.missing
                    and timestamp - state.last_seen
                    > self.missing_grace
                ):
                    del self.states[track_id]

            elif track_id in visible_ids:
                # Confirmed outside observation ends the episode.
                del self.states[track_id]

            elif (
                timestamp - state.last_seen
                > self.missing_grace
            ):
                del self.states[track_id]

            else:
                # Retain the episode briefly without advancing
                # its displayed duration while unseen.
                state.missing = True

        for track_id in inside_ids:
            if track_id not in self.states:
                self.states[track_id] = IntrusionState(
                    timestamp,
                    timestamp,
                )

            state = self.states[track_id]

            state.last_seen = timestamp
            state.duration = timestamp - state.started_at
            state.missing = False

            if state.duration >= self.prolonged_after:
                state.level = "PROLONGED"

            elif state.duration >= self.persistent_after:
                state.level = "PERSISTENT"

            else:
                state.level = "ENTRY"

        return self.states