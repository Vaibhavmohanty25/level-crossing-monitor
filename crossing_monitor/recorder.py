"""Convert active intrusions into persistent incident records."""

from dataclasses import dataclass, replace

import cv2

from crossing_monitor.evidence import save_snapshot
from crossing_monitor.escalation import IntrusionState


@dataclass
class RecordedTrack:
    incident_id: str
    state: IntrusionState
    last_saved: float


class IncidentRecorder:
    def __init__(
        self,
        store,
        evidence_directory,
        source,
    ):
        self.store = store
        self.evidence_directory = evidence_directory
        self.source = source

        self.active: dict[int, RecordedTrack] = {}

    def snapshot(self, frame, incident_id, level):
        try:
            path = save_snapshot(
                frame,
                self.evidence_directory,
                incident_id,
                level,
            )

        except (OSError, cv2.error) as error:
            print(
                "Screenshot unavailable; "
                f"event record retained: {error}"
            )
            return

        self.store.add_snapshot(
            incident_id,
            level,
            path,
        )

    def sync(
        self,
        states,
        frame,
        closed,
        visible_ids,
        timestamp,
    ):
        # Close incidents removed by the escalation engine.
        for track_id in list(self.active):
            if track_id not in states:
                if not closed:
                    reason = "crossing_open"

                elif track_id in visible_ids:
                    reason = "zone_exit"

                else:
                    reason = "track_lost"

                self.finish(track_id, reason)

        for track_id, state in states.items():
            record = self.active.get(track_id)

            # The same tracking ID can start another episode
            # after its previous intrusion has expired.
            if (
                record is not None
                and record.state.started_at != state.started_at
            ):
                self.finish(
                    track_id,
                    "track_lost",
                )

                record = None

            is_new = record is None

            if is_new:
                incident_id = self.store.start(
                    self.source,
                    track_id,
                    state.started_at,
                )

                record = RecordedTrack(
                    incident_id,
                    replace(state),
                    timestamp,
                )

                self.active[track_id] = record

                print(
                    f"Incident started: {incident_id[:8]} | "
                    f"Track {track_id}"
                )

            level_changed = (
                record.state.level != state.level
            )

            # Save one screenshot per observed escalation stage.
            if is_new or level_changed:
                self.snapshot(
                    frame,
                    record.incident_id,
                    state.level,
                )

            # Commit at entry, escalation, and approximately
            # once per source second.
            if (
                is_new
                or level_changed
                or timestamp - record.last_saved >= 1.0
            ):
                self.store.update(
                    record.incident_id,
                    state.duration,
                    state.level,
                )

                record.last_saved = timestamp

            # Keep an independent copy because the escalation
            # engine changes its state objects in place.
            record.state = replace(state)

    def finish(self, track_id, reason):
        record = self.active[track_id]
        state = record.state

        self.store.update(
            record.incident_id,
            state.duration,
            state.level,
        )

        self.store.finish(
            record.incident_id,
            state.last_seen,
            reason,
        )

        del self.active[track_id]

        print(
            f"Incident ended: {record.incident_id[:8]} | "
            f"{state.duration:.1f}s | {reason}"
        )

    def finish_all(self, reason="monitor_stopped"):
        for track_id in list(self.active):
            self.finish(track_id, reason)